"""Generate clusters and structures for the database."""

import itertools as it
import logging
import tempfile

import numpy as np
import rich.progress as riprg
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.cluster.wulff import wulff_construction
from ase.data import atomic_numbers, covalent_radii, reference_states
from ase.io import read as ase_read
from ase.optimize.minimahopping import MinimaHopping
from pymatgen.core import Element, Lattice, Structure

import atlas.core.initial_db as atl_indb
import atlas.core.phase_diagram as atl_phase
import atlas.core.structure as atl_struct
import atlas.core.utils as atl_utils

logger = logging.getLogger(__name__)

# Default initial lattice size.
LATT = [20, 21, 22]


def get_element_constants(element_symbol: str) -> dict:
    """
    Retrieves the conventional lattice constants for any element,
    using ASE's and Pymatgen's data sources, automatically handling cubic,
    HCP, and other crystal systems.
    """
    try:
        # 1. Build the default primitive/standard bulk structure
        atoms = bulk(element_symbol)

        # 2. Convert it to its standard conventional Bravais lattice representations
        bravais = atoms.cell.get_bravais_lattice()

        # The bravais object stores the standardized conventional lattice parameters
        # (e.g., Pearson symbol alignments) inside its '.vars' attribute.
        lattice_vars = bravais.vars()

        # Extract 'a' and 'c' safely depending on what the crystal structure possesses
        a_param = lattice_vars.get('a', atoms.cell.cellpar()[0])

        # Will be present for HCP, tetragonal, etc.
        c_param = lattice_vars.get('c', None)

        # 3. Pymatgen data collection for dimer proxy
        pmg_element = Element(element_symbol)
        rad = getattr(pmg_element, 'covalent_radius', None)
        if rad is None:
            rad = getattr(pmg_element, 'atomic_radius', None)

        dimer_proxy = rad * 2 if rad else None

        # Build the output dictionary dynamically
        result = {
            'a': round(float(a_param), 4),
            'dimer_dist': round(float(dimer_proxy), 2) if dimer_proxy else None,
            'crystal_system': bravais.name,
        }

        # If it's a hexagonal/HCP system, include 'c' so we don't lose structural info
        if c_param is not None:
            result['c'] = round(float(c_param), 4)

        return result

    except Exception as e:
        raise ValueError(
            f"Could not resolve bulk data for '{element_symbol}': {e}"
        ) from e


# Maximum vacuum thickness, in Angs.
MAX_VAC = 12

# LJ equilibrium factor: r_eq = 2^(1/6) * sigma
_LJ_EQUILIBRIUM_FACTOR = 2 ** (1 / 6)


def _get_nearest_neighbor_distance(symbol: str) -> float:
    """Return the bulk nearest-neighbor distance (Å) for an element."""
    z = atomic_numbers[symbol]
    ref = reference_states[z]
    if ref is None:
        return covalent_radii[z] * 2.0

    a = ref.get('a', None)
    symmetry = ref.get('symmetry', '').lower()

    if a is None:
        return covalent_radii[z] * 2.0

    if symmetry == 'fcc':
        return a / np.sqrt(2)
    elif symmetry == 'bcc':
        return a * np.sqrt(3) / 2
    elif symmetry == 'hcp' or symmetry in ('sc', 'diamond'):
        return float(a)
    else:
        return covalent_radii[z] * 2.0


def _get_crystal_structure(symbol: str) -> str:
    """Return the crystal structure string ('fcc', 'bcc', 'hcp') for an element."""
    ref = reference_states[atomic_numbers[symbol]]
    if ref is not None and 'symmetry' in ref:
        return ref['symmetry'].lower()
    return 'fcc'


def _generate_wulff_cluster(symbol: str, num_atoms: int) -> Atoms:
    """
    Generate a cluster using Wulff construction, pruned/grown to exact size.

    Falls back to spherical confinement for HCP elements since
    wulff_construction only supports FCC, BCC, and SC.
    """
    crystal_structure = _get_crystal_structure(symbol)

    if crystal_structure not in ('fcc', 'bcc', 'sc'):
        logger.warning(
            "Wulff construction does not support '%s' (%s). "
            'Falling back to spherical confinement.',
            symbol,
            crystal_structure,
        )
        return _generate_spherical_cluster(symbol, num_atoms)

    if crystal_structure in ('fcc', 'sc'):
        surfaces = [(1, 1, 1), (1, 0, 0)]
        surface_energies = [1.0, 1.15]
    else:  # bcc
        surfaces = [(1, 1, 0), (1, 0, 0)]
        surface_energies = [1.0, 1.15]

    best_atoms = None
    best_diff = float('inf')

    for size_param in range(5, 50):
        try:
            cluster = wulff_construction(
                symbol=symbol,
                surfaces=surfaces,
                energies=surface_energies,
                size=size_param,
                structure=crystal_structure,
            )
        except Exception:
            continue
        diff = abs(len(cluster) - num_atoms)
        if diff < best_diff:
            best_diff = diff
            best_atoms = cluster.copy()
        if len(cluster) > num_atoms + 20:
            break

    if best_atoms is None:
        raise RuntimeError(
            f'Could not construct a Wulff shape for {symbol} '
            f'with target size {num_atoms}.'
        )

    atoms = best_atoms
    current_count = len(atoms)
    com = atoms.get_center_of_mass()

    if current_count > num_atoms:
        distances = np.linalg.norm(atoms.get_positions() - com, axis=1)
        furthest = np.argsort(distances)[::-1]
        del atoms[furthest[: current_count - num_atoms].tolist()]

    elif current_count < num_atoms:
        nn_dist = _get_nearest_neighbor_distance(symbol)
        r_nn = max(covalent_radii[atomic_numbers[symbol]] * 2.0, nn_dist)
        min_sep = r_nn * 0.8
        positions = atoms.get_positions()
        added = []
        for pos in positions:
            if len(added) >= num_atoms - len(atoms):
                break
            direction = pos - com
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            new_pos = pos + direction * r_nn
            all_pos = positions if len(added) == 0 else np.vstack([positions, added])
            if not np.any(np.linalg.norm(all_pos - new_pos, axis=1) < min_sep):
                added.append(new_pos)
        rng = np.random.default_rng()
        cluster_radius = np.max(np.linalg.norm(positions - com, axis=1)) + r_nn
        fallback_attempts = 0
        while len(added) < num_atoms - len(atoms) and fallback_attempts < 10_000:
            fallback_attempts += 1
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            radius = rng.uniform(0, cluster_radius)
            new_pos = com + direction * radius
            all_pos = np.vstack([positions] + ([added] if added else []))
            if not np.any(np.linalg.norm(all_pos - new_pos, axis=1) < min_sep):
                added.append(new_pos)
        if len(added) < num_atoms - len(atoms):
            logger.warning(
                'Could only place %d of %d extra atoms for %s cluster '
                'while respecting minimum distance of %.2f Å.',
                len(added),
                num_atoms - len(atoms),
                symbol,
                min_sep,
            )
        extra = Atoms(symbols=[symbol] * len(added), positions=added)
        atoms.extend(extra)

    return atoms


def _generate_spherical_cluster(
    symbol: str, num_atoms: int, packing_efficiency: float = 0.35
) -> Atoms:
    """Generate a cluster by random packing inside a sphere sized by bulk density."""
    z = atomic_numbers[symbol]
    r_cov = covalent_radii[z]
    nn_dist = _get_nearest_neighbor_distance(symbol)
    min_dist = max(nn_dist * 0.85, r_cov * 1.5)

    half_min = min_dist / 2.0
    vol_per_atom = (4 / 3) * np.pi * half_min**3
    sphere_radius = (
        3 * num_atoms * vol_per_atom / (4 * np.pi * packing_efficiency)
    ) ** (1 / 3)

    rng = np.random.default_rng()
    positions = []
    max_attempts = 50_000

    for _ in range(max_attempts):
        if len(positions) >= num_atoms:
            break
        pos = rng.uniform(-sphere_radius, sphere_radius, 3)
        if np.linalg.norm(pos) > sphere_radius:
            continue
        if len(positions) > 0:
            dists = np.linalg.norm(np.array(positions) - pos, axis=1)
            if np.any(dists < min_dist):
                continue
        positions.append(pos)

    if len(positions) < num_atoms:
        raise RuntimeError(
            f'Could not pack {num_atoms} {symbol} atoms into sphere '
            f'(radius={sphere_radius:.2f} Å) after {max_attempts} attempts. '
            f'Try lowering packing_efficiency (current: {packing_efficiency}).'
        )

    positions = np.array(positions[:num_atoms])
    positions -= positions.mean(axis=0)

    atoms = Atoms(symbols=[symbol] * num_atoms, positions=positions)
    return atoms


def _relax_basin_hopping(
    atoms: Atoms,
    totalsteps: int = 50,
    fmax: float = 0.05,
    lj_sigma: float | None = None,
    lj_epsilon: float = 1.0,
) -> Atoms:
    """Relax a cluster toward a global minimum using basin hopping with LJ potential.

    When lj_sigma is None, it is derived from the element's nearest-neighbor
    distance so that the LJ equilibrium matches real interatomic spacing.
    """
    atoms = atoms.copy()

    symbol = atoms.get_chemical_symbols()[0]
    nn_dist = _get_nearest_neighbor_distance(symbol)

    if lj_sigma is None:
        lj_sigma = nn_dist / _LJ_EQUILIBRIUM_FACTOR
        logger.info(
            'Auto-derived LJ sigma=%.3f Å for %s (NN distance=%.3f Å).',
            lj_sigma,
            symbol,
            nn_dist,
        )
    else:
        lj_equilibrium = lj_sigma * _LJ_EQUILIBRIUM_FACTOR
        if lj_equilibrium < nn_dist * 0.5:
            logger.warning(
                'LJ sigma=%.3f Å gives equilibrium distance %.3f Å, '
                'far below %s NN distance %.3f Å. Consider removing '
                'lj_sigma from config to auto-derive a physical value.',
                lj_sigma,
                lj_equilibrium,
                symbol,
                nn_dist,
            )

    box_size = np.ptp(atoms.get_positions()) + 15.0
    atoms.set_cell([box_size, box_size, box_size])
    atoms.center()
    atoms.calc = LennardJones(sigma=lj_sigma, epsilon=lj_epsilon)

    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = f'{tmpdir}/minima.traj'
        log_path = f'{tmpdir}/hop.log'
        hop = MinimaHopping(
            atoms,
            minima_traj=traj_path,
            logfile=log_path,
            fmax=fmax,
        )
        hop(totalsteps=totalsteps)

        all_minima = ase_read(traj_path, index=':')
        best = min(all_minima, key=lambda a: a.get_potential_energy())

    return best


def _ase_to_pymatgen_cluster(
    atoms: Atoms,
    element,
    size: int,
) -> Structure:
    """Convert ASE Atoms to a centered pymatgen Structure with a vacuum box."""
    cart_coords = atoms.get_positions()

    lattice = Lattice.from_parameters(
        a=LATT[0], b=LATT[1], c=LATT[2], alpha=90, beta=90, gamma=90
    )
    struct = Structure(
        lattice=lattice,
        coords=cart_coords,
        species=list(it.repeat(element, size)),
        coords_are_cartesian=True,
    )

    vac_thick = (
        np.min(struct.cart_coords, axis=0)
        + np.array(LATT)
        - np.max(struct.cart_coords, axis=0)
    )
    new_latt_size = (np.array(LATT) - vac_thick) + MAX_VAC
    lattice = Lattice.from_parameters(
        a=new_latt_size[0],
        b=new_latt_size[1],
        c=new_latt_size[2],
        alpha=90,
        beta=90,
        gamma=90,
    )

    struct = Structure(
        lattice=lattice,
        coords=struct.cart_coords,
        species=list(it.repeat(element, size)),
        coords_are_cartesian=True,
    )

    struct = center_structure(struct)
    return struct


def make_clean_cluster(
    indb_obj,
    size,
    phase: atl_phase.Phase,
    method: str = 'wulff',
    basin_hopping: bool = False,
    bh_totalsteps: int = 50,
    bh_fmax: float = 0.05,
    lj_sigma: float | None = None,
    lj_epsilon: float = 1.0,
):
    elem = phase.cluster_elem.symbol
    get_element_constants(elem)

    if method == 'wulff':
        atoms = _generate_wulff_cluster(elem, size)
    elif method == 'spherical':
        atoms = _generate_spherical_cluster(elem, size)
    else:
        raise ValueError(
            f"Unknown cluster method '{method}'. Use 'wulff' or 'spherical'."
        )

    if basin_hopping:
        atoms = _relax_basin_hopping(
            atoms,
            totalsteps=bh_totalsteps,
            fmax=bh_fmax,
            lj_sigma=lj_sigma,
            lj_epsilon=lj_epsilon,
        )

    if len(atoms) > 1:
        dists = atoms.get_all_distances(mic=False)
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        r_cov = covalent_radii[atomic_numbers[elem]]
        threshold = r_cov * 1.4
        if min_dist < threshold:
            logger.warning(
                'Cluster %s (%d atoms) has minimum interatomic distance '
                '%.3f Å, below threshold %.3f Å.',
                elem,
                len(atoms),
                min_dist,
                threshold,
            )

    struct = _ase_to_pymatgen_cluster(atoms, phase.cluster_elem, size)

    cluster_name = f'base_cluster_{phase.name}_{struct.formula}'
    clust_obj = atl_struct.Cluster(
        material_name=cluster_name,
        structure=struct,
        base=True,
        phase=phase,
    )

    return clust_obj


def make_clean_dimer(indb_obj, phase: atl_phase.Phase):
    curr_at_data = get_element_constants(phase.cluster_elem.symbol)

    # Dimer created by hand
    at_posc = np.array([[0, 0, 0], [0, curr_at_data['dimer_dist'], 0]])

    # Generating initial structure with a large cell size.
    # It will be used to check the cluster size and the cell siize
    # reduction needed in order to get a vacuum thickness of MAX_VAC
    lattice = Lattice.from_parameters(
        a=LATT[0], b=LATT[1], c=LATT[2], alpha=90, beta=90, gamma=90
    )
    struct = Structure(
        lattice=lattice,
        coords=at_posc,
        species=list(it.repeat(phase.cluster_elem, 2)),
        coords_are_cartesian=True,
    )

    # Vaccuum thickness for every axis,
    # considering if the cluster goes over the cell boundary.
    vac_thick = (
        np.min(struct.cart_coords, axis=0)
        + np.array(LATT)
        - np.max(struct.cart_coords, axis=0)
    )
    new_latt_size = (np.array(LATT) - vac_thick) + MAX_VAC
    lattice = Lattice.from_parameters(
        a=new_latt_size[0],
        b=new_latt_size[1],
        c=new_latt_size[2],
        alpha=90,
        beta=90,
        gamma=90,
    )

    # New structure with the desired vacuum thickness
    struct = Structure(
        lattice=lattice,
        coords=struct.cart_coords,
        species=list(it.repeat(phase.cluster_elem, 2)),
        coords_are_cartesian=True,
    )

    # Centering structure using center of mass
    struct = center_structure(struct)

    cluster_name = f'base_cluster_{phase.name}_{struct.formula}'
    clust_obj = atl_struct.Cluster(
        material_name=cluster_name,
        structure=struct,
        base=True,
        phase=phase,
    )

    return clust_obj


def center_structure(
    structure: Structure,
) -> Structure:
    """
    Move the center of mass of the structure towards the
    center of the cell.

    Parameters
    ----------
    structure : Structure
        Target structure to move to the center.

    Returns
    -------
    Structure
        Pymatgen structure containing a structure placed on the center
        of the cell, with the same attributes as the original.
    """
    # Getting the center of mass
    com = get_center_of_mass(structure)

    # Centering
    dist = np.array(structure.lattice.abc) / 2 - com
    modified_coords = structure.cart_coords + dist

    # Generating new pymatgen structure with the modifications
    new_structure = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )

    return new_structure


def bottom_structure(
    structure: Structure,
    offset: int = 2,
) -> Structure:
    """
    Move the structure towards the bottom of the cell, leaving a
    offset wide margin at the bottom.

    Parameters
    ----------
    structure : Structure
        Target structure to move to the bottom.
    offset : int, optional
        Separation to be left between the bottom of the cell
        and the structure, by default 2, in Angstrom.


    Returns
    -------
    Structure
        Pymatgen structure containing a structure placed on the bottom,
        with the same attributes as the original.
    """
    # Getting the position closest to the bottom
    bottom = np.min(structure.cart_coords)
    bottom_arr = np.zeros(shape=structure.cart_coords.shape)

    # Applying the offset
    bottom_arr[:, 2] += bottom - offset

    # Substracting the bottom position from the structure plus an offset
    modified_coords = structure.cart_coords + bottom_arr

    # Generating new pymatgen structure with the modifications
    new_structure = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )

    return new_structure


def get_center_of_mass(structure: Structure):
    """
    Get the center of mass (COM) of a given structure.

    The center of mass is computed by using:

        `COM = sum(r_i*m_i)/sum(m_i)`

    Where r_i are the coordinates for each atom and
    m_i their atomic masses.

    Parameters
    ----------
    structure : Structure
        Structure for which the COM will be found

    Returns
    -------
    np.array
        Coordinates of the COM
    """
    # Getting the atomic mass for each atom and the total mass
    atomic_mass_arr = np.array([atom.atomic_mass for atom in structure.species])
    total_mass = np.sum(atomic_mass_arr)

    # Multiplying the coordinates by the atomic mass, adding an additional empty axis
    com = np.multiply(structure.cart_coords, atomic_mass_arr[:, None])
    com = np.sum(com, axis=0)

    # Dividing by the total mass in order to get the center of mass
    com /= total_mass

    return com


def apply_replacement_cluster(
    db_obj: 'atl_indb.InitialDatabase',
    cluster: 'atl_struct.Cluster',
    phase: 'atl_phase.Phase',
    num_struct: int,
    num_repeat: int,
):
    replaced_clusters = []
    rng = np.random.default_rng()
    rnd_ratios = (phase.base_elem_comp_max - phase.base_elem_comp_min) * rng.random(
        size=num_struct
    ) + phase.base_elem_comp_min
    (other_elem,) = db_obj.phase_diagram.alloy_set - {phase.cluster_elem}

    structure_len = len(cluster.structure.species)

    for repl_ind, ratio in enumerate(rnd_ratios):
        for repeat_ind in range(num_repeat):
            n_replace = int(structure_len * ratio)
            if n_replace == 0:
                n_replace = 1

            other_elem_choices = rng.choice(
                a=structure_len,
                size=n_replace,
                replace=False,
                shuffle=True,
            )

            # Creating a new pymatgen structure using the base one as a template
            new_structure = cluster.structure.copy(sanitize=True)
            site_props_before = cluster.structure.site_properties

            # Replacing atoms in the structures
            for ind in other_elem_choices:
                new_structure.replace(ind, other_elem)

            # TODO: Instead of this, create a new structure
            # Copying site properties
            new_structure = new_structure.copy(
                sanitize=True, site_properties=site_props_before
            )

            cluster_name = (
                f'cluster_{new_structure.formula}_replacement'
                f'-{n_replace}_repeat-{repeat_ind}'
            )
            clust_obj = atl_struct.Cluster(
                material_name=cluster_name,
                structure=new_structure,
                replacement=True,
                replacement_ind=repl_ind,
                phase=phase,
            )
            replaced_clusters.append(clust_obj)

    return replaced_clusters


def apply_replacement_cluster_db(
    db_obj: 'atl_indb.InitialDatabase',
    phase: atl_phase.Phase,
    num_struct: int,
    num_repeat: int,
    similarity_check=True,
    save_in_db=True,
    max_structures=None,
):
    replaced_clusters = []
    rng = np.random.default_rng()
    rnd_ratios = (phase.base_elem_comp_max - phase.base_elem_comp_min) * rng.random(
        size=num_struct
    ) + phase.base_elem_comp_min
    (other_elem,) = db_obj.phase_diagram.alloy_set - {phase.cluster_elem}

    # Selecting only non-replaced structures from the current phase
    base_clusters = db_obj.df.loc[db_obj.df['replacement'] == False]  # noqa: E712
    if phase is not None:
        base_clusters = base_clusters.loc[base_clusters['phase'] == phase.name]

    # Ratio-first loop order ensures the replacement budget is distributed
    # evenly across all base cluster sizes instead of being exhausted by the
    # first few clusters in DataFrame order.
    base_rows = list(base_clusters.iterrows())
    _stop = False
    for repl_ind, ratio in riprg.track(
        enumerate(rnd_ratios),
        total=len(rnd_ratios),
        description='Cluster replacements...',
    ):
        if _stop:
            break
        for _row_idx, row in base_rows:
            if _stop:
                break
            cluster = row.structure
            structure_len = len(cluster.species)

            for repeat_ind in range(num_repeat):
                n_replace = int(structure_len * ratio)
                if n_replace == 0:
                    n_replace = 1

                other_elem_choices = rng.choice(
                    a=structure_len,
                    size=n_replace,
                    replace=False,
                    shuffle=True,
                )

                # Ensure we work with a plain Structure (Slab.copy lacks sanitize)
                base_struct = Structure.from_sites(cluster)
                new_structure = base_struct.copy(sanitize=True)
                site_props_before = base_struct.site_properties

                # Replacing atoms in the structures
                for ind in other_elem_choices:
                    new_structure.replace(ind, other_elem)

                new_structure = new_structure.copy(
                    sanitize=True, site_properties=site_props_before
                )

                cluster_name = (
                    f'cluster_{new_structure.formula}_replacement'
                    f'-{n_replace}_repeat-{repeat_ind}'
                )
                clust_obj = atl_struct.Cluster(
                    material_name=cluster_name,
                    structure=new_structure,
                    replacement=True,
                    replacement_ind=repl_ind,
                    phase=phase,
                )
                replaced_clusters.append(clust_obj)

                if max_structures and len(replaced_clusters) >= max_structures:
                    _stop = True
                    break

    if max_structures and len(replaced_clusters) >= max_structures:
        atl_utils.custom_print(
            f'Replacement cap reached ({len(replaced_clusters)} structures).',
            'info',
        )

    atl_utils.custom_print(
        f'Generated {len(replaced_clusters)} clusters after replacement', 'debug'
    )

    if similarity_check:
        replaced_clusters = atl_utils.similarity_check_list(
            db_obj=db_obj,
            replaced_structures=replaced_clusters,
            save_in_db=save_in_db,
        )

    if save_in_db and not similarity_check:
        atl_utils.custom_print('Saving to db...', 'debug')
        for _idx, cluster in enumerate(replaced_clusters):
            db_obj._save_row(structure=cluster)

    return replaced_clusters


def _apply_perturbation_cluster(center, row, per_idx, seed=None):
    # Applying deformation to the structure
    new_struct_perturb = atl_utils.gauss_perturb(
        center=center,
        structure=row.structure,
        seed=seed,
    )

    # Creating perturbed cluster object
    mat_str = f'{row.material_name}_perturb_gauss_{per_idx + 1}'
    clust_obj = atl_struct.Cluster(
        material_name=mat_str,
        structure=new_struct_perturb,
        replacement_ind=row.replacement_ind,
        phase=row.phase,
        perturb=True,
    )
    return clust_obj


def apply_gauss_perturb_list(
    repeat: int,
    cluster_list: list,
    center: float = 0.04,
    seed: int | None = None,
):
    perturbed_clusters = []
    rng = np.random.default_rng(seed)
    for cluster in cluster_list:
        for per_idx in range(repeat):
            perturb_seed = int(rng.integers(0, 2**32 - 1))
            clust_obj = _apply_perturbation_cluster(
                center,
                cluster,
                per_idx,
                seed=perturb_seed,
            )
            perturbed_clusters.append(clust_obj)

    return perturbed_clusters


def apply_gauss_perturb_db(
    repeat: int,
    db_obj,
    center: float = 0.04,
    max_structures=None,
    phase: 'atl_phase.Phase' = None,
    seed: int | None = None,
):
    perturbed_clusters = []

    if not isinstance(db_obj, atl_indb.InitialDatabase):
        raise TypeError(
            f"'{apply_gauss_perturb_db.__name__}' expects a ATLAS "
            f'database object, not a {type(db_obj)}.'
        )

    # Filter to current phase to avoid re-perturbing structures from other phases
    if phase is not None:
        df_iter = db_obj.df.loc[db_obj.df['phase'] == phase.name]
    else:
        df_iter = db_obj.df

    atl_utils.custom_print(f'Perturbation db_obj shape: {df_iter.shape}', 'debug')
    rng = np.random.default_rng(seed)
    _stop = False
    for _, row in riprg.track(
        df_iter.iterrows(),
        total=len(df_iter),
        description='Cluster perturbations...',
    ):
        if _stop:
            break
        for per_idx in range(repeat):
            perturb_seed = int(rng.integers(0, 2**32 - 1))
            clust_obj = _apply_perturbation_cluster(
                center,
                row,
                per_idx,
                seed=perturb_seed,
            )
            perturbed_clusters.append(clust_obj)
            if max_structures and len(perturbed_clusters) >= max_structures:
                _stop = True
                break

    if max_structures and len(perturbed_clusters) >= max_structures:
        atl_utils.custom_print(
            f'Perturbation cap reached ({len(perturbed_clusters)} structures).',
            'info',
        )

    atl_utils.custom_print(
        f'Total structures perturbed: {len(perturbed_clusters)}', 'debug'
    )

    # Saving in database
    for cluster in perturbed_clusters:
        db_obj._save_row(structure=cluster)

    return perturbed_clusters
