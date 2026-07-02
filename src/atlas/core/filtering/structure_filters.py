"""Functions to filter structures based on various criteria."""

import inspect
import sys

import numpy as np
import rich.progress as riprg
from ase import Atoms, geometry
from ase import io as aseio
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    natural_cutoffs,
)
from pymatgen.core import Structure as pmg_struct
from pymatgen.io.ase import AseAtomsAdaptor

import atlas.core.code_utils as atl_cud


def apply_struct_filters_atl_db(structures, config_dict: dict):
    """
    Applies user-specified structure filters.

    Parameters
    ----------
    structures : list[ase_atoms] | InitialDatabase
        Database or list of structures to be filtered.
    config_dict : dict
        Dictionary containing user-defined filter settings from TOML.

    Returns
    -------
    dict
        Dictionary with filtered structure UUIDs and breakdown by phase,
        type, modification, and filter trigger:
        {
            'filtered_uuids': list[str],
            'by_phase': {'phase_name': count, ...},
            'by_type': {'bulk': count, 'surface': count, ...},
            'by_modification': {'perturb': count, 'vacancy': count, ...},
            'by_filter': {'filter_name': count, ...},
        }
    """
    if 'struct_filters' in config_dict:
        filter_settings = config_dict.get('struct_filters', {})

        # Gathering available filters
        available_filters: dict = get_available_filters()
        atl_cud.custom_print(f'Available filters: {available_filters}', 'debug')

        atl_cud.custom_print('Applying structure filters...', 'info')

        # Validate filter names
        invalid_filters = [
            filt_name
            for filt_name in filter_settings
            if filt_name not in available_filters
        ]

        if invalid_filters:
            atl_cud.custom_print(
                f'Invalid filter names: {invalid_filters}. Omitting...'
            )
            for filt in invalid_filters:
                filter_settings.pop(filt)

        # Apply each filter to the structures
        filtered_uuids = []
        by_phase = {}
        by_type = {'bulk': 0, 'surface': 0, 'cluster': 0, 'isolated_atom': 0, 'md': 0}
        by_modification = {'perturb': 0, 'vacancy': 0, 'deformation': 0, 'none': 0}
        by_filter = {}

        def _get_phase_key(phase):
            if hasattr(phase, 'name'):
                return phase.name
            return phase

        # Reformat structures to have similar structure as in the database
        if isinstance(structures, list):
            structures = [(idx, row) for idx, row in enumerate(structures)]

        for row in riprg.track(
            structures, description='Applying filters...', total=len(structures)
        ):
            # print('structure: ', type(row))
            # print('structure: ', type(structures))
            structure = AseAtomsAdaptor().get_atoms(row[1].structure)

            if getattr(row[1], 'base', False) or getattr(
                row[1], 'isolated_atom', False
            ):
                continue

            struct_filter_results = []
            triggered_filter = None
            for filt_name, filt_params in filter_settings.items():
                # Retrieve function and apply it
                filter_func = available_filters[filt_name]

                if filt_name == 'duplicate_slabs':
                    surface_miller = row[1].surface_miller
                    if not surface_miller:
                        surface_miller = (0, 0, 1)
                    filt_params['miller_index'] = surface_miller

                if filt_name == 'layer_distance' and (
                    row[1].cluster or not row[1].surface
                ):
                    continue

                # Apply filter function
                try:
                    result = filter_func(structure, **filt_params)
                except Exception:
                    print(f"'{filt_name}' failed for structure '{row[0]}'. Skipping...")
                    continue
                if result:
                    if triggered_filter is None:
                        triggered_filter = filt_name
                    struct_filter_results.append(result)

            if any(struct_filter_results):
                filtered_uuids.append(row[1].unique_id)

                # Track removal metadata
                phase_key = _get_phase_key(row[1].phase)
                by_phase[phase_key] = by_phase.get(phase_key, 0) + 1

                if row[1].bulk:
                    by_type['bulk'] += 1
                elif row[1].surface:
                    by_type['surface'] += 1
                elif row[1].cluster:
                    by_type['cluster'] += 1
                elif row[1].isolated_atom:
                    by_type['isolated_atom'] += 1
                elif row[1].init_md:
                    by_type['md'] += 1

                if row[1].perturb:
                    by_modification['perturb'] += 1
                if row[1].vacancy:
                    by_modification['vacancy'] += 1
                if row[1].deformation:
                    by_modification['deformation'] += 1
                if not row[1].perturb and not row[1].vacancy and not row[1].deformation:
                    by_modification['none'] += 1

                if triggered_filter:
                    by_filter[triggered_filter] = by_filter.get(triggered_filter, 0) + 1

        # Get filtered structures from structures.df
        filtered_structs = [
            structures.db_struct_to_ase(row=row[1])
            for row in structures.df[
                structures.df['unique_id'].isin(filtered_uuids)
            ].iterrows()
        ]
        # Save filtered structures to file
        aseio.write('filtered_structures.xyz', filtered_structs, format='extxyz')

        # Removing filtered structures
        structures.df = structures.df[~structures.df['unique_id'].isin(filtered_uuids)]

        removal_data = {
            'filtered_uuids': filtered_uuids,
            'by_phase': by_phase,
            'by_type': by_type,
            'by_modification': by_modification,
            'by_filter': by_filter,
        }

        return removal_data


def get_max_layer_distance(struct: Atoms) -> float:
    """Get the maximum distance between layers in a structure."""
    # Get the layers and their distance with respect to the origin
    tags, levels = geometry.get_layers(atoms=struct, miller=(0, 0, 1), tolerance=0.1)

    # Compute the maximum layer height
    layer_distances = []
    for layer_index, layer_height in enumerate(levels[1:]):
        layer_height_diff = layer_height - levels[layer_index]
        layer_distances.append(layer_height_diff)

    max_layer_distance: float = np.max(layer_distances)

    return max_layer_distance


def get_coord_nums(struct: Atoms, cov_rad_multiplier: float = 1.0) -> tuple:
    """
    Get the connectivity matrix, check for disconnected atoms and get coord. numbers.

    Parameters
    ----------
    struct : Atoms
        Structure to check.

    Returns
    -------
    tuple
        Tuple containing the connectivity matrix, a boolean indicating if there are
        disconnected atoms and an array with the coordination numbers.
    """
    cutoffs: list = np.array(natural_cutoffs(struct)) * cov_rad_multiplier
    nl = NeighborList(
        cutoffs,
        skin=0.01,
        sorted=False,
        self_interaction=False,
        bothways=True,
        primitive=NewPrimitiveNeighborList,
    )

    nl.update(struct)
    conn_matr: np.array = nl.get_connectivity_matrix(sparse=False)

    # Check if there is any row in conn_matr that has only zeros
    has_disconnected_atoms: bool = np.any(np.all(conn_matr == 0, axis=1))

    # Get coordination numbers
    coord_nums = np.sum(conn_matr, axis=1)
    return conn_matr, has_disconnected_atoms, coord_nums


def apply_filter_layer_distance(struct: Atoms, max_layer_distance_ang: float) -> bool:
    """
    Evaluates whether the layer distace is above max_layer_distance_ang.

    Parameters
    ----------
    struct : ase.Atoms
        structure to check.
    max_layer_distance_ang : float
        Maximum distance between layers

    Returns
    -------
    bool
        Returns `True` if the layer distace is above max_layer_distance_ang,
        `False` if otherwise.
    """
    is_structure_wrong = False

    if struct.info.get('cluster', False) or not struct.info.get('surface', False):
        return is_structure_wrong

    if isinstance(struct, pmg_struct):
        struct = AseAtomsAdaptor().get_atoms(structure=struct)

    # Apply wrapping to structure copy, considering the minimum z value
    curr_struct = struct.copy()
    min_z = np.min(curr_struct.positions[:, 2])
    curr_struct.positions[:, 2] += np.abs(min_z) + np.abs(min_z) * 0.1
    curr_struct.wrap()

    max_dist = get_max_layer_distance(curr_struct)

    # Filtering using the max_layer_distance_ang
    if max_dist > max_layer_distance_ang:
        is_structure_wrong = True

    return is_structure_wrong


def apply_filter_duplicate_slabs(struct, tolerance: float, miller_index: tuple | list):
    """
    Detects duplicate slabs in a given structure.

    Parameters
    ----------
    atoms : ase.Atoms
        The slab structure.
    miller_index : tuple, optional
        The miller index to identify layers along a direction, by default (0, 0, 1).
    tolerance : float, optional
        The tolerance in angstroms for layer separation, by default 0.2.

    Returns
    -------
    bool
        True if duplicate slabs are detected, False otherwise.
    """
    from ase.geometry import get_layers

    # Identify slab layers
    tags, levels = get_layers(struct, miller=miller_index, tolerance=tolerance)

    # Compute distances between unique layers
    unique_levels = np.unique(levels)
    layer_distances = np.diff(unique_levels)

    # Check if there are repeated layer spacings
    if len(layer_distances) > 1:
        repeated_spacing = np.isclose(
            layer_distances, layer_distances[0], atol=tolerance
        )

        # Return true if multiple identical spacings are detected
        if np.all(repeated_spacing):
            return True

    return False  # No duplicates detected


def check_disconn_neighbors(
    conn_matr: np.array, coord_nums: np.array, min_coord: int = 3
) -> bool:
    """
    Check if there are disconnected atoms in the structure.

    Uses the connectivity matrix and the coordination numbers to check if there are
    atoms with disconnected neighbors. This is done by checking if the coordination
    number of an atom is below a certain threshold and if any of its neighbors also
    have a coordination number below the threshold.

    Parameters
    ----------
    conn_matr : np.array
        Connectivity matrix as given by the ASE NeighborList.
    coord_nums : np.array
        Array containing the coordination numbers of each atom.
    min_coord : int, optional
        Threshold for the coordination number, by default 3.

    Returns
    -------
    bool
        Whether there are disconnected atoms in the structure.
    """
    if len(coord_nums) <= min_coord:
        min_coord -= 1

    has_disconnected_neighbors: bool = False
    for at_id, coord_num in enumerate(coord_nums):
        if coord_num <= min_coord:
            conn_arr_curr_at = conn_matr[at_id]
            neig_idxs = np.where(conn_arr_curr_at == 1)[0]
            for nn_id in neig_idxs:
                neigh_coord = coord_nums[nn_id]
                if neigh_coord < min_coord:
                    has_disconnected_neighbors = True
                    break
    return has_disconnected_neighbors


def apply_filter_evaporation(struct, max_allowed_thickness: float) -> bool:
    """
    Fast geometric check to see if a slab has artificially expanded or
    an atom has evaporated into the vacuum.
    """
    z_thickness = np.ptp(struct.positions[:, 2])
    return z_thickness > max_allowed_thickness


def get_available_filters() -> dict:
    """
    Dynamically gathers all functions in the current module that start with
    'apply_filter_' and creates a dictionary mapping filter names to functions.

    Returns
    -------
    dict
        A dictionary mapping filter names (without 'apply_filter_') to their
        corresponding function objects.
    """
    # Get the current module
    current_module = sys.modules[__name__]

    # Gather all functions in the module
    functions = inspect.getmembers(current_module, inspect.isfunction)

    # Filter functions that start with 'apply_filter_'
    available_filters = {
        func_name.replace('apply_filter_', ''): func
        for func_name, func in functions
        if func_name.startswith('apply_filter_')
    }

    return available_filters


def apply_filter_exploding_structures(
    dyn=None,
    struct=None,
    max_F: float = 25.0,
    max_V: float = 2.0,
    max_T: float = None,
    max_T_multiplier: float = 10,
    T_list: list[float] = None,
    remove_positive_E: bool = False,
) -> bool:
    """Check if the given structure has an unrealistic structure (explosion)."""
    if struct is None and dyn:
        struct = dyn.atoms

    # Check energy first to be able to skip the expensive distance calculations
    # for high-energy structures
    if remove_positive_E:
        curr_energy = struct.info.get('REF_energy')
        if curr_energy is not None and curr_energy > 0:
            return True

    # Check temperature to be able to skip the expensive distance calculations
    # for high-temperature structures
    if max_T and max_T_multiplier and T_list:
        T_arr = np.array(T_list)
        # The comparison must happen inside the np.any()
        if np.any(T_arr > (max_T * max_T_multiplier)):
            return True

    # Check Forces
    if max_F is not None:
        try:
            # In ASE trajectories, this reads pre-calculated forces
            # from the SinglePointCalculator.
            # It will NOT trigger a new DFT/empirical calculation.
            forces = struct.get_forces()

            # np.abs().max() checks the highest single force
            # component (x, y, or z) on any atom.
            # This is significantly faster than calculating
            # the vector magnitude (np.linalg.norm) as in previous versions,
            # and works for finding anomalies.
            if np.abs(forces).max() > max_F:
                return True
        except Exception:
            # Failsafe: If the trajectory frame doesn't contain force data, just skip.
            pass

    # Check Velocities
    if max_V is not None:
        velocities = struct.get_velocities()
        # get_velocities() returns None if velocity data wasn't saved to the trajectory
        if velocities is not None and np.abs(velocities).max() > max_V:
            return True

    # If it passed all thermodynamic and kinetic checks, the structure is stable.
    return False


def apply_filter_exploding_structures_distances(
    dyn=None,
    struct=None,
    cutoffs_max_base: np.ndarray = None,
    cutoffs_min_base: np.ndarray = None,
    max_T: float = None,
    max_T_multiplier: float = 10,
    T_list: list[float] = None,
    remove_positive_E: bool = False,
) -> bool:
    """Check if the given structure has an unrealistic structure (explosion)."""
    if struct is None and dyn:
        struct = dyn.atoms

    # Check energy first to be able to skip the expensive distance calculations
    # for high-energy structures
    if remove_positive_E:
        curr_energy = struct.info.get('REF_energy')
        if curr_energy is not None and curr_energy > 0:
            return True

    # Check temperature to be able to skip the expensive distance calculations
    # for high-temperature structures
    if max_T and max_T_multiplier and T_list:
        T_arr = np.array(T_list)
        # The comparison must happen inside the np.any()
        if np.any(T_arr > (max_T * max_T_multiplier)):
            return True

    # If the structure passed the quick checks, we do the expensive math.
    # Calculate cell limits
    max_cell_val = np.max(struct.cell)

    # Vectorized check to cap cutoffs_max
    # Assuming cutoffs_max_base was passed in from the outer loop
    if np.all(max_cell_val > cutoffs_max_base):
        cutoffs_max = np.full_like(cutoffs_max_base, max_cell_val)
    else:
        cutoffs_max = cutoffs_max_base

    # Get the distances between atoms
    # This is the O(N^2) bottleneck.
    all_distances = struct.get_all_distances(mic=True)

    # For max distance, 0 (self-interaction) won't alter the max value,
    # so we can just compute it directly without masking.
    max_dist = np.max(all_distances, axis=0)
    if np.any(max_dist > cutoffs_max):
        return True

    # For min distance, we must ignore the diagonal (self-interactions = 0)
    # np.fill_diagonal is significantly faster than np.where()
    np.fill_diagonal(all_distances, np.inf)
    min_dist = np.min(all_distances, axis=0)

    return bool(np.any(min_dist < cutoffs_min_base))
