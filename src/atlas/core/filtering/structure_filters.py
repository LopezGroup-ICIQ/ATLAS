"""Functions to filter structures based on various criteria."""

import inspect
import pathlib as pl
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import rich.progress as riprg
from ase import Atoms, geometry
from ase import io as aseio
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    natural_cutoffs,
)
from dscribe.descriptors import SOAP
from pymatgen.core import Structure as pmg_struct
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.sparse.csgraph import connected_components

import atlas.core.code_utils as atl_cud
from atlas.core.metadata import (
    DFT_CONVERGED_KEY,
    DFT_IONIC_CONVERGED_KEY,
    DFT_IS_RELAXATION_KEY,
    DFT_SCF_CONVERGED_KEY,
)

_STRUCT_TYPES = ('bulk', 'surface', 'cluster', 'isolated_atom')

# Filters that only make sense for a given structure type. Filters not listed
# here are applied to every structure (except isolated atoms, which are
# skipped by default).
_FILTER_APPLICABILITY = {
    'layer_distance': {'surface'},
    'duplicate_slabs': {'surface'},
    'evaporation': {'surface'},
}

_MILLER_FILTERS = {'duplicate_slabs', 'layer_distance', 'evaporation'}


@dataclass
class FilterOutcome:
    """Result of applying a set of structure filters."""

    kept: list[Atoms] = field(default_factory=list)
    removed: list[Atoms] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    by_filter: dict[str, int] = field(default_factory=dict)
    first_filter: dict[str, int] = field(default_factory=dict)
    by_phase: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    by_modification: dict[str, int] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    invalid_filters: list[str] = field(default_factory=list)


@dataclass
class DuplicateOutcome:
    """Result of a SOAP duplicate search."""

    duplicate_indices: list[int] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)


def _get_struct_type(struct: Atoms) -> str:
    """Return the ATLAS structure type stored in ``Atoms.info``."""
    struct_type = struct.info.get('atl_struct_type')
    if struct_type:
        return str(struct_type).lower()

    for candidate in _STRUCT_TYPES:
        if struct.info.get(candidate):
            return candidate

    return 'unknown'


def _get_phase_key(struct: Atoms) -> str:
    """Return a normalized phase name for reporting."""
    phase = struct.info.get('phase', 'unknown')
    if hasattr(phase, 'name'):
        phase = phase.name
    return str(phase)


def _get_composition_key(struct: Atoms) -> tuple:
    """Return an exact atom-count/composition key for duplicate grouping."""
    species_counts = Counter(struct.get_chemical_symbols())
    return len(struct), tuple(sorted(species_counts.items()))


def _normalize_miller_index(miller_index) -> tuple[int, int, int]:
    """Normalize an ASE info Miller index (including legacy strings)."""
    if miller_index is None or miller_index is False:
        return (0, 0, 1)

    if isinstance(miller_index, str):
        cleaned = miller_index.strip().strip('[]()')
        parts = cleaned.replace(',', ' ').split()
        try:
            values = [int(round(float(part))) for part in parts]
        except ValueError:
            return (0, 0, 1)
        return tuple(values) if len(values) == 3 else (0, 0, 1)

    try:
        values = tuple(int(round(float(value))) for value in miller_index)
    except (TypeError, ValueError):
        return (0, 0, 1)
    return values if len(values) == 3 else (0, 0, 1)


def _get_atl_id(struct: Atoms, fallback: int) -> str:
    """Return the structure ID, with a deterministic fallback for bare Atoms."""
    atl_id = struct.info.get('atl_id')
    if atl_id is None:
        return f'index-{fallback}'
    return str(atl_id)


def coerce_to_atoms_list(structures) -> list[Atoms]:
    """Normalize supported structure containers into a list of ASE Atoms."""
    if structures is None:
        return []

    if isinstance(structures, Atoms):
        return [structures]

    atoms_list = []
    for item in structures:
        if isinstance(item, Atoms):
            atoms_list.append(item)
        elif isinstance(item, pmg_struct):
            atoms_list.append(AseAtomsAdaptor().get_atoms(item))
        elif hasattr(item, 'to_ase_atoms'):
            atoms_list.append(item.to_ase_atoms())
        elif hasattr(item, 'structure'):
            atoms_list.append(AseAtomsAdaptor().get_atoms(item.structure))
        else:
            raise TypeError(
                'Unsupported structure type for filtering: '
                f"'{type(item).__name__}'."
            )

    return atoms_list


def _prepare_filter_params(filter_name: str, params: dict, struct: Atoms) -> dict:
    """Build the keyword arguments for a single filter call."""
    filter_params = dict(params or {})

    if filter_name in _MILLER_FILTERS:
        miller_index = _normalize_miller_index(
            struct.info.get('surface_miller')
        )
        filter_params.setdefault('miller_index', miller_index)

    for key in ('cutoffs_max_base', 'cutoffs_min_base'):
        value = filter_params.get(key)
        if isinstance(value, list):
            filter_params[key] = np.asarray(value, dtype=float)

    return filter_params


def _record_structure_metadata(outcome: FilterOutcome, struct: Atoms) -> None:
    """Increment the reporting counters for one structure."""
    struct_type = _get_struct_type(struct)
    if struct_type not in outcome.by_type:
        struct_type = 'md' if struct.info.get('init_md') else 'unknown'
    outcome.by_type[struct_type] += 1

    phase_key = _get_phase_key(struct)
    outcome.by_phase[phase_key] = outcome.by_phase.get(phase_key, 0) + 1

    modified = False
    if struct.info.get('perturb'):
        outcome.by_modification['perturb'] += 1
        modified = True
    if struct.info.get('vacancy'):
        outcome.by_modification['vacancy'] += 1
        modified = True
    if struct.info.get('deformation'):
        outcome.by_modification['deformation'] += 1
        modified = True
    if not modified:
        outcome.by_modification['none'] += 1


def apply_structure_filters(
    structures: list[Atoms],
    filter_settings: dict,
    *,
    skip_base: bool = False,
    skip_isolated_atoms: bool = True,
    progress: bool = True,
) -> FilterOutcome:
    """Apply the filters declared in ``filter_settings`` to a structure list.

    Parameters
    ----------
    structures : list of ase.Atoms
        Structures to filter. Metadata is read from ``Atoms.info``.
    filter_settings : dict
        Mapping of filter names to their keyword arguments, as read from the
        ``[struct_filters]`` TOML section.
    skip_base : bool, optional
        Keep structures marked as ``base`` without filtering them.
    skip_isolated_atoms : bool, optional
        Keep isolated atoms (used as MLIP reference energies) without
        filtering them. Enabled by default because such structures have no
        neighbours and would otherwise be removed by the neighbour filter.
    progress : bool, optional
        Show a progress bar while filtering.

    Returns
    -------
    FilterOutcome
        Kept and removed structures plus per-filter and per-metadata counters.
    """
    outcome = FilterOutcome(
        by_type={key: 0 for key in (*_STRUCT_TYPES, 'md', 'unknown')},
        by_modification={
            'perturb': 0,
            'vacancy': 0,
            'deformation': 0,
            'none': 0,
        },
    )

    if not filter_settings:
        outcome.kept = list(structures)
        return outcome

    available_filters = get_available_filters()
    active_filters = {}

    for filter_name, filter_params in filter_settings.items():
        if filter_name not in available_filters:
            outcome.invalid_filters.append(filter_name)
            continue
        active_filters[filter_name] = filter_params or {}

    if outcome.invalid_filters:
        atl_cud.custom_print(
            f'Invalid filter names: {outcome.invalid_filters}. Omitting...',
            'warn',
        )

    if not active_filters:
        outcome.kept = list(structures)
        return outcome

    atl_cud.custom_print(f'Available filters: {available_filters}', 'debug')
    atl_cud.custom_print('Applying structure filters...', 'info')

    iterator = structures
    if progress and len(structures) > 0:
        iterator = riprg.track(
            structures,
            description='Applying filters...',
            total=len(structures),
        )

    for idx, struct in enumerate(iterator):
        atl_id = _get_atl_id(struct, idx)
        struct_type = _get_struct_type(struct)

        if skip_base and struct.info.get('base'):
            outcome.kept.append(struct)
            continue

        if skip_isolated_atoms and struct_type == 'isolated_atom':
            outcome.kept.append(struct)
            continue

        triggered_filters = []
        for filter_name, filter_params in active_filters.items():
            allowed_types = _FILTER_APPLICABILITY.get(filter_name)
            if allowed_types is not None and struct_type not in allowed_types:
                continue

            filter_func = available_filters[filter_name]
            kwargs = _prepare_filter_params(filter_name, filter_params, struct)

            try:
                triggered = bool(filter_func(struct=struct, **kwargs))
            except Exception as exc:
                outcome.failures.append(
                    {
                        'atl_id': atl_id,
                        'filter': filter_name,
                        'error': f'{type(exc).__name__}: {exc}',
                    }
                )
                continue

            if triggered:
                triggered_filters.append(filter_name)
                outcome.by_filter[filter_name] = (
                    outcome.by_filter.get(filter_name, 0) + 1
                )

        if triggered_filters:
            outcome.reasons[atl_id] = triggered_filters
            outcome.first_filter[triggered_filters[0]] = (
                outcome.first_filter.get(triggered_filters[0], 0) + 1
            )
            outcome.removed.append(struct)
            _record_structure_metadata(outcome, struct)
        else:
            outcome.kept.append(struct)

    return outcome


def apply_struct_filters_atl_db(
    structures,
    config_dict: dict,
    *,
    write_filtered: bool = True,
    filtered_path: str = 'filtered_structures.xyz',
    skip_base: bool = False,
    skip_isolated_atoms: bool = True,
    progress: bool = True,
) -> dict:
    """Apply user-specified structure filters to a database or a list.

    This is the backwards-compatible entry point used by the database
    generation pipeline. New code should call :func:`apply_structure_filters`
    directly when a ``list[ase.Atoms]`` is available.

    Parameters
    ----------
    structures : list[ase.Atoms] | InitialDatabase
        Database or list of structures to filter.
    config_dict : dict
        Configuration dictionary containing the ``[struct_filters]`` section.
    write_filtered : bool, optional
        Write removed structures to ``filtered_path``.
    filtered_path : str, optional
        Destination for the removed structures in extended XYZ format.
    skip_base : bool, optional
        Keep structures marked as ``base`` without filtering them.
    skip_isolated_atoms : bool, optional
        Keep isolated atoms without filtering them.
    progress : bool, optional
        Show a progress bar while filtering.

    Returns
    -------
    dict
        Dictionary with filtered structure IDs and counters. Keys:
        ``filtered_uuids``, ``by_phase``, ``by_type``, ``by_modification``,
        ``by_filter``, ``removed_structures``, ``kept_structures``, and
        ``failures``.
    """
    filter_settings = (config_dict or {}).get('struct_filters', {}) or {}
    database_obj = None
    source_indices = []

    if hasattr(structures, 'df') and hasattr(structures, 'db_struct_to_ase'):
        database_obj = structures
        source_indices = list(structures.df.index)
        atoms_list = [
            structures.db_struct_to_ase(row=row)
            for _, row in structures.df.iterrows()
        ]
    else:
        atoms_list = coerce_to_atoms_list(structures)

    outcome = apply_structure_filters(
        atoms_list,
        filter_settings,
        skip_base=skip_base,
        skip_isolated_atoms=skip_isolated_atoms,
        progress=progress,
    )

    if write_filtered and outcome.removed and filtered_path:
        filtered_path = pl.Path(filtered_path)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        aseio.write(str(filtered_path), outcome.removed, format='extxyz')

    removed_object_ids = {id(struct) for struct in outcome.removed}
    removed_positions = [
        idx
        for idx, struct in enumerate(atoms_list)
        if id(struct) in removed_object_ids
    ]
    filtered_uuids = [
        _get_atl_id(atoms_list[idx], idx) for idx in removed_positions
    ]

    if database_obj is not None:
        removed_source_indices = [source_indices[idx] for idx in removed_positions]
        database_obj.df = database_obj.df.drop(index=removed_source_indices)
        database_obj.df = database_obj.df.reset_index(drop=True)

    return {
        'filtered_uuids': filtered_uuids,
        'by_phase': outcome.by_phase,
        'by_type': outcome.by_type,
        'by_modification': outcome.by_modification,
        'by_filter': outcome.first_filter,
        'by_filter_all': outcome.by_filter,
        'removed_structures': outcome.removed,
        'kept_structures': outcome.kept,
        'failures': outcome.failures,
        'invalid_filters': outcome.invalid_filters,
    }


def _normalize_soap_output(raw_descriptors, n_structures: int) -> list[np.ndarray]:
    """Normalize dscribe SOAP output to a list of per-atom arrays."""
    if isinstance(raw_descriptors, np.ndarray):
        if raw_descriptors.ndim == 3:
            return [
                np.asarray(raw_descriptors[idx], dtype=float)
                for idx in range(n_structures)
            ]
        if raw_descriptors.ndim == 2:
            return [np.asarray(raw_descriptors, dtype=float)]

    return [np.asarray(descriptor, dtype=float) for descriptor in raw_descriptors]


def find_duplicate_structures(
    structures: list[Atoms],
    *,
    r_cut: float = 4.0,
    n_max: int = 4,
    l_max: int = 4,
    similarity_threshold: float = 0.999,
    group_by_phase: bool = True,
    n_jobs: int = -1,
    progress: bool = True,
) -> DuplicateOutcome:
    """Detect duplicate structures with SOAP average-kernel similarity.

    For two structures ``i`` and ``j`` the similarity is the normalized
    average of ``exp(-||d_a - d_b||^2)`` over all atom-descriptor pairs
    ``(a, b)``. This matches
    ``dscribe.kernels.AverageKernel(metric='rbf', gamma=1)`` while avoiding an
    ``N x N`` kernel matrix.

    Structures are grouped by phase (optional), atom count, and composition,
    then visited in input order. A structure is marked as a duplicate when it
    is similar to an earlier kept structure in the same group.

    Parameters
    ----------
    structures : list of ase.Atoms
        Structures to compare.
    r_cut : float, optional
        SOAP cutoff radius in Angstrom.
    n_max : int, optional
        Number of radial basis functions.
    l_max : int, optional
        Maximum angular momentum.
    similarity_threshold : float, optional
        Minimum similarity to consider two structures duplicates.
    group_by_phase : bool, optional
        Compare structures only within the same phase.
    n_jobs : int, optional
        Number of parallel jobs passed to dscribe.
    progress : bool, optional
        Show progress bars for descriptor generation and comparison.

    Returns
    -------
    DuplicateOutcome
        Indices of structures to remove and the matched pairs with their
        similarity values.
    """
    outcome = DuplicateOutcome()
    if len(structures) < 2:
        return outcome

    species = sorted(
        {
            symbol
            for struct in structures
            for symbol in struct.get_chemical_symbols()
        }
    )
    if not species:
        return outcome

    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sparse=False,
    )

    groups: dict[tuple, list[int]] = {}
    for idx, struct in enumerate(structures):
        phase_key = _get_phase_key(struct) if group_by_phase else None
        group_key = (phase_key, _get_composition_key(struct))
        groups.setdefault(group_key, []).append(idx)

    active_groups = [
        group_indices for group_indices in groups.values() if len(group_indices) > 1
    ]
    if not active_groups:
        return outcome

    active_indices = [
        struct_idx for group_indices in active_groups for struct_idx in group_indices
    ]

    atl_cud.custom_print(
        (
            f'Detecting duplicates in {len(active_indices)} structures with SOAP '
            f'(r_cut={r_cut}, n_max={n_max}, l_max={l_max}, '
            f'threshold={similarity_threshold}).'
        ),
        'info',
    )

    descriptors_by_index = {}
    chunk_size = 256

    with riprg.Progress(
        riprg.SpinnerColumn(),
        riprg.TextColumn('[progress.description]{task.description}'),
        riprg.BarColumn(),
        riprg.MofNCompleteColumn(),
        riprg.TimeElapsedColumn(),
        transient=True,
        disable=not progress,
    ) as progress_bar:
        descriptor_task = progress_bar.add_task(
            'Computing SOAP descriptors',
            total=len(active_indices),
        )
        comparison_task = progress_bar.add_task(
            'Comparing structures',
            total=len(active_indices),
        )

        for start_idx in range(0, len(active_indices), chunk_size):
            chunk_indices = active_indices[start_idx : start_idx + chunk_size]
            chunk_structures = [structures[idx] for idx in chunk_indices]

            raw_descriptors = soap.create(
                chunk_structures,
                n_jobs=n_jobs,
                verbose=False,
            )
            chunk_descriptors = _normalize_soap_output(
                raw_descriptors,
                len(chunk_structures),
            )

            for struct_idx, descriptor in zip(
                chunk_indices,
                chunk_descriptors,
                strict=True,
            ):
                descriptors_by_index[struct_idx] = descriptor

            progress_bar.update(descriptor_task, advance=len(chunk_indices))

        for group_indices in active_groups:
            descriptors = [descriptors_by_index[idx] for idx in group_indices]
            self_similarities = [
                _average_rbf_kernel(descriptor, descriptor)
                for descriptor in descriptors
            ]

            representative_positions = []
            representative_indices = []

            for local_idx, global_idx in enumerate(group_indices):
                descriptor = descriptors[local_idx]

                if representative_positions:
                    similarities = [
                        _normalized_average_rbf_kernel(
                            descriptors[rep_pos],
                            descriptor,
                            self_similarities[rep_pos],
                            self_similarities[local_idx],
                        )
                        for rep_pos in representative_positions
                    ]
                    best_rep = int(np.argmax(similarities))
                    best_similarity = float(similarities[best_rep])

                    if best_similarity >= similarity_threshold:
                        representative_idx = representative_indices[best_rep]
                        outcome.duplicate_indices.append(global_idx)
                        outcome.pairs.append(
                            {
                                'kept_index': representative_idx,
                                'duplicate_index': global_idx,
                                'kept_atl_id': structures[
                                    representative_idx
                                ].info.get('atl_id'),
                                'duplicate_atl_id': structures[global_idx].info.get(
                                    'atl_id'
                                ),
                                'similarity': best_similarity,
                            }
                        )
                        progress_bar.update(comparison_task, advance=1)
                        continue

                representative_positions.append(local_idx)
                representative_indices.append(global_idx)
                progress_bar.update(comparison_task, advance=1)

    atl_cud.custom_print(
        f'Duplicate search finished: {len(outcome.duplicate_indices)} '
        'duplicate(s) found.',
        'info',
    )

    return outcome


def _average_rbf_kernel(descriptor_a: np.ndarray, descriptor_b: np.ndarray) -> float:
    """Average RBF kernel between two per-atom descriptor arrays."""
    norm_a = np.sum(descriptor_a * descriptor_a, axis=1)[:, None]
    norm_b = np.sum(descriptor_b * descriptor_b, axis=1)[None, :]
    squared_distances = norm_a + norm_b - 2.0 * descriptor_a @ descriptor_b.T
    squared_distances = np.maximum(squared_distances, 0.0)
    return float(np.mean(np.exp(-squared_distances)))


def _normalized_average_rbf_kernel(
    descriptor_a: np.ndarray,
    descriptor_b: np.ndarray,
    self_similarity_a: float,
    self_similarity_b: float,
) -> float:
    """Normalized average RBF kernel, as used by dscribe AverageKernel."""
    raw_similarity = _average_rbf_kernel(descriptor_a, descriptor_b)
    normalization = np.sqrt(self_similarity_a * self_similarity_b)
    if normalization <= 0.0:
        return 0.0
    return raw_similarity / normalization


def get_max_layer_distance(
    struct: Atoms,
    miller: tuple | list = (0, 0, 1),
) -> float:
    """Get the maximum distance between layers along ``miller``."""
    _, levels = geometry.get_layers(atoms=struct, miller=miller, tolerance=0.1)
    if len(levels) < 2:
        return 0.0

    layer_distances = []
    for layer_index, layer_height in enumerate(levels[1:]):
        layer_height_diff = layer_height - levels[layer_index]
        layer_distances.append(layer_height_diff)

    return float(np.max(layer_distances))


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


def _get_connectivity(
    struct: Atoms,
    cov_rad_multiplier: float = 1.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return adjacency, coordination, component labels, counts, and LCC size."""
    if len(struct) == 0:
        return (
            np.zeros((0, 0), dtype=bool),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            0,
            0,
        )

    conn_matr, _, coord_nums = get_coord_nums(
        struct,
        cov_rad_multiplier=cov_rad_multiplier,
    )
    n_components, labels = connected_components(conn_matr, directed=False)
    largest_component = int(np.bincount(labels).max())
    return conn_matr, coord_nums, labels, n_components, largest_component


def apply_filter_no_neighbors(
    struct: Atoms,
    cov_rad_multiplier: float = 1.2,
    min_coord: int = 0,
) -> bool:
    """Detect structures containing atoms below a coordination threshold.

    Parameters
    ----------
    struct : ase.Atoms
        Structure to check.
    cov_rad_multiplier : float, optional
        Multiplier applied to the natural covalent cutoffs.
    min_coord : int, optional
        Flag an atom when its coordination number is less than or equal to
        this value. The default ``0`` preserves the original behaviour of
        flagging only atoms with no neighbours.

    Returns
    -------
    bool
        ``True`` when at least one atom has coordination <= ``min_coord``.
    """
    _, _, coord_nums = get_coord_nums(
        struct,
        cov_rad_multiplier=cov_rad_multiplier,
    )
    return bool(np.any(coord_nums <= min_coord))


def apply_filter_fragmented(
    struct: Atoms,
    cov_rad_multiplier: float = 1.1,
    min_largest_fraction: float = 0.95,
    max_components: int | None = None,
) -> bool:
    """Detect structures split into disconnected fragments.

    A first-neighbour graph is built from ASE natural covalent cutoffs. A
    structure is flagged when its largest connected component contains less
    than ``min_largest_fraction`` of the atoms, or when the number of
    components exceeds ``max_components``.

    Parameters
    ----------
    struct : ase.Atoms
        Structure to check.
    cov_rad_multiplier : float, optional
        Multiplier applied to the natural covalent cutoffs.
    min_largest_fraction : float, optional
        Minimum allowed fraction of atoms in the largest component.
    max_components : int, optional
        Optional hard limit on the number of connected components.

    Returns
    -------
    bool
        ``True`` when the structure is fragmented.
    """
    if len(struct) == 0:
        return False

    _, _, _, n_components, largest_component = _get_connectivity(
        struct,
        cov_rad_multiplier=cov_rad_multiplier,
    )
    largest_fraction = largest_component / len(struct)

    if max_components is not None and n_components > max_components:
        return True

    return largest_fraction < min_largest_fraction


def apply_filter_low_coordination(
    struct: Atoms,
    cov_rad_multiplier: float = 1.1,
    min_coord: int = 3,
    max_low_coord_fraction: float = 0.5,
) -> bool:
    """Detect structures with too many undercoordinated atoms.

    Parameters
    ----------
    struct : ase.Atoms
        Structure to check.
    cov_rad_multiplier : float, optional
        Multiplier applied to the natural covalent cutoffs.
    min_coord : int, optional
        Atoms with coordination <= this value are considered undercoordinated.
    max_low_coord_fraction : float, optional
        Maximum allowed fraction of undercoordinated atoms.

    Returns
    -------
    bool
        ``True`` when the undercoordinated fraction exceeds the threshold.
    """
    if len(struct) == 0:
        return False

    _, coord_nums, _, _, _ = _get_connectivity(
        struct,
        cov_rad_multiplier=cov_rad_multiplier,
    )
    low_coord_fraction = float(np.mean(coord_nums <= min_coord))
    return low_coord_fraction > max_low_coord_fraction


def apply_filter_dft_converged(
    struct: Atoms,
    require_scf: bool = True,
    require_ionic: bool = False,
    remove_unknown: bool = False,
) -> bool:
    """Detect structures whose DFT calculation is not converged.

    Missing convergence keys mean "unknown"; they are kept unless
    ``remove_unknown`` is set.

    Parameters
    ----------
    struct : ase.Atoms
        Structure to check.
    require_scf : bool, optional
        Require explicit SCF convergence.
    require_ionic : bool, optional
        Require explicit ionic convergence for relaxation calculations.
        Static calculations are unaffected.
    remove_unknown : bool, optional
        Also flag structures whose convergence metadata is missing.

    Returns
    -------
    bool
        ``True`` when the structure should be removed.
    """
    info = struct.info
    overall = info.get(DFT_CONVERGED_KEY)
    if overall is not None and not bool(overall):
        return True

    if require_scf:
        scf = info.get(DFT_SCF_CONVERGED_KEY)
        if scf is None:
            if remove_unknown:
                return True
        elif not bool(scf):
            return True

    if require_ionic:
        is_relaxation = info.get(DFT_IS_RELAXATION_KEY)
        if is_relaxation:
            ionic = info.get(DFT_IONIC_CONVERGED_KEY)
            if ionic is None:
                if remove_unknown:
                    return True
            elif not bool(ionic):
                return True

    return False


def apply_filter_layer_distance(
    struct: Atoms,
    max_layer_distance_ang: float,
    is_surface: bool | None = None,
    miller_index: tuple | list | None = None,
) -> bool:
    """
    Evaluates whether the layer distace is above max_layer_distance_ang.

    Parameters
    ----------
    struct : ase.Atoms
        structure to check.
    max_layer_distance_ang : float
        Maximum distance between layers
    is_surface : bool, optional
        Whether the structure is a surface. If omitted, the type is read
        from ``Atoms.info``.
    miller_index : tuple | list, optional
        Surface orientation. If omitted, it is read from
        ``Atoms.info['surface_miller']`` and defaults to ``(0, 0, 1)``.

    Returns
    -------
    bool
        Returns `True` if the layer distace is above max_layer_distance_ang,
        `False` if otherwise.
    """
    is_structure_wrong = False

    if isinstance(struct, pmg_struct):
        struct = AseAtomsAdaptor().get_atoms(structure=struct)

    if is_surface is None:
        is_surface = _get_struct_type(struct) == 'surface'

    if not is_surface:
        return is_structure_wrong

    if miller_index is None:
        miller_index = struct.info.get('surface_miller')
    miller_index = _normalize_miller_index(miller_index)

    # Wrap into the cell so that layers crossing the boundary are handled for
    # any surface orientation.
    curr_struct = struct.copy()
    curr_struct.wrap()

    max_dist = get_max_layer_distance(curr_struct, miller=miller_index)

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

    miller_index = _normalize_miller_index(miller_index)

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


def _get_surface_normal(
    struct: Atoms,
    miller_index: tuple | list | None = None,
) -> np.ndarray:
    """Return the Cartesian surface normal for a Miller index."""
    miller = np.asarray(miller_index or (0, 0, 1), dtype=float)
    reciprocal = np.asarray(struct.cell.reciprocal(), dtype=float)
    normal = miller @ reciprocal
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def apply_filter_evaporation(
    struct: Atoms,
    max_allowed_thickness: float,
    miller_index: tuple | list | None = None,
) -> bool:
    """Check whether a slab expanded or an atom evaporated into the vacuum.

    The thickness is measured along the surface normal derived from
    ``miller_index`` (or ``Atoms.info['surface_miller']``), rather than
    always along z.

    Parameters
    ----------
    struct : ase.Atoms
        Slab structure to check.
    max_allowed_thickness : float
        Maximum allowed slab thickness in Angstroms along the surface normal.
    miller_index : tuple | list, optional
        Surface orientation. Defaults to ``surface_miller`` from ``Atoms.info``
        or ``(0, 0, 1)``.

    Returns
    -------
    bool
        ``True`` when the slab is thicker than ``max_allowed_thickness``.
    """
    if miller_index is None:
        miller_index = struct.info.get('surface_miller')
    miller_index = _normalize_miller_index(miller_index)

    normal = _get_surface_normal(struct, miller_index=miller_index)
    thickness = float(np.ptp(struct.positions @ normal))
    return thickness > max_allowed_thickness


def get_available_filters() -> dict:
    """Dynamically gathers all module functions prefixed with ``apply_filter_``.

    Returns
    -------
    dict
        Dictionary mapping filter names to their function objects.
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

    # Check Forces. Prefer DFT reference forces when present, since database
    # frames usually do not carry an ASE calculator.
    if max_F is not None:
        forces = struct.arrays.get('REF_forces')
        if forces is None:
            try:
                # In ASE trajectories, this reads pre-calculated forces
                # from the SinglePointCalculator.
                # It will NOT trigger a new DFT/empirical calculation.
                forces = struct.get_forces()
            except Exception:
                # No force data available; skip this check.
                forces = None

        # np.abs().max() checks the highest single force component (x, y, or z)
        # on any atom. This is significantly faster than calculating the vector
        # magnitude (np.linalg.norm) as in previous versions.
        if forces is not None and np.abs(forces).max() > max_F:
            return True

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
