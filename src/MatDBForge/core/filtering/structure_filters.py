"""Functions to filter structures based on various criteria."""

import inspect
import sys

import numpy as np
import rich.progress as riprg
from ase import Atoms, geometry
from ase import io as aseio
from ase.data import covalent_radii
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    natural_cutoffs,
)
from pymatgen.core import Structure as pmg_struct
from pymatgen.io.ase import AseAtomsAdaptor

import MatDBForge.core.code_utils as mdb_cud


def apply_struct_filters_mdb_db(structures, config_dict: dict):
    """
    Applies user-specified structure filters.

    Parameters
    ----------
    structures : list[ase_atoms] | InitialDatabase
        Database or list of structures to be filtered.
    config_dict : dict
        Dictionary containing user-defined filter settings from TOML.
    """
    if 'struct_filters' in config_dict:
        filter_settings = config_dict.get('struct_filters', {})

        # Gathering available filters
        available_filters: dict = get_available_filters()
        mdb_cud.custom_print(f'Available filters: {available_filters}', 'debug')

        mdb_cud.custom_print('Applying structure filters...', 'info')

        # Validate filter names
        invalid_filters = [
            filt_name
            for filt_name in filter_settings
            if filt_name not in available_filters
        ]

        if invalid_filters:
            mdb_cud.custom_print(
                f'Invalid filter names: {invalid_filters}. Omitting...'
            )
            for filt in invalid_filters:
                filter_settings.pop(filt)

        # Apply each filter to the structures
        filtered_uuids = []

        # Reformat structures to have similar structure as in the database
        if isinstance(structures, list):
            structures = [(idx, row) for idx, row in enumerate(structures)]

        for row in riprg.track(
            structures, description='Applying filters...', total=len(structures)
        ):
            # print('structure: ', type(row))
            # print('structure: ', type(structures))
            structure = AseAtomsAdaptor().get_atoms(row[1].structure)

            if structure.info.get('base') or structure.info.get('base'):
                continue

            struct_filter_results = []
            for filt_name, filt_params in filter_settings.items():
                # Retrieve function and apply it
                filter_func = available_filters[filt_name]

                if filt_name == 'duplicate_slabs':
                    surface_miller = row[1].surface_miller
                    if not surface_miller:
                        surface_miller = (0, 0, 1)
                    filt_params['miller_index'] = surface_miller

                # Apply filter function
                try:
                    result = filter_func(structure, **filt_params)
                except Exception:
                    print(f"'{filt_name}' failed for structure '{row[0]}'. Skipping...")
                if result:
                    struct_filter_results.append(result)

            if any(struct_filter_results):
                filtered_uuids.append(row[1].unique_id)

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
        return filtered_uuids


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
        print('layer_distances: ', layer_distances)
        print('layer_distances[0]: ', layer_distances[0])
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


def apply_filter_no_neighbors(struct, cov_rad_multiplier: float):
    """
    Use neighbor list to check if there are any atoms with no neighbors.

    Parameters
    ----------
    struct : ase.Atoms
        structure to check.

    Returns
    -------
    bool
        Returns `True` if there are atoms with no neighbors, `False` if otherwise.
    """
    if isinstance(struct, pmg_struct):
        struct = AseAtomsAdaptor().get_atoms(structure=struct)

    # Apply wrapping to structure copy, considering the minimum z value
    curr_struct = struct.copy()
    min_z = np.min(curr_struct.positions[:, 2])
    curr_struct.positions[:, 2] += np.abs(min_z) + np.abs(min_z) * 0.1
    curr_struct.wrap()

    conn_matr, has_disconnected_atoms, coord_nums = get_coord_nums(
        curr_struct, cov_rad_multiplier=cov_rad_multiplier
    )
    has_disconnected_neighbors = check_disconn_neighbors(conn_matr, coord_nums)

    return has_disconnected_atoms or has_disconnected_neighbors


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
    struct: Atoms,
    cov_rad_multiplier_max: float = 15.0,
    cov_rad_multiplier_min: float = 0.78,
    max_T: float = None,
    max_T_multiplier: float = 10,
) -> bool:
    """
    Check if the given structure has an unrealistic structure (explosion).
    This is done by checking if the distance betwen atoms is above or below the
    covalent radius modified by an appropiate multiplier, or by checking if the
    temperature is abnormally high, by checking a multiplier of the maximum
    MD simulation temperature.
    The function uses the ASE built-in function to compute the distances between
    atoms, and the covalent radii from the ASE data module.

    Parameters
    ----------
    struct : ase.Atoms
        Structure to check.
    cov_rad_multiplier_max : float, optional
        Multiplier for the maximum covalent radius, by default 15.0.
    cov_rad_multiplier_min : float, optional
        Multiplier for the minimum covalent radius, by default 0.78.
    max_T : float, optional
        Maximum temperature of the MD simulation, by default None.
        Necessary for the optional temperature check.
    max_T_multiplier : float, optional
        Multiplier for the maximum temperature, by default 10.
        Necessary for the optional temperature check.

    Returns
    -------
    bool
        Returns `True` if the structure is exploding, `False` if otherwise.
    """
    symbols = struct.get_chemical_symbols()
    atomic_nums = struct.get_atomic_numbers()
    num_atoms = len(symbols)

    # Create covalent radii array
    cov_radii = covalent_radii[atomic_nums]

    # Compute distance thresholds (upper triangular only)
    radii_sum_matrix = cov_radii[:, None] + cov_radii[None, :]
    min_distances = radii_sum_matrix * cov_rad_multiplier_min
    max_distances = radii_sum_matrix * cov_rad_multiplier_max

    # Compute distances using ASE built-in function (upper triangular only)
    distances = struct.get_all_distances(mic=True)

    # Check upper triangular indices only (avoids duplicate checks)
    upper_tri_idx = np.triu_indices(num_atoms, k=1)

    relevant_distances = distances[upper_tri_idx]
    relevant_min_distances = min_distances[upper_tri_idx]
    relevant_max_distances = max_distances[upper_tri_idx]

    # Quickly check violations
    below_min = relevant_distances < relevant_min_distances
    above_max = relevant_distances > relevant_max_distances
    # Early termination if any violation found
    if np.any(below_min) or np.any(above_max):
        return True

    # Check if temperature is abnormally high
    if max_T and max_T_multiplier:
        curr_struct_T = struct.info.get('md_temperature', np.nan)
        if curr_struct_T > max_T * max_T_multiplier:
            return True

    return False
