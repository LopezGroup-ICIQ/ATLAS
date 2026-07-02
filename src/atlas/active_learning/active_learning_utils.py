"""General utility functions for the active learning workflows."""

from __future__ import annotations

import ctypes
import gc
import io
import itertools as it
import logging
import os
import pickle as pkl
import tempfile
import time
import tomllib as toml
import warnings
from collections import Counter
from contextlib import contextmanager, redirect_stdout, suppress
from pathlib import Path, PosixPath
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import slugify
import torch
import wonderwords as ww
from aiida import orm
from aiida.common.log import LOG_LEVEL_REPORT
from aiida.engine import (
    calcfunction,
)
from aiida.plugins import CalculationFactory
from ase import Atoms, units
from ase.data import atomic_numbers, covalent_radii
from ase.geometry.analysis import Analysis
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import TrajectoryWriter
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.npt import NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from e3nn.util import jit
from pymatgen.io.ase import AseAtomsAdaptor
from rich.pretty import pprint as rprint
from shapely.geometry import Point, Polygon

from atlas.active_learning import conversion as atl_conv
from atlas.core import code_utils as atl_cut
from atlas.core.exceptions import MissingMandatoryParameterError
from atlas.core.filtering.structure_filters import (
    apply_filter_exploding_structures,
)
from atlas.workflows import aiida_utils as atl_aut
from atlas.workflows.aiida_utils import can_submit_calculation

# Silencing specific warnings and log messages
warnings.filterwarnings('ignore', category=UserWarning, message='.*weights_only.*')

# Force third party loggers to only show errors and critical messages
logging.getLogger('mace').setLevel(logging.ERROR)
logging.getLogger('e3nn').setLevel(logging.ERROR)


@contextmanager
def suppress_stdout():
    """Temporarily redirects standard output to the void."""
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
        yield


def flush_system_memory():
    """
    Forces Python to sweep unreferenced objects and instructs the
    Linux C-allocator to return freed memory blocks to the OS.
    Use this only after deleting massive arrays or dataframes.
    """
    # Force Python to clean up cyclically isolated objects
    gc.collect()

    # Force Linux glibc to release the RAM to the OS
    # This helps freeing up memory after some multiprocessing tasks,
    # such as SOAP with several jobs.
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.malloc_trim(0)
    except Exception as e:
        # Fails gracefully on Windows/Mac or if libc is unavailable
        logging.debug(f'malloc_trim unsupported or failed: {e}')


def check_atl_ids(atoms_list: list[Atoms]):
    """Checks for 'atl_id' key in info dicts and reports missing or repeated IDs."""
    atl_ids = []
    missing_count = 0
    missing_indices = []

    for i, atoms in enumerate(atoms_list):
        if 'atl_id' in atoms.info:
            atl_ids.append(atoms.info['atl_id'])
        else:
            missing_count += 1
            missing_indices.append(i)

    # Check for duplicates
    id_counts = Counter(atl_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    return missing_indices, duplicates


def aiida_wait_submit(
    builder, computer: orm.Computer, calc_count: int = 0, code: orm.Code | str = None
):
    # Get code label if provided
    # If the code is not provided, use the code from the builder
    if code is None:
        code_label = builder.code.label
    else:
        code_label = code if isinstance(code, str) else code.label

    # Get the calculation limit, from the computer metadata set to 0
    # if not present.
    # `atl_calc_limit` is a custom property set with:
    # computer.set_property(name='atl_calc_limit', value=366)
    try:
        calc_limit = builder.metadata.computer.metadata.get('atl_calc_limit', 0)
    except AttributeError:
        # If the builder does not have computer metadata, get the limit
        # from the computer
        calc_limit = computer.metadata.get('atl_calc_limit', 0)
    except Exception:
        # If `atl_calc_limit` is not set, set the limit to 0
        calc_limit = 0

    # Check if the calculation can be submitted
    if calc_limit == 0:
        can_submit = True
    else:
        can_submit, calc_count_sch = can_submit_calculation(
            computer=computer,
            code=code_label,
            limit=calc_limit,
        )
    atl_cut.custom_print(f'Can submit: {can_submit}.', 'debug')
    if calc_count == 0:
        calc_count = calc_count_sch

    # If the calculation cannot be submitted, wait for a minute and check again
    while not can_submit or ((calc_count + 1) >= calc_limit):
        time.sleep(30)
        can_submit, calc_count_sch = can_submit_calculation(
            computer=computer,
            code=builder.code.label,
            limit=calc_limit,
        )
        calc_count = calc_count_sch
    calc_count += 1

    return calc_count

    # REMOVE: Check the status of the calculation on the remote machine. Extremely
    # slow (and safe) but not needed for the current implementation.
    # # Wait for the calculation to get recognized by the queue manager
    # while future.get_state() == CalcJobState.SUBMITTING or not future.get_state():
    #     print('future.get_state(): ', future.get_state())
    #     time.sleep(10)


def manual_progress_display(dyn):
    print(
        f'Step: {dyn.nsteps:<6} ({dyn.nsteps / dyn.max_steps * 100:.1f} %)',
        end='\r',
    )


def md_apply_temperature_ramp(dyn, total_steps, T_start, T_end, T_list):
    """
    Function to compute the temperature ramp during ASE MD simulations.

    Parameters
    ----------
    step : int
        Current step in the MD simulation.
    total_steps : int
        Total number of steps in the MD simulation.
    T_start : float
        Initial temperature of the MD simulation.
    T_end : float
        Final temperature of the MD simulation.

    Returns
    -------
    float
        Temperature to set for the current step in the MD simulation.
    """
    # Adding current T value to the list
    if dyn.todict().get('temperature_K') is not None:
        T_list.append(dyn.todict().get('temperature_K'))
    elif dyn._temperature_K is not None:
        T_list.append(dyn._temperature_K)
    else:
        # convert eV to K
        T_list.append(dyn.temperature * 11604.5250061657)

    # Update the temperature using the ramp function
    current_temperature = T_start + (T_end - T_start) * dyn.nsteps / total_steps

    # Set the temperature in units of energy
    if hasattr(dyn, 'set_temperature'):
        dyn.set_temperature(temperature_K=current_temperature)
    else:
        dyn._temperature_K = current_temperature

    # Adding T value to info dict
    dyn.atoms.info['md_temperature'] = current_temperature


def md_coexistence_final_step_log(dyn, T_list):
    """
    Function to compute the temperature ramp during ASE MD simulations.

    Parameters
    ----------
    step : int
        Current step in the MD simulation.
    total_steps : int
        Total number of steps in the MD simulation.
    T_start : float
        Initial temperature of the MD simulation.
    T_end : float
        Final temperature of the MD simulation.

    Returns
    -------
    float
        Temperature to set for the current step in the MD simulation.
    """
    # Adding current T value to the list
    t_val = dyn.atoms.get_temperature()
    T_list.append(t_val)

    # Print the current temperature and total energy
    print(
        f'Step: {dyn.nsteps:<6} ({dyn.nsteps / dyn.max_steps * 100:.1f} %) '
        f'- Current Temperature: {t_val:.6} K ',
        f'- Total Energy: {dyn.atoms.get_total_energy():.6} eV',
        end='\r',
    )


def md_write_frame_traj(dyn, traj, stage_name: str = None):
    """
    Function to write frames to a trajectory during ASE MD simulations.

    Parameters
    ----------
    dyn : ASE MD object
        ASE MD object used to run the MD simulation.
    traj : TrajectoryWriter
        ASE trajectory object to store the MD simulation.
    stage_name : str
        Name of the MD stage.

    """
    # Write the frame to the trajectory
    REF_energy = dyn.atoms.get_potential_energy()
    REF_forces = dyn.atoms.get_forces()
    dyn.atoms.info['REF_energy'] = REF_energy
    dyn.atoms.arrays['REF_forces'] = REF_forces

    # Assign new uuid to the atoms object
    dyn.atoms.info['atl_id'] = str(uuid4())
    dyn.atoms.info['md_stage_name'] = stage_name

    traj.write(dyn.atoms, energy=REF_energy, forces=REF_forces)


def generate_descriptors(
    database: list[Atoms] | np.ndarray,
    descriptor_type: str,
    descriptor_settings: dict,
    model_path: str | Path | None = None,
    outer_average_mace: bool = False,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, list[str]]:
    """
    Wrapper function to generate descriptors for a given database.

    Allows for the generation of descriptors using different methods
    (e.g., MACE, SOAP) based on the `descriptor_type` parameter.

    Parameters
    ----------
    database : list[Atoms] | np.ndarray
        List or array of structures for which to generate descriptors.
    descriptor_type : str
        Type of descriptor to generate. Options are 'soap' or 'mace'.
    device : str, optional
        Compute device, by default 'cpu'
    dtype : str, optional
        Floating point number precision, by default 'float32'
    model_path : str, optional
        For MLIP based descriptors, the pretrained model path, by default None
    descriptor_settings : dict, optional
        Descriptor settings dictionary.

    Returns
    -------
    tuple[dict, np.ndarray, list[str]]
        A tuple containing a dictionary of descriptors and a numpy array
        of vstacked descriptors.
    """
    if descriptor_type == 'soap':
        return generate_descriptors_soap(
            database=database,
            descriptor_settings=descriptor_settings,
            verbose=verbose,
        )
    elif descriptor_type == 'mace':
        return generate_descriptors_mace(
            model_path=model_path,
            database=database,
            descriptor_settings=descriptor_settings,
            outer_average=outer_average_mace,
            verbose=verbose,
        )


def save_descriptors(
    save_path: str | Path,
    descriptor_dict: dict,
    save_format: str = 'pkl',
):
    save_path = Path(save_path).with_suffix(f'.{save_format}')

    match save_format:
        case 'pkl':
            with open(save_path, 'wb') as f:
                pkl.dump(descriptor_dict, f)
        case _:
            raise ValueError(f'Format {save_format} not supported')


# def load_descriptors(descriptor_file_path: str | Path, format: str = 'pkl'):
#     return descr_dict, descr_arr, uuids


def calculate_fps_scores_descriptor(
    selected_uuids: list[str], descriptor_dict: dict, n_to_select: int
) -> dict[str, dict]:
    """
    Calculates FPS scores for candidate structures by finding points farthest
    from an ALREADY SELECTED subset of structures.
    """
    # Prepare data for vectorized operations
    uuids = list(descriptor_dict.keys())

    # Stack the descriptors
    descriptors = np.vstack([descriptor_dict[uuid]['descriptors'] for uuid in uuids])

    # Force the array to be exactly 2D (number of structures, flattened features)
    # This makes it compatible with MACE (N, 1, 256) while leaving
    # SOAP (N, 256) unchanged.
    descriptors = descriptors.reshape(len(uuids), -1)

    total_structures = len(uuids)

    uuid_to_idx = {uuid: i for i, uuid in enumerate(uuids)}

    scores = {}
    selected_indices = np.zeros(total_structures, dtype=bool)
    min_distances = np.full(total_structures, np.inf, dtype=descriptors.dtype)

    # Initialize distances using the already selected subset
    # This guarantees new points are far from the existing cluster
    for uuid in selected_uuids:
        if uuid in uuid_to_idx:
            idx = uuid_to_idx[uuid]
            selected_indices[idx] = True
            # Update minimum distance array from this established point
            distances_to_existing = np.linalg.norm(
                descriptors - descriptors[idx], axis=1
            )
            min_distances = np.minimum(min_distances, distances_to_existing)

    # Prevent already selected points from being picked again
    min_distances[selected_indices] = -1.0

    # Main FPS expansion loop for the NEW structures
    for rank in range(n_to_select):
        # Find the unselected point farthest from the established set
        farthest_idx = np.argmax(min_distances)
        farthest_uuid = uuids[farthest_idx]

        # Record score (scaled for consistency) and distance
        scores[farthest_uuid] = {
            'score': float(total_structures - rank) / total_structures,
            'distance': min_distances[farthest_idx],
        }

        # Mark as selected
        selected_indices[farthest_idx] = True

        # Update distances based on this newly selected point
        distances_to_last = np.linalg.norm(
            descriptors - descriptors[farthest_idx], axis=1
        )
        min_distances = np.minimum(min_distances, distances_to_last)
        min_distances[selected_indices] = -1.0

    return scores


def select_structures_random(database: list[Atoms], n_structures: int) -> list[Atoms]:
    """
    Randomly select n structures from the database.

    Parameters
    ----------
    database : list[Atoms]
        List of ASE Atoms objects to select from.
    n_structures : int
        Number of structures to select.

    Returns
    -------
    list[Atoms]
        List of selected structures.
    """
    if n_structures >= len(database):
        return database.copy()

    indices = np.random.choice(len(database), size=n_structures, replace=False)
    return [database[i] for i in indices]


def select_structures_lowest_energy(
    database: list[Atoms], n_structures: int
) -> list[Atoms]:
    """
    Select the n structures with lowest energy from the database.

    Parameters
    ----------
    database : list[Atoms]
        List of ASE Atoms objects to select from.
    n_structures : int
        Number of structures to select.

    Returns
    -------
    list[Atoms]
        List of selected structures sorted by energy.
    """
    if n_structures >= len(database):
        return database.copy()

    # Sort by REF_energy and select the lowest n
    sorted_db = sorted(
        database, key=lambda x: float(x.info.get('REF_energy', float('inf')))
    )
    return sorted_db[:n_structures]


def select_structures_fps(
    candidate_db: list[Atoms],
    selected_db: list[Atoms],
    n_structures: int,
    descriptor_settings: dict,
    outer_average_mace: bool = False,
    model_path: str = None,
) -> list[Atoms]:
    """Select n structures using Farthest Point Sampling."""
    if n_structures >= len(candidate_db):
        return candidate_db.copy()

    # We need descriptors for BOTH selected and candidates to compute distances
    combined_db = selected_db + candidate_db

    descriptor_type = descriptor_settings.get('descriptor_type', 'soap')

    # (Add your MACE path checks here if needed)

    # Generate descriptors for the combined pool
    descr_dict, _, _ = generate_descriptors(
        database=combined_db,
        descriptor_type=descriptor_type,
        descriptor_settings=descriptor_settings.get('descriptor', {}),
        outer_average_mace=outer_average_mace,
        model_path=model_path,
        verbose=False,
    )

    # Extract the UUIDs of the structures we ALREADY have
    selected_uuids = [s.info['atl_id'] for s in selected_db]

    # Calculate FPS scores asking only for 'n_structures' new points
    scores = calculate_fps_scores_descriptor(
        selected_uuids=selected_uuids,
        descriptor_dict=descr_dict,
        n_to_select=n_structures,
    )

    # Extract just the newly selected candidates and apply their scores
    selected_candidates = []
    for struct in candidate_db:
        uuid = struct.info['atl_id']
        if uuid in scores:
            struct.info['fps_score'] = scores[uuid]['score']
            struct.info['fps_distance'] = scores[uuid]['distance']
            selected_candidates.append(struct)

    # Sort descending by score just to keep them ordered properly
    selected_candidates.sort(key=lambda x: x.info['fps_score'], reverse=True)

    return selected_candidates[:n_structures]


def select_structures_uncertainty(
    database: list[Atoms],
    n_structures: int,
    model_files: list,
    descriptor_settings: dict,
) -> list[Atoms]:
    """
    Select n structures with highest uncertainty based on model committee disagreement.

    Parameters
    ----------
    database : list[Atoms]
        List of ASE Atoms objects to select from.
    n_structures : int
        Number of structures to select.
    model_files : list
        List of model file paths for the committee.
    descriptor_settings : dict
        Settings for model evaluation.

    Returns
    -------
    list[Atoms]
        List of selected structures with highest uncertainty.
    """
    if n_structures >= len(database):
        return database.copy()

    from mace.calculators import MACECalculator

    device = descriptor_settings.get('device', 'cpu')
    dtype = descriptor_settings.get('dtype', 'float32')

    # Calculate energies and forces for each structure using all models
    uncertainties = []

    for struct in database:
        energies = []
        forces_norms = []

        for model_file in model_files:
            try:
                calculator = MACECalculator(
                    models=[model_file], device=device, default_dtype=dtype
                )
                struct.calc = calculator

                energy = struct.get_potential_energy()
                forces = struct.get_forces()

                energies.append(energy / len(struct))  # energy per atom
                forces_norms.append(np.mean(np.linalg.norm(forces, axis=1)))

            except Exception as e:
                # If model evaluation fails, assign low uncertainty
                struct_id = struct.info.get('atl_id', 'unknown')
                msg = f'Warning: Model evaluation failed for structure {struct_id}: {e}'
                print(msg)
                energies.append(0.0)
                forces_norms.append(0.0)

        # Calculate uncertainty as standard deviation of energies and forces
        energy_std = np.std(energies) if len(energies) > 1 else 0.0
        forces_std = np.std(forces_norms) if len(forces_norms) > 1 else 0.0

        # Combined uncertainty (weighted sum)
        uncertainty = energy_std + 0.1 * forces_std  # Weight can be adjusted
        uncertainties.append((uncertainty, struct))

    # Sort by uncertainty (descending) and select top n
    uncertainties.sort(key=lambda x: x[0], reverse=True)
    return [struct for _, struct in uncertainties[:n_structures]]


def select_structures_data_reduction(
    database: list[Atoms],
    n_structures: int,
    selection_method: str,
    descriptor_settings: dict = None,
    model_files: list = None,
    **kwargs,
) -> list[Atoms]:
    """
    Select structures from database using the specified method for data reduction.

    Parameters
    ----------
    database : list[Atoms]
        List of ASE Atoms objects to select from.
    n_structures : int
        Number of structures to select.
    selection_method : str
        Selection method: 'random', 'lowest_energy', 'fps', 'uncertainty'.
    descriptor_settings : dict, optional
        Settings for descriptor calculation (needed for fps and uncertainty).
    model_files : list, optional
        List of model files for uncertainty calculation.
    **kwargs
        Additional arguments for specific selection methods.

    Returns
    -------
    list[Atoms]
        List of selected structures.
    """
    if selection_method == 'random':
        return select_structures_random(database, n_structures)
    elif selection_method == 'lowest_energy':
        return select_structures_lowest_energy(database, n_structures)
    elif selection_method == 'fps':
        if descriptor_settings is None:
            raise ValueError('descriptor_settings required for fps selection method')
        return select_structures_fps(
            candidate_db=database,
            selected_db=[],
            n_structures=n_structures,
            descriptor_settings=descriptor_settings,
            outer_average_mace=kwargs.get('outer_average_mace', False),
            model_path=kwargs.get('model_path', None),
        )
    elif selection_method == 'uncertainty':
        if model_files is None:
            raise ValueError('model_files required for uncertainty selection method')
        return select_structures_uncertainty(
            database, n_structures, model_files, descriptor_settings
        )
    else:
        raise ValueError(f'Unknown selection method: {selection_method}')


def prepare_test_set(
    test_db_path: str, test_db_frac: float, training_db: Atoms | list[Atoms]
) -> (orm.SinglefileData, list[Atoms]):
    """
    Prepare a test set from the training database based on the provided settings.

    A test set is prepared either by reading a user provided file or by randomly
    selecting structures from the training database (Dt). Structures selected from
    Dt are then removed in order to avoid data leakage.

    Parameters
    ----------
    test_db_path: str
        Path to the user provided test set file. If None, random selection is used.
    test_db_frac : float
        Fraction of structures to select from the training database for the test set.
    training_db : Atoms | list[Atoms]
        List of ASE Atoms objects representing the training database.

    Returns
    -------
    orm.SinglefileData
        A SinglefileData object containing the test set structures.
    list[Atoms]
        The obtained test database.
    list[Atoms]
        The updated training database with test set structures removed.
    """
    # Check if a test set file must be loaded
    must_load_file = test_db_path is not None and Path(test_db_path).exists()

    # Load user provided file
    if must_load_file:
        test_db_path: Path = Path(test_db_path)
        test_db_structures = ase_read(
            filename=test_db_path,
            format='extxyz',
            index=':',
        )
        return (
            orm.SinglefileData(file=test_db_path.resolve()),
            test_db_structures,
            training_db,
        )
    else:
        # Get the test set fraction and select random structures
        n_test_structures = max(1, int(len(training_db) * test_db_frac))
        n_test_structures = min(n_test_structures, len(training_db) - 1)
        sel_struct_idx = np.random.choice(
            len(training_db), size=n_test_structures, replace=False
        )

        # Get the test structures
        test_db_structures = []
        for i in sel_struct_idx:
            if training_db[i].info.get('atl_struct_type') != 'isolated_atom':
                test_db_structures.append(training_db[i])

        # Capture output and write to buffer
        test_db_io = io.StringIO()
        with redirect_stdout(test_db_io):
            ase_write(filename='-', images=test_db_structures, format='extxyz')
        test_db_string = test_db_io.getvalue()

        # Create SinglefileData from the buffer
        test_db_file = orm.SinglefileData(
            file=io.BytesIO(str.encode(test_db_string)), filename='test_set.extxyz'
        )

        # Remove the selected test structures from the training database
        training_db = [
            struct for i, struct in enumerate(training_db) if i not in sel_struct_idx
        ]

    return test_db_file, test_db_structures, training_db


def generate_descriptors_mace(
    model_path: str,
    database,
    descriptor_settings: dict,
    outer_average: bool = False,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, list[str]]:
    from mace.calculators import MACECalculator

    if model_path is None:
        raise MissingMandatoryParameterError(
            'Missing model path for MACE descriptor generation.'
        )

    device = descriptor_settings.get('device', 'cpu')
    dtype = descriptor_settings.get('dtype', 'float32')

    is_mp_foundation = False
    is_off_foundation = False

    if isinstance(model_path, PosixPath | Path):
        model_path = str(model_path)

    with suppress_stdout():
        try:
            # Use torch.load with map_location to ensure model
            # loads on the correct device
            model_loaded = torch.load(model_path, map_location=torch.device(device))
        except RuntimeError:
            model_loaded = torch.load(model_path, map_location=torch.device('cpu'))
        except FileNotFoundError as e:
            # Check if the model path indicates a MACE foundation model
            if 'mace:mp-' in model_path:
                model_variant = model_path.split('mace:mp-')[-1]
                if model_variant in ['small', 'medium', 'large', 'medium-mpa-0']:
                    is_mp_foundation = True
                    model_loaded = model_variant
            elif 'mace:off-' in model_path:
                model_variant = model_path.split('mace:off-')[-1]
                if model_variant in ['small', 'medium', 'large']:
                    is_off_foundation = True
                    model_loaded = model_variant
            else:
                raise FileNotFoundError(
                    'Model file not found. Please provide a valid model path'
                    'or a mace foundation model name, using the following syntax:'
                    ' "mace:mp-small", "mace:off-medium", etc.'
                ) from e

        if is_mp_foundation:
            from mace.calculators import mace_mp

            calculator = mace_mp(
                model=model_loaded,
                device=device,
                default_dtype=dtype,
            )
        elif is_off_foundation:
            from mace.calculators import mace_off

            calculator = mace_off(
                model=model_loaded,
                device=device,
                default_dtype=dtype,
            )
        else:
            calculator = MACECalculator(
                models=[model_loaded], device=device, default_dtype=dtype
            )

    descriptor_dict = {}
    descriptor_list = []
    uuid_list = []

    # Getting descriptors for every structure
    tot_num_structures = len(database)

    iterable = (
        atl_cut.atl_show_progress(
            enumerate(database),
            total=tot_num_structures,
            interval=100,
            prepend='MACE:',
        )
        if verbose
        else enumerate(database)
    )

    for _, struct in iterable:
        if struct.info.get('atl_id'):
            struct_key = struct.info.get('atl_id')
        elif struct.info.get('aiida_uuid'):
            struct_key = struct.info.get('aiida_uuid')
        else:
            struct_key = str(uuid4())
            uuid_list.append(struct_key)
            struct.info['atl_id'] = struct_key

        # Creating empty lists to store the descriptors if not already present
        if descriptor_dict.get(struct_key) is None:
            descriptor_dict[struct_key] = {
                'descriptors': [],
                'latent_space': [],
            }

        # Getting the descriptors for the current structure
        curr_struct_descriptors = calculator.get_descriptors(struct)

        # By averaging all vectors we get a single vector for the whole structure
        # Similar to SOAP's "outer average"
        if outer_average:
            curr_struct_descriptors = np.mean(
                curr_struct_descriptors, axis=0, keepdims=True
            )

        descriptor_list.append(curr_struct_descriptors)

        # Appending the descriptors to the dictionary
        descriptor_dict[struct_key]['descriptors'].append(curr_struct_descriptors)

    # Generating a numpy array from the list of all descriptors, stacked
    # vertically.
    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr, uuid_list


def get_species_from_database(database: list[Atoms] | Atoms) -> list[str]:
    """
    Get the list of species from the database of structures.

    Parameters
    ----------
    database : list[Atoms]
        List of ASE Atoms objects.

    Returns
    -------
    list[str]
        List of unique species in the database.
    """
    species = set()
    if isinstance(database, list):
        for struct in database:
            species.update(struct.get_chemical_symbols())
    elif isinstance(database, Atoms):
        species.update(struct.get_chemical_symbols())
    return sorted(species)


def generate_descriptors_soap(
    database: Atoms | list[Atoms],
    descriptor_settings: dict,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, list[str]]:
    # Initializing the SOAP calculator
    from dscribe.descriptors import SOAP

    # To avoid modifying the original dict
    descriptor_settings_copy = descriptor_settings.copy()
    # Setting default SOAP parameters
    r_cut = descriptor_settings_copy.pop('r_cut', 6.0)
    n_max = int(descriptor_settings_copy.pop('n_max', 8))
    l_max = int(descriptor_settings_copy.pop('l_max', 6))
    periodic = descriptor_settings_copy.pop('periodic', True)
    average = descriptor_settings_copy.pop('average', 'off')

    # Getting the species from the database
    if 'species' in descriptor_settings:
        species = descriptor_settings_copy.pop('species')
    else:
        species = get_species_from_database(database)

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        average=average,
        l_max=l_max,
        **descriptor_settings_copy,
    )

    descriptor_dict = {}
    descriptor_list = []
    uuid_list = []

    tot_num_structures = len(database)

    # Getting descriptors for every structure
    iterable = (
        atl_cut.atl_show_progress(
            enumerate(database), total=tot_num_structures, interval=100, prepend='SOAP:'
        )
        if verbose
        else enumerate(database)
    )

    for _, struct in iterable:
        if struct.info.get('atl_id'):
            struct_key = struct.info.get('atl_id')
        # elif struct.info.get('aiida_uuid'):
        # struct_key = struct.info.get('aiida_uuid')
        else:
            struct_key = str(uuid4())
            uuid_list.append(struct_key)
            struct.info['atl_id'] = struct_key

        # Creating empty lists to store the descriptors if not already present
        if descriptor_dict.get(struct_key) is None:
            descriptor_dict[struct_key] = {
                'descriptors': [],
                'latent_space': [],
            }

        # Getting the descriptors for the current structure
        curr_struct_descriptors = soap.create(struct, n_jobs=-2)
        descriptor_list.append(curr_struct_descriptors)

        # Appending the descriptors to the dictionary
        descriptor_dict[struct_key]['descriptors'] = [curr_struct_descriptors]

    # Generating a numpy array from the list of all descriptors, stacked
    # vertically.
    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr, uuid_list


def run_mace_md_ase(
    init_conf: Atoms,
    md_params: dict,
    T_start: float,
    traj_obj: TrajectoryWriter | None,
    prepend_path: str | Path = '.',
    explode_filter_dict: dict = None,
    mode='normal',
    md_struct_list: list = None,
    enable_cueq: bool = False,
    model_name: str = None,
    stage_name: str = None,
):
    """
    Run MD simulations using ASE and MACE.

    Parameters
    ----------
    init_conf : Atoms
        Initial structure to use for the MD simulation.
    md_params : dict
        Dictionary containing the MD parameters.
    T_start : float
        Initial temperature of the MD simulation.
    traj_obj : ASE trajectory object
        ASE trajectory object to store the MD simulation.
    prepend_path : str, optional
        Path to prepend to the model path, by default None
    explode_filter: bool, optional
        Whether to apply the MD explode filter.
    mode: str
        Operation mode for this function. One of 'normal' or 'init_db'.
    enable_cueq: bool, optional
        Whether to enable the CUEQ mode for the MD simulation.
        If True, the MD simulation will be run in CUEQ mode.
        Default is False.
    model_name: str
        Name of the model to use. If None, 'curr_model.model' is used.
        Default is None.
    stage_name: str
        Name of the MD stage. Default is None.
    """
    from mace.calculators import MACECalculator

    from atlas.active_learning.md.ase_calculators import ATLSafeCalculatorWrapper

    T_multiplier = md_params.get('max_temp_multiplier', 1.0)
    T_end = T_start * T_multiplier
    timestep_ps = md_params['timestep_duration_ps']

    if not explode_filter_dict:
        explode_filter_dict = {}

    if md_params.get('langevin_friction_ps-1'):
        friction = md_params['langevin_friction_ps-1']
    else:
        friction = md_params['timestep_duration_ps'] * 100

    if md_params.get('npt_ttime_fs'):
        npt_ttime_fs = md_params.get('npt_ttime_fs', 100.0)

    if md_params.get('npt_ptime_fs'):
        npt_ptime_fs = md_params.get('npt_ptime_fs', 25.0)

    num_steps = md_params['num_steps']
    write_interval = md_params.get('md_write_interval', 1)
    thermostat = md_params.get('md_thermostat', 'langevin')

    # If sampling is to be done during MD, set the write interval
    # to the number of steps divided by the number of frames to keep
    # (this is done to avoid writing too many frames)
    # Disabled by default
    if md_params.get('sample_frames_during_md'):
        # Get the number of frames to keep
        md_duration_ps = num_steps * timestep_ps
        keep_interval_ps = md_params['al_keep_struct_every_n_ps']
        num_frames_to_keep = int(md_duration_ps / keep_interval_ps)

        # Get write interal
        write_interval = int(num_steps / num_frames_to_keep)

    md_params['T_start'] = T_start
    md_params['T_end'] = T_end
    md_params['langevin_friction_ps-1'] = friction
    md_params['write_interval'] = write_interval
    if md_params.get('log_save_interval'):
        log_interval = md_params['log_save_interval']
    else:
        log_interval = 1

    md_type = md_params.get('md_type', 'mace')

    T_list = []

    if md_type == 'mace':
        mace_foundation = md_params.get('mace_foundation')

        if mace_foundation:
            from mace.calculators import mace_mp

            nn_calculator = mace_mp(
                device=md_params.get('device', 'cpu'),
                default_dtype=md_params.get('default_dtype', 'float64'),
            )
        else:
            # Load the trained model as an ASE calculator and attach it to the
            # atoms object
            if model_name:
                model_path = Path(prepend_path) / model_name
            else:
                model_path = Path(prepend_path) / 'curr_model.model'

            nn_calculator = MACECalculator(
                model_paths=model_path,
                device=md_params.get('device', 'cpu'),
                default_dtype=md_params.get('default_dtype', 'float64'),
                enable_cueq=enable_cueq,
            )

        # Wrap the calculator in a custom calculator to check for
        # unphysical states
        wrapped_calc = ATLSafeCalculatorWrapper(
            calculator=nn_calculator,
            max_energy_threshold_per_atom=md_params.get(
                'max_energy_threshold_per_atom', 1000
            ),
        )
        init_conf.calc = wrapped_calc

    # Set the momenta corresponding to the initial temperature
    MaxwellBoltzmannDistribution(init_conf, temperature_K=T_start)

    # Zero linear momentum
    Stationary(init_conf)
    # Zero angular momentum
    ZeroRotation(init_conf)

    # Creating the log folder (if it does not exist, this applies when running
    # from a docker container)
    log_folder = Path(prepend_path) / 'logs'
    log_folder.mkdir(exist_ok=True)

    if stage_name:
        log_file_path = log_folder / f'{stage_name}_md_info-{T_start}K.log'
    else:
        log_file_path = log_folder / f'md_info-{T_start}K.log'

    match thermostat:
        case 'langevin' | 'nvt':
            # Define the Langevin dynamics
            dyn = Langevin(
                atoms=init_conf,
                # convert timestep in ps to fs
                timestep=(timestep_ps * 1000) * units.fs,
                temperature_K=T_start,
                # convert friction in ps-1 to fs-1
                friction=(friction / 1000) / units.fs,
                logfile=log_file_path,
                loginterval=log_interval,
            )
        case 'npt-melchionna':
            # Change the simulation box to remove any small numbers in the diagonal
            # of the box matrix
            from ase.geometry import cellpar_to_cell

            cellpar = init_conf.get_cell().cellpar()
            new_cell_matrix = cellpar_to_cell(cellpar)
            init_conf.set_cell(new_cell_matrix, scale_atoms=True)

            dyn = NPT(
                atoms=init_conf,
                timestep=(timestep_ps * 1000) * units.fs,
                temperature_K=T_start,
                externalstress=0,
                ttime=npt_ttime_fs * units.fs,
                pfactor=npt_ptime_fs * units.fs,
                logfile=log_file_path,
                loginterval=log_interval,
            )
        case 'npt' | 'npt-mtk':
            timestep = (timestep_ps * 1000) * units.fs
            dyn = MTKNPT(
                atoms=init_conf,
                timestep=timestep,
                temperature_K=T_start,
                pressure_au=0,
                tdamp=100 * timestep,
                pdamp=1000 * timestep,
                tchain=3,
                pchain=3,
                tloop=1,
                ploop=1,
                logfile=log_file_path,
            )

    # Handling logging of the MD parameters
    curr_logger = atl_cut.custom_print('Running MD simulation using settings:', 'info')
    rich_handler = [
        handl for handl in curr_logger.handlers if handl.name == 'atl_rich_handler'
    ]
    rich_handler = rich_handler[0] if len(rich_handler) > 0 else None
    file_handler = [
        handl for handl in curr_logger.handlers if handl.name == 'atl_file_handler'
    ]
    file_handler = file_handler[0] if len(file_handler) > 0 else None

    md_params_print = md_params.copy()
    print_stages = md_params_print.pop('stages', None)
    if print_stages:
        md_params_print['stages'] = print_stages

    if rich_handler:
        rprint(md_params_print)
    if file_handler:
        atl_cut.custom_print(f'{md_params_print}', 'none', extras={'block': 'console'})

    # Attach the thermostat function to increase the temperature
    # linearly from T_start to T_end
    dyn.attach(
        interval=1,
        function=md_apply_temperature_ramp,
        dyn=dyn,
        total_steps=num_steps,
        T_start=T_start,
        T_end=T_end,
        T_list=T_list,
    )

    if traj_obj is not None and mode == 'normal':
        dyn.attach(
            md_write_frame_traj,
            # atoms=dyn.atoms,
            # REF_energy=dyn.atoms.get_potential_energy(),
            # REF_forces=dyn.atoms.get_forces(),
            dyn=dyn,
            traj=traj_obj,
            stage_name=stage_name,
            interval=write_interval,
        )

    if explode_filter_dict.get('enable') and mode == 'normal':
        dyn.attach(
            md_stop_explode_filter,
            dyn=dyn,
            interval=int(
                num_steps * explode_filter_dict.get('explode_check_interval_perc', 0.1)
            ),
            cov_rad_multiplier_max=explode_filter_dict.get('cov_rad_multiplier_max'),
            cov_rad_multiplier_min=explode_filter_dict.get('cov_rad_multiplier_min'),
            max_T=T_end,
            max_T_multiplier=explode_filter_dict.get('max_T_multiplier', 10),
            T_list=T_list,
            remove_positive_E=explode_filter_dict.get('remove_positive_E', False),
        )

    if mode != 'normal':
        if not md_struct_list:
            md_struct_list = []

        dyn.attach(
            md_save_gen_structs,
            dyn=dyn,
            struct_list=md_struct_list,
            interval=write_interval,
        )

    # Run the MD simulation
    try:
        # Timing the MD run
        t_init = time.time()
        dyn.run(num_steps)
        t_end = time.time()
        elapsed_time = t_end - t_init

        # Get performance in ns/day
        timestep_fs = dyn.dt
        total_ns = num_steps * timestep_fs / 1e6
        performance = total_ns / (elapsed_time / 86400)

        atl_cut.custom_print(
            f'MD statistics: Runtime {elapsed_time:.2f} seconds. '
            f'Total steps: {num_steps}. '
            f'Performance {performance:.3f} ns/day. ',
            'info',
        )
    except Exception as e:
        atl_cut.custom_print(f'Error in MD simulation: {e}', 'error')

    if mode != 'normal':
        return md_struct_list


def md_stop_explode_filter(
    dyn,
    cov_rad_multiplier_min,
    cov_rad_multiplier_max,
    max_T,
    max_T_multiplier,
    T_list,
    remove_positive_E,
):
    has_exploded = apply_filter_exploding_structures(
        dyn,
        cov_rad_multiplier_min=cov_rad_multiplier_min,
        cov_rad_multiplier_max=cov_rad_multiplier_max,
        max_T=max_T,
        max_T_multiplier=max_T_multiplier,
        T_list=T_list,
        remove_positive_E=remove_positive_E,
    )
    if has_exploded:
        raise RuntimeError(f'Wrong structure in step {dyn.nsteps} :(')


def md_save_gen_structs(dyn, struct_list):
    struct_list.append(dyn.atoms.copy())


def simplify_forces_struct(forces: np.ndarray):
    # forces shape: (n_atoms, 3)

    # Getting the magnitude for the force vector (Euclidean norm)
    # Shape: (n_atoms)
    forces_std_norm = np.linalg.norm(forces, axis=1)

    # Get maximum and average forces
    forces_max = np.amax(forces_std_norm)
    forces_avg = np.average(forces_std_norm)

    return forces_max, forces_avg


def model_res_dict_to_arr(res_dict: dict, dict_type: str) -> np.ndarray:
    """Convert a dictionary of model results to a numpy array.

    Parameters
    ----------
    res_dict : dict
        Dictionary containing the model results.
    dict_type : str
        Type of the dictionary. Either "energy" or "forces".

    Returns
    -------
    np.ndarray
        Numpy array containing the model results.
    """
    res_model_list = []

    # Gathering all trajectories
    for _, res in res_dict.items():
        res_model_list.append(res)

    # Find the maximum length of the inner lists.
    # This length corresponds with the number of gathered frames.
    # All trajs should have the same number of frames. When frames
    # are missing, np.nan is used to pad their lists.
    sublist_lens = set(len(sublist) for sublist in res_model_list)
    max_len = max(sublist_lens)

    # If lists need to be padded, checking all trajs while taking into
    # account if they are energy or forces, as energies will use lists
    # and forces will use arrays.
    if len(sublist_lens) > 1:
        padded_list = []
        for sublist in res_model_list:
            # Pad the energy lists with np.nan
            if dict_type == 'energy':
                nan_list = list(it.repeat(np.nan, (max_len - len(sublist))))
                padded_sublist = list(sublist) + nan_list
                padded_list.append(padded_sublist)
            # Pad the forces arrays with (n_at, 3) arrays filled with np.nan
            elif dict_type == 'forces':
                nan_list = list(
                    it.repeat(
                        object=np.full(shape=sublist[0].shape, fill_value=np.nan),
                        times=(max_len - len(sublist)),
                    )
                )
                padded_sublist = list(sublist) + nan_list
                padded_list.append(padded_sublist)
        res_model_list = padded_list

    res_model_list = np.array(res_model_list, dtype=float)

    return res_model_list


def get_model_forces_variance(forces_dict: dict) -> np.ndarray:
    """Get the variance of the forces for each structure in the dict."""
    forces_model_list = model_res_dict_to_arr(forces_dict, dict_type='forces')
    forces_var = forces_model_list.var(axis=0)

    return forces_var


def get_model_energies_variance(energies_dict: dict) -> np.ndarray:
    """Get the variance of the energies for each structure in the dict."""
    energies_model_list = model_res_dict_to_arr(energies_dict, dict_type='energy')
    energies_var = energies_model_list.var(axis=0)

    return energies_var


def get_model_forces_std(forces_dict: dict) -> np.ndarray:
    """Get the standard deviation of the forces for each structure in the dict."""
    forces_model_list = model_res_dict_to_arr(forces_dict, dict_type='forces')

    # If there is only 1 (or 0) model, the standard deviation is 0 (no disagreement)
    if forces_model_list.shape[0] < 2:
        return np.zeros(forces_model_list.shape[1:])

    # Calculate the sample standard deviation of the forces
    # for each structure
    forces_std = np.nanstd(forces_model_list, axis=0, ddof=1)

    return forces_std


def get_model_energies_std(energies_dict: dict) -> np.ndarray:
    """Get the standard deviation of the energies for each structure in the dict."""
    # Convert the energies dict to a numpy array with the following shape:
    # (num_models, num_structures, num_frames)
    energies_model_arr: np.ndarray = model_res_dict_to_arr(
        energies_dict, dict_type='energy'
    )

    # If there is only 1 (or 0) model, the standard deviation is 0 (no disagreement)
    if energies_model_arr.shape[0] < 2:
        return np.zeros(energies_model_arr.shape[1:])

    # Calculate the sample standard deviation of the energies
    # for each structure
    energies_std = np.nanstd(energies_model_arr, axis=0, ddof=1)
    return energies_std


def load_database(path: str) -> list[Atoms]:
    """Load an extended xyz file (database) from a given path as a list of ASE Atoms."""
    database = ase_read(
        filename=path,
        format='extxyz',
        index=':',
    )
    return database


def convert_database_to_ase_atoms(
    database: list, deserialize: bool = False
) -> list[Atoms]:
    """Converts a struture list/array into containing both dicts and ase.Atoms
    into a list containing only ase.Atoms.
    """
    upd_database = []
    for struct in database:
        if isinstance(struct, dict):
            if deserialize:
                struct = aiida_serialized_ase_dict_to_atoms(struct)
            else:
                struct = Atoms.fromdict(struct)
        upd_database.append(struct)
    return upd_database


def return_code_from_settings(
    current_settings: dict,
    code_settings: dict,
    workchain: orm.Node,
    num_threads: int,
    executable_name: str,
    code_path: str,
    portable_code_label: str,
    builder,
) -> orm.Code:
    # Getting container settings
    ignore_container = code_settings.get('ignore_container', False)
    containerized = False
    if current_settings.get('code', {}).get('container'):
        container_dict = current_settings['code']['container']
    else:
        container_dict = workchain.inputs.container_settings.get_dict()

    if container_dict.get('use_container'):
        containerized = container_dict.get('use_container', False)
    if ignore_container is True:
        containerized = False

    if containerized:
        image_name = container_dict.get('image_name', '')
        engine_command = container_dict.get('engine_command', '')
        prepend_text = (
            code_settings['metadata'].get('prepend_text', '')
            + '\n'
            + container_dict.get('prepend_text', '')
            + f'\nexport OMP_NUM_THREADS={num_threads}'
        )
        code = orm.ContainerizedCode(
            computer=builder.metadata.computer,
            image_name=image_name,
            filepath_executable=executable_name,
            prepend_text=prepend_text,
            engine_command=engine_command,
        )
    else:
        prepend_text = (
            code_settings['metadata'].get('prepend_text', '')
            + '\nexport PATH=$PATH:.'
            + f'\nexport OMP_NUM_THREADS={num_threads}'
        )
        code = orm.PortableCode(
            label=portable_code_label,
            filepath_files=code_path,
            filepath_executable=executable_name,
            prepend_text=prepend_text,
        )
    return code


def select_dft_structures(struct_arr, frame_interval):
    """
    Select DFT structures using the interval given as an input of the workchain.

    Parameters
    ----------
    struct_arr : np.array
        Array containing all possible structures to compute.
    frame_interval : orm.Int
        Integer representing the interval between structures to keep.

    Returns
    -------
    np.array
        Array containing only the selected structures.
    """
    slice_step = int(len(struct_arr) * frame_interval)

    if slice_step == 0:
        slice_step = int(len(struct_arr) / 2)

    selected_dft_structs_idxs = range(len(struct_arr))[::slice_step]
    selected_dft_structs = struct_arr[::slice_step]
    selected_high_error = np.nonzero(selected_dft_structs)[0]
    selected_high_error_idxs = np.array(selected_dft_structs_idxs)[selected_high_error]

    return selected_high_error_idxs


def get_total_num_frames(len_traj, md_tstep_duration_ps, frame_interval):
    """Compute the number of frames to get from the trajectory using user input."""
    # Get total MD time in picoseconds
    total_duration_ps = len_traj * md_tstep_duration_ps

    # Get total number of frames in that time.
    # Frame interval represents every how many ps of MD simulation
    # save a frame.
    total_num_frames = np.ceil(total_duration_ps * 1 / frame_interval)

    return total_num_frames


def select_md_frames_to_keep(
    frame_interval: int,
    # total_n_frames: int,
    md_tstep_duration_ps: float,
    traj,
    steps_E_F_arr: np.array,
    forces: np.array,
):
    """Select MD frames to keep using the frame interval and total number of frames."""
    len_traj = len(traj)
    total_num_frames = get_total_num_frames(
        len_traj, md_tstep_duration_ps, frame_interval
    )

    # Choose the number of frames evenly and create a mask for the arrays
    mask = np.linspace(0, len_traj - 1, total_num_frames, dtype=int)

    # Apply the mask to traj, steps_E_F_arr and forces
    traj_sampled = traj[mask]
    steps_E_F_arr_sampled = steps_E_F_arr[mask]
    forces_sampled = forces[mask]

    return traj_sampled, steps_E_F_arr_sampled, forces_sampled


def get_dft_calc_builder_vasp(
    struct,
    row,
    calc_idx: int,
    group,
    dft_settings: dict,
):
    """Generate a aiida-vasp calculation builder for a given structure and row."""
    # The dft_settings dict corresponds to the [dft.vasp] key in the input toml.
    struct_type = row['atl_struct_type']
    struct_type = dft_settings.get('calc_type', 'single_point') + '_' + struct_type
    struct_type = atl_aut.CalcType.from_string(struct_type)

    # Gathering row information
    (curr_structure, curr_material_name, curr_unique_id, curr_phase) = (
        atl_aut.gather_calc_data_from_row(row, curr_structure=struct)
    )

    # Getting default potential mapping
    potential_mapping = atl_aut.generate_potential_mapping()

    # Updating general INCAR with calc type specific options
    specific_options = dft_settings.get(struct_type)
    if specific_options:
        specific_options = specific_options.get('incar')
        for setting, val in specific_options.items():
            dft_settings['incar'][setting] = val

    builder = atl_aut.submit_aiida_vasp_calculation(
        index=calc_idx,
        target_structure=struct,
        phase=curr_phase,
        material_name=curr_material_name,
        unique_id=curr_unique_id,
        kspacing_dict=dft_settings['kspacing'],
        calc_type=struct_type,
        queue_dict=dft_settings['queue'],
        potential_family=dft_settings['potential_family'],
        potential_mapping=potential_mapping,
        return_builder=True,
        dry_run=False,
        incar_settings_dict=dft_settings['incar'],
        group=group,
        aiida_vasp_settings=dft_settings.get('aiida_vasp', {}),
    )
    return builder


def _clean_stress_info_for_mace(atoms: Atoms) -> None:
    """Clean the stress information from ase, which leads in error for mace."""
    if hasattr(atoms, 'stress'):
        with suppress(Exception):
            delattr(atoms, 'stress')
    for key in ('stress', 'REF_stress'):
        if key in atoms.info:
            with suppress(Exception):
                del atoms.info[key]


def sampler_populate_E_and_F_list(
    structure_list: list[Atoms],
    model_file: orm.SinglefileData,
):
    from io import BytesIO

    from mace.calculators import MACECalculator

    # Load model from SinglefileData and pass it to MACE
    model_file_content = model_file.get_content(mode='rb')
    model_file_io = BytesIO(model_file_content)
    mace_model = torch.load(model_file_io)
    calc = MACECalculator(models=[mace_model])

    updated_struct_list = []

    # Calculating energies and forces for all structures
    # in the structure list using the current iteration
    # of the MLIP
    for struct in structure_list:
        # Convert from aiida-serialized dict to ASE Atoms if needed
        if isinstance(struct, dict):
            atoms = aiida_serialized_ase_dict_to_atoms(struct)
        else:
            atoms = struct

        atoms_for_mace = atoms.copy()
        _clean_stress_info_for_mace(atoms_for_mace)

        # Attach the calculator and compute E and F
        atoms_for_mace.calc = calc
        E_nn = atoms_for_mace.get_potential_energy()
        F_nn = atoms_for_mace.get_forces()
        atoms.info['curr_model_energy'] = E_nn
        atoms.arrays['curr_model_forces'] = F_nn

        updated_struct_list.append(atoms)

    return updated_struct_list


def get_dft_calc_builder_mace_list(
    struct_list: list,
    dft_settings: dict,
    container_settings: dict,
):
    """Get a MACE calculation builder for a given structure list and row."""
    updated_struct_list = []

    # Whether to use a containerized version of the evaluator
    containerized: bool = container_settings.get(
        'use_container', False
    ) and not dft_settings.get('ignore_container')

    for idx, curr_struct in enumerate(struct_list):
        curr_struct = struct_list[idx]

        # Gathering material information
        curr_material_name: str = curr_struct.info.get('struct_name')

        if not isinstance(curr_struct, Atoms):
            curr_struct = AseAtomsAdaptor().get_atoms(curr_struct)

        # If there's an E or F evaluation from the current step, save it so it can
        # be used for outlier detection.
        curr_model_forces = curr_struct.arrays.get('REF_forces', None)
        if curr_model_forces is not None:
            curr_struct.arrays['curr_model_forces'] = curr_model_forces

        curr_model_energy = curr_struct.info.get('REF_energy', None)
        if curr_model_energy is not None:
            curr_struct.info['curr_model_energy'] = curr_model_energy

        updated_struct_list.append(curr_struct)

    # Write xyz file into a string captured in the stdout,
    # write it to a temporary file.
    mace_xyz_file = gen_xyz_file_from_traj(updated_struct_list)

    # Prepare GetMACEDescriptorsCalculation
    # Generate builder
    mace_descr_calc = CalculationFactory('mace-eval')
    mace_builder = mace_descr_calc.get_builder()

    mace_builder.mace_settings_dict = dft_settings['settings']

    # Load model from absolute path
    mace_model_path = Path(dft_settings['mace_potential_path']).absolute()
    model = orm.SinglefileData(file=mace_model_path)
    mace_builder.model_file = model

    # Load structure as orm.SinglefileData
    mace_builder.configuration_to_evaluate = mace_xyz_file

    # Pass the containerized settings to the builder
    mace_builder.use_container = containerized

    if containerized:
        import os

        image_name = container_settings.get('image_name', '')
        engine_command = container_settings.get('engine_command', '')

        options_dict = dft_settings['options']
        metadata_dict = dft_settings.get('metadata', {})

        num_threads = options_dict.get('resources', {}).get(
            'num_cores_per_mpiproc', os.cpu_count()
        )
        prepend_text = (
            metadata_dict.get('prepend_text', '')
            + '\n'
            + container_settings.get('prepend_text', '')
            + f'\nexport OMP_NUM_THREADS={num_threads}'
        )
        computer = orm.load_computer(metadata_dict.get('computer', None))
        code = orm.ContainerizedCode(
            computer=computer,
            image_name=image_name,
            filepath_executable='mace_eval_configs',
            prepend_text=prepend_text,
            engine_command=engine_command,
        )
        mace_builder.code = code
    else:
        # Get code and remove from settings dict
        mace_builder.code = orm.load_code(dft_settings['options']['code_string'])

    dft_settings['options'].pop('code_string', None)

    # Load scheduler and resources options
    mace_builder.metadata.options = dft_settings['options']
    mace_builder.metadata.options.parser_name = 'mace-eval-parser'

    struct_name = curr_material_name
    mace_builder.metadata.label = struct_name

    return mace_builder


def gen_xyz_file_from_traj(struct_list):
    """Generate a temporary xyz file from a list of structures."""
    f = io.StringIO()
    with redirect_stdout(f):
        ase_write(
            filename='-',
            format='extxyz',
            images=struct_list,
        )
    xyz_string = f.getvalue()

    # Generating tmp file
    mace_xyz_file = orm.SinglefileData(
        file=io.BytesIO(str.encode(xyz_string)),
        filename='mace_structures.xyz',
    )

    return mace_xyz_file


def generate_model_name():
    """
    Generate a unique NNP model name combining random words and a number.

    This function creates a unique model name by concatenating randomly selected
    adjective, noun, and verb, followed by a random number. This combination ensures
    the generation of distinctive and memorable names suitable for labeling models
    in simulations or learning tasks.

    Returns
    -------
    str
        A string consisting of a random adjective, noun, and verb followed by a hyphen
        and a random number between 1 and 99, forming a unique model name.
    """
    r = ww.RandomWord()
    randint = np.random.randint(low=1, high=10000)

    adj = r.word(include_parts_of_speech=['adjective'])
    noun = r.word(include_parts_of_speech=['noun'])
    verb = r.word(include_parts_of_speech=['verb'])
    model_name = slugify.slugify(f'{adj}-{noun}-{verb}-{randint}'.replace(' ', '_'))

    return model_name


def get_final_db_path(result_dir_path, final_db_name, node):
    """Get the path to the final database file."""
    result_dir_path = Path(result_dir_path)
    caller_uuid = process_call_root(node) if not isinstance(node, str) else node
    curr_run_dir: Path = result_dir_path / f'run_{caller_uuid}'

    if not curr_run_dir.exists():
        curr_run_dir.mkdir()

    # Adding the final database path and the 'atl_train_db_' prefix
    # used to identify the final database.
    final_db_path = curr_run_dir / f'atl_train_db_{final_db_name}.xyz'
    return final_db_path, curr_run_dir


def get_results_dir_path(result_dir_path, node, check_temp_dir=True):
    """Get the path to the results directory."""
    result_dir_path = Path(result_dir_path)

    caller_uuid = process_call_root(node) if not isinstance(node, str) else node
    curr_run_dir: Path = result_dir_path / f'run_{caller_uuid}'

    if not curr_run_dir.exists():
        curr_run_dir.mkdir()
    if check_temp_dir and not (curr_run_dir / 'run_tmp_data').exists():
        (curr_run_dir / 'run_tmp_data').mkdir()

    return curr_run_dir


def process_call_root(process):
    """Show root process of the call stack for the given process."""
    caller = process.caller

    if caller is None:
        return process.uuid

    while True:
        next_caller = caller.caller
        if next_caller is None:
            break
        caller = next_caller

    return caller.uuid


@calcfunction
def prepare_output_dataframe(md_seed_results_df):
    """Prepare the output dataframe for the active learning workflow."""
    md_seed_results_df.index = md_seed_results_df.index.map(str)
    training_df = orm.Dict(md_seed_results_df.to_dict(orient='index'))
    return training_df


@calcfunction
def update_mace_train_settings_dict(
    settings_dict: dict,
    train_data_path: str,
    curr_model: str,
    curr_iter: int,
    db_size: int,
    containerized: orm.Bool = False,
):
    """Update the MACE training settings dictionary with the new database path."""
    if isinstance(settings_dict, orm.Dict):
        settings_dict: dict = settings_dict.get_dict()

    # Update training file path in mace train settings
    # to include the new database.
    if isinstance(train_data_path, orm.Str):
        train_data_path: Path = Path(train_data_path.value)
    elif isinstance(train_data_path, str):
        train_data_path: Path = Path(train_data_path)

    if containerized.value is True:
        # When using containerized code, the training file
        # is expected to be in the /atl_data folder inside the container
        train_data_path = Path('/atl_data') / train_data_path.name
        settings_dict['train_file'] = str(train_data_path)
        settings_dict['results_dir'] = str(Path('/atl_data') / 'results')
        settings_dict['checkpoints_dir'] = str(Path('/atl_data') / 'checkpoints')
        settings_dict['model_dir'] = str(Path('/atl_data'))
        settings_dict['log_dir'] = str(Path('/atl_data') / 'logs')
    else:
        settings_dict['train_file'] = str(train_data_path.name)

    # Updating name to include model and iteration number
    curr_name = settings_dict['name']

    # For very small datasets (testing), the batch size must be lower than the
    # database size
    if db_size < settings_dict.get('batch_size', 0):
        settings_dict['batch_size'] = db_size // 2

    if isinstance(curr_model, orm.Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, orm.Int):
        curr_iter = curr_iter.value

    settings_dict['name'] = (
        str(curr_model) + '_' + curr_name + '_al-iteration_' + str(curr_iter)
    )

    return orm.Dict(settings_dict)


@calcfunction
def create_mace_lammps_model(model_file: orm.SinglefileData):
    """
    Create a LAMMPS potential from a MACE model.

    Parameters
    ----------
    model_file : orm.SinglefileData
        A MACE model file to convert to a LAMMPS potential.

    Returns
    -------
    orm.SinglefileData
        A LAMMPS potential file generated from the MACE model.
    """
    from mace.calculators import LAMMPS_MACE

    with model_file.as_path() as model_path:
        # Loading model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model = model.double().to('cpu')
        lammps_model = LAMMPS_MACE(model)
        lammps_model_compiled = jit.compile(lammps_model)

        # Creating new path
        new_model_path = str(model_path) + '-lammps.pt'

        # Saving LAMMPS model
        lammps_model_compiled.save(new_model_path)

        return orm.SinglefileData(file=new_model_path)


@calcfunction
def check_atom_in_domain(
    concave_hull: orm.ArrayData, descriptors: orm.ArrayData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    concave_hull = concave_hull.get_array()
    descriptors = descriptors.get_array()
    polygon = Polygon(concave_hull)
    for point in descriptors:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
            all_points_in_out.append(True)
        else:
            point_outside.append(point)
            all_points_in_out.append(False)

    return orm.Dict(
        {'inside': point_inside, 'outside': point_outside, 'all': all_points_in_out}
    )


@calcfunction
def plot_concave_hull(
    concave_hull: np.ndarray,
    point_inside: np.ndarray,
    point_outside: np.ndarray,
    latent_space: np.ndarray,
    filename: str = 'concave_hull.png',
):
    # Getting arrays from ArrayData objects
    concave_hull = concave_hull.get_array()
    latent_space = latent_space.get_array()
    point_inside = point_inside.get_array()
    point_outside = point_outside.get_array()

    # Plotting the concave hull in 2D space using lines
    plt.plot(concave_hull[:, 0], concave_hull[:, 1], 'r-')
    plt.plot(
        latent_space[:, 0],
        latent_space[:, 1],
        'o',
        markersize=2,
        alpha=0.5,
        label='Descriptor in database',
        markeredgewidth=0,
        color='#b16286',
    )
    plt.plot(
        point_inside[:, 0],
        point_inside[:, 1],
        's',
        label='structure in domain',
        color='#8ec07c',
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor='#282828',
    )
    plt.plot(
        point_outside[:, 0],
        point_outside[:, 1],
        's',
        label='structure out of domain',
        color='#fb4934',
        markersize=5,
        markeredgewidth=1.5,
        markeredgecolor='#282828',
    )
    plt.title('Concave Hull')
    plt.xlabel('x')
    plt.legend()

    # Create tmp file
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        plt.savefig(f.name, dpi=300)
        plt.close()
        return {'plot': orm.SinglefileData(file=f.name, filename=filename)}


def aiida_serialized_ase_dict_to_atoms(struct_dict: dict) -> Atoms:
    """Convert a serialized Atoms dictionary to an Atoms object."""
    struct_dict['pbc'] = np.array([bool(boo) for boo in struct_dict['pbc']])

    for key, val in struct_dict.items():
        if key != 'pbc' and isinstance(val, list):
            struct_dict[key] = np.array(val)

    if 'info' in struct_dict:
        for key, val in struct_dict['info'].items():
            if key == 'REF_stress':
                if isinstance(val, (list, np.ndarray)):
                    struct_dict['info'][key] = ' '.join(
                        str(x) for x in np.array(val).flatten()
                    )
                continue
            if key != 'pbc' and isinstance(val, list):
                struct_dict['info'][key] = np.array(val)

    return Atoms.fromdict(struct_dict)


def serialize_ase(curr_s: dict | Atoms) -> dict:
    """Serialize an ASE Atoms object to a dictionary."""
    if not isinstance(curr_s, dict):
        curr_s = curr_s.todict()

    curr_s['pbc'] = [bool(boo) for boo in curr_s['pbc']]

    for key, val in curr_s.items():
        if key != 'forces' and isinstance(val, np.ndarray):
            curr_s[key] = list(val)
        # Check for inf or nan values in key initial_magmoms
        # and convert them to 0.0
        if key == 'initial_magmoms':
            curr_s[key] = np.nan_to_num(val, copy=False)
            curr_s[key] = list(curr_s[key])

    return curr_s


@calcfunction
def prepare_output_final_training_db(training_db_path):
    """Convert the training database to a orm.SinglefileData object."""
    train_db = orm.SinglefileData(file=training_db_path.value)
    return train_db


@calcfunction
def gather_dft_calcs_vasp(dft_calc_list: list) -> orm.List:
    """
    Collect and preprocess VASP DFT calculation results for active learning input.

    This function takes a list of DFT calculation nodes, extracts the calculation
    results, and processes these results into a format suitable for active learning
    input. Specifically, it converts VASP runs into ASE Atoms objects and collects
    additional calculation data like forces. It also augments the Atoms objects with
    metadata necessary for the active learning workflow. Failed calculations are skipped
    ensuring that only successfully completed CalcJobs are included.
    The function returns a list of serialized ASE Atoms objects, ready for inclusion
    in the active learning database.

    Parameters
    ----------
    dft_calc_list : list
        A list of identifiers for completed DFT calculation nodes.

    Returns
    -------
    orm.List
        An AiiDA orm.List object containing serialized ASE Atoms objects,
        each representing a completed DFT calculation augmented with necessary
        metadata and calculation results.

    Notes
    -----
    - The ASE Atoms objects are serialized to ensure compatibility with AiiDA's data
    storage and manipulation frameworks.
    - Extra care is taken to include forces (and optionally, stress) in the Atoms
    objects, as these are critical for many active learning applications but
    are not included by default in the extxyz format's `Properties` tag.
    - Skips any DFT calculations that encountered errors.
    """
    vasprun_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        finished_dft_calc = orm.load_node(finished_dft_calc)

        try:
            # Gathering the vasprun as an ASE Atoms object. This object won't
            # collect automatically all of the extra information such as forces
            # or energies, and must be collected using methods from ase.calc.u
            vasprun: Atoms = atl_conv._gather_mace_req_calc_data_from_node(
                finished_dft_calc
            )

            # Gathering extra DFT calculation information from vasprun.xml
            calc_info_dict = atl_conv.gather_calc_data_from_node(
                finished_dft_calc, units='mace'
            )

        except Exception:
            # If the calculation fails for any reason, skip this calculation
            continue

        # Adding forces manually as an array into the atoms object.
        # This is needed for the atoms object to be able to include the forces in the
        # extxyz format `Properties` tag.
        if 'forces' not in vasprun.arrays:
            vasprun.new_array(
                name='forces',
                a=np.array(calc_info_dict['forces']),
            )

        # Adding the type of structure to the atoms.info dict
        struct_type = atl_conv.get_struct_type(
            vasprun=vasprun, dft_calc_node=finished_dft_calc
        )
        calc_info_dict['atl_struct_type'] = struct_type
        vasprun = vasprun_add_info_dict(vasprun, calc_info_dict)

        # Generate a structure name and gathering the aiida_uuid
        vasprun: Atoms = atl_conv._add_entry_to_mace_input(
            vasprun=vasprun,
            node=finished_dft_calc,
            to_file=False,
            remove_dipole=True,
            remove_stress=False,
        )

        if 'energy' in vasprun.info:
            vasprun.info['REF_energy'] = vasprun.info.pop('energy')
        elif vasprun.calc:
            vasprun.info['REF_energy'] = vasprun.calc.get_potential_energy()

        if 'forces' in vasprun.arrays:
            vasprun.arrays['REF_forces'] = vasprun.arrays.pop('forces')
        elif vasprun.calc:
            vasprun.arrays['REF_forces'] = vasprun.calc.get_forces()

        if 'stress' in vasprun.arrays:
            vasprun.info['REF_stress'] = vasprun.arrays.pop('stress').tolist()
        elif 'stress' in vasprun.info:
            vasprun.info['REF_stress'] = vasprun.info.pop('stress')
        elif vasprun.calc:
            # another possible way could be:
            # vasprun.get_stress(voigt=False).reshape(9)
            # to get a str-like representation in extxyz
            vasprun.info['REF_stress'] = vasprun.calc.get_stress(voigt=False).tolist()

        vasprun: dict = serialize_ase(vasprun)

        vasprun_list.append(vasprun)

    # TODO: Checking for outliers
    # result_list, outlier_list = get_outliers_from_calc_list(
    #     curr_struct_res, result_list, outlier_list
    # )

    return_list = orm.List([val for val in vasprun_list])
    return return_list


def remove_isolated_atoms(
    train_db,
    E_dft_list_per_at,
    E_nn_list_per_at,
    F_dft_list_per_at,
    F_nn_list_per_at,
    E_diff_list_meV,
    F_diff_list_meV,
):
    """Remove isolated atoms from the training database and associated lists.
    This function identifies isolated atoms in the training database and
    removes them from the database and associated lists. It returns the
    updated training database and lists without the isolated atoms.
    """
    # Identify IsolatedAtom structures from the database
    # so they can be removed when plotting
    isolated_atom_idxs = [
        idx
        for idx, atoms in enumerate(train_db)
        if atoms.info.get('atl_struct_type', '') == 'isolated_atom'
        or atoms.info.get('phase', '').lower() == 'isolatedatom'
    ]

    # Create a set of these indices for efficient lookup
    isolated_atom_idxs_set = set(isolated_atom_idxs)

    # Build new lists excluding elements at the identified indices
    new_train_db = [
        atoms for idx, atoms in enumerate(train_db) if idx not in isolated_atom_idxs_set
    ]
    new_E_dft_list_per_at = [
        val
        for idx, val in enumerate(E_dft_list_per_at)
        if idx not in isolated_atom_idxs_set
    ]
    new_E_nn_list_per_at = [
        val
        for idx, val in enumerate(E_nn_list_per_at)
        if idx not in isolated_atom_idxs_set
    ]
    new_F_dft_list_per_at = [
        val
        for idx, val in enumerate(F_dft_list_per_at)
        if idx not in isolated_atom_idxs_set
    ]
    new_F_nn_list_per_at = [
        val
        for idx, val in enumerate(F_nn_list_per_at)
        if idx not in isolated_atom_idxs_set
    ]
    new_E_diff_list_meV = [
        val
        for idx, val in enumerate(E_diff_list_meV)
        if idx not in isolated_atom_idxs_set
    ]
    new_F_diff_list_meV = [
        val
        for idx, val in enumerate(F_diff_list_meV)
        if idx not in isolated_atom_idxs_set
    ]

    return (
        new_train_db,
        new_E_dft_list_per_at,
        new_E_nn_list_per_at,
        new_F_dft_list_per_at,
        new_F_nn_list_per_at,
        new_E_diff_list_meV,
        new_F_diff_list_meV,
    )


def filter_dft_calcs_threshold(
    dft_calc_list: list,
    threshold_E_meV: float,
    threshold_F_meV: float,
    workchain=None,
) -> list:
    """
    Filter DFT calculations based on energy and force thresholds.

    Returns a list of serialized ASE Atoms objects that have forces and energy
    below the specified thresholds.
    """
    filtered_dft_calc_list = []
    if workchain:
        workchain.report('Running threshold filters...')

    for calc in dft_calc_list:
        if isinstance(calc, dict):
            calc = aiida_serialized_ase_dict_to_atoms(calc)

        # Get the energy and forces from the DFT calculation and NN
        n_at = len(calc)
        # only check the existing values
        E_dft = calc.info.get('REF_energy', None)
        E_nn = calc.info.get('curr_model_energy', None)
        F_dft = calc.arrays.get('REF_forces', None)
        F_nn = calc.arrays.get('curr_model_forces', None)

        missing_parameters = []
        if E_dft is None:
            missing_parameters.append('REF_energy')
        if E_nn is None:
            missing_parameters.append('curr_model_energy')
        if F_dft is None:
            missing_parameters.append('REF_forces')
        if F_nn is None:
            missing_parameters.append('curr_model_forces')

        if missing_parameters:
            msg = (
                f'Skipping filtering for DFT calculation: '
                f"'{calc.info.get('aiida_uuid', 'unknown')}' "
                f"for structure '{calc.info.get('atl_id', 'unknown')}' "
                f'due to missing values: {", ".join(missing_parameters)}.'
                'Structure will not be not checked or removed.'
            )
            if workchain:
                workchain.report(msg)
            else:
                print(msg)

            # Adding it into the filtered list despite the filter
            # failing
            calc = serialize_ase(calc)
            filtered_dft_calc_list.append(calc)
            continue

        # Normalize energy and forces per atom
        E_dft_at = calc.info.get('REF_energy') / n_at
        E_nn_at = calc.info.get('curr_model_energy') / n_at
        F_dft_at = simplify_forces_struct(calc.arrays.get('REF_forces'))[0] / n_at
        F_nn_at = simplify_forces_struct(calc.arrays.get('curr_model_forces'))[0] / n_at

        # Calculate difference between E and F from DFT and NN
        E_diff_meV = np.abs(E_dft_at - E_nn_at) * 1000
        F_diff_meV = np.abs(F_dft_at - F_nn_at) * 1000

        # Check if the differences are above the specified thresholds
        if E_diff_meV < threshold_E_meV and F_diff_meV < threshold_F_meV:
            # Serialize the structure and add it to the filtered list
            calc = serialize_ase(calc)
            filtered_dft_calc_list.append(calc)
        else:
            if workchain:
                aiida_uuid = calc.info.get('aiida_uuid', 'unknown')
                workchain.report(
                    f"Filtered DFT calculation '{aiida_uuid}' "
                    f"for structure '{calc.info['atl_id']}' "
                    f'with E_diff_meV: {E_diff_meV} and F_diff_meV: {F_diff_meV}'
                )

    if workchain:
        workchain.report('Done running threshold filter!')

    return filtered_dft_calc_list


def write_gathered_dft_calcs_to_file(
    dft_calc_list: orm.List, results_dir: str, workchain=None
) -> tuple[Path, Path]:
    # In case the list is empty, return empty strings
    if len(dft_calc_list) == 0:
        if workchain:
            workchain.report('No DFT calculations to gather.')
        return '', ''

    # Write the results to a temporary file in the calculation directory
    if isinstance(results_dir, orm.Str):
        results_dir = Path(results_dir.value)
    elif isinstance(results_dir, str):
        results_dir = Path(results_dir)

    # Gather calcualtion results from the list of DFT calculation dicts
    results_file_path: Path = results_dir / 'run_tmp_data' / 'gathered_dft_calcs.xyz'
    if isinstance(dft_calc_list[0], dict):
        ase_atoms_list = []
        for calc in dft_calc_list:
            ase_atoms_list.append(aiida_serialized_ase_dict_to_atoms(calc))
        dft_calc_list = ase_atoms_list

    if isinstance(dft_calc_list[0], str):
        dft_calc_list = [orm.load_node(uuid) for uuid in dft_calc_list]

    parsed_structs = []
    for struct in dft_calc_list:
        if isinstance(struct, orm.CalcJobNode):
            with struct.outputs.configuration_result_file.open() as f:
                parsed_struct = ase_read(filename=f, format='extxyz', index=':')
                parsed_structs.extend(parsed_struct)

    if len(parsed_structs) > 0:
        ase_write(filename=results_file_path, images=parsed_structs, format='extxyz')
    else:
        ase_write(filename=results_file_path, images=dft_calc_list, format='extxyz')
    return results_file_path, results_dir


def get_outliers_from_calc_list(curr_struct_res, result_list, outlier_list):
    # Some calculations reported high E and F thoughout all
    # steps. Checking for outliers using the bond distance
    for struct in curr_struct_res:
        outlier_flag = False
        min_dists = []

        symbols = set(struct.get_chemical_symbols())

        struct_ana = Analysis(struct)
        struct_ana.clear_cache()

        # Checking all bond distances for all bond types
        for at_A, at_B in it.combinations_with_replacement(symbols, 2):
            # Getting covalent radii from ase.data
            R_atA = covalent_radii[atomic_numbers[at_A]]
            R_atB = covalent_radii[atomic_numbers[at_B]]

            try:
                # Getting bond distances. This will consider pbc.
                vals = struct_ana.get_values(
                    struct_ana.get_bonds(at_A, at_B, unique=True)
                )
            except IndexError:
                # If there is only one atom of a type (X), there are no
                # distances for atoms of this type (X-X), and therefore
                # the bond distance calculation must be skipped for the
                # current atom pair (X-X).
                continue

            min_bond_length = np.min(vals[0])
            min_dists.append(min_bond_length)

            # Structures with bond distances below this value will be
            # considered outliers.
            bond_length_threshold = (R_atA + R_atB) * 0.7

            # Check for any outliers below the minimum bond length threshold
            # and add the structure to the outlier list, which will not be
            # part of the final database.
            if np.any(np.array(min_dists) < bond_length_threshold):
                outlier_list.append(struct)
                outlier_flag = True
                break

        if not outlier_flag:
            result_list.append(struct)

    return result_list, outlier_list


@calcfunction
def gather_dft_calcs_mace(
    dft_calc_list: list, results_dir: str, workchain=None
) -> orm.List:
    """Collect and preprocess MACE DFT calculation results for active learning input."""
    result_list = []
    outlier_list = []

    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        calc_node = orm.load_node(finished_dft_calc)

        try:
            # Gathering the calculation data from a extxyz stored
            # as a orm.SinglefileData.
            # Depending on the calculation, the output file may be stored
            # in different nodes.
            if hasattr(calc_node.outputs, 'configuration_result_file'):
                struct_file: orm.SinglefileData = (
                    calc_node.outputs.configuration_result_file
                )
            elif hasattr(calc_node.outputs, 'extrapolating_structures'):
                struct_file: orm.SinglefileData = (
                    calc_node.outputs.extrapolating_structures
                )

            # Reading the extxyz file and getting all structures
            with struct_file.as_path() as struct_file_path:
                result_structures = ase_read(
                    struct_file_path, format='extxyz', index=':'
                )

        # If parsing the calculation fails for any reason, skip it.
        except Exception:
            continue

        curr_struct_res = []
        for structure in result_structures:
            # Gathering extra DFT calculation information from calculation
            # and its extras
            if calc_node.base.extras.all.get('atl_struct_type'):
                atl_struct_type = calc_node.base.extras.all.get('atl_struct_type')
            else:
                atl_struct_type = structure.info.get('atl_struct_type', 'unknown')

            if calc_node.base.extras.all.get('atl_calc_uuid'):
                aiida_uuid = calc_node.base.extras.all.get('atl_calc_uuid')
            else:
                aiida_uuid = structure.info.get('aiida_uuid', 'unknown')

            calc_info_dict = {
                'struct_name': calc_node.label,
                'dft_calc_uuid': calc_node.uuid,
                'aiida_uuid': aiida_uuid,
                'atl_struct_type': atl_struct_type,
                # "atl_md_node": calc_node.uuid,
            }

            for key, val in calc_info_dict.items():
                structure.info[key] = val

            # Renaming temporary energy key
            if 'atl_mace_eval_energy' in structure.info:
                structure.info['REF_energy'] = structure.info.pop(
                    'atl_mace_eval_energy'
                )

            # Renaming forces dict
            if 'atl_mace_eval_forces' in structure.arrays:
                structure.arrays['REF_forces'] = structure.arrays.pop(
                    'atl_mace_eval_forces'
                )

            if 'forces' in structure.arrays:
                structure.arrays.pop('forces')

            # result_list.append(structure)
            curr_struct_res.append(structure)

        # Checking for outliers
        result_list, outlier_list = get_outliers_from_calc_list(
            curr_struct_res, result_list, outlier_list
        )

    if workchain:
        node = orm.load_node(workchain.value)
        node.logger.log(
            level=LOG_LEVEL_REPORT,
            msg=f'[{node.pk}|{node.process_label}|gather_dft_calcs_mace]:'
            f' Removed {len(outlier_list)} outliers.',
        )

    # Serializing the structures
    result_list = [serialize_ase(struct) for struct in result_list]

    # Converting results directory to Path object if it's an aiida string
    if isinstance(results_dir, orm.Str) and results_dir.value is not None:
        results_dir = Path(results_dir.value)

    # Saving outliers
    if outlier_list:
        outliers_file_path = results_dir / 'outliers.extxyz'
        ase_write(filename=outliers_file_path, images=outlier_list)

    # Return the structure list
    return orm.List(result_list)


def iqr_outlier_check(res_list: list) -> np.ndarray:
    """
    Identifies outliers in a list of E/F values using the interquartile range (IQR).

    Parameters
    ----------
    res_list : list
        A list of numerical values to check for outliers.

    Returns
    -------
    np.ndarray
        An array where outliers are replaced with NaN and non-outliers are retained.

    Notes
    -----
    The function calculates the 30th and 70th percentiles of the input list
    to determine the interquartile range (IQR).
    Values outside the range [Q1 - 1.5 * IQR, Q2 + 1.5 * IQR] are considered
    outliers and replaced with NaN.
    """
    q1 = np.percentile(res_list, 30)
    q2 = np.percentile(res_list, 70)
    iqr = q2 - q1

    # Define outlier thresholds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q2 + 1.5 * iqr

    # Identify and remove outliers
    filtered_data = np.array(
        [x if lower_bound <= x <= upper_bound else np.nan for x in res_list]
    )

    return filtered_data


def vasprun_add_info_dict(vasprun_dict: dict, calc_info_dict: dict) -> dict:
    """Add calculation information to the vasprun dictionary."""
    info_list = [
        'stress',
        'dipole',
        'forces',
        'struct_name',
        'energy',
        'aiida_uuid',
        'free_energy',
        'atl_struct_type',
    ]

    # If forces already in the arrays dictionary, it is not needed in
    # atoms.info
    if 'forces' in vasprun_dict.arrays:
        calc_info_dict.pop('forces')

    if not isinstance(vasprun_dict, dict):
        vasprun_dict = Atoms.todict(vasprun_dict)

    if not vasprun_dict.get('info'):
        vasprun_dict['info'] = {}

    for key, val in calc_info_dict.items():
        if key not in vasprun_dict['info'] and key in info_list:
            if key == 'free_energy':
                key.replace('free_', '')

            vasprun_dict['info'][key] = val
    return vasprun_dict


@calcfunction
def remove_structs_from_seed_gen_db(
    seed_gen_path: orm.Str, delete_indices: list
) -> orm.List:
    """
    Remove specified structures from a seed generation database based on UUIDs.

    This function iterates over a list of UUIDs (delete_indices) and removes the
    corresponding structures from a seed generation database. The database is accessed
    via the `seed_gen_db` object, which is loaded from the `seed_gen_path` using ASE.
    Each element of the list is an ase.Atoms object with an unique identifier
    (`atl_id`/`aiida_uuid`) in the info attribute.
    The function writes the modified list back into `seed_gen_path` after the specified
    ones have been removed. No list is returned into the workchain to avoid having
    to serialize the atoms list.

    Parameters
    ----------
    seed_gen : orm.Str | str
        The path to the seed generation database.
    delete_indices : list
        A list of UUIDs (strings) identifying the structures to be removed
        from the seed generation database.

    """
    if isinstance(seed_gen_path, str):
        seed_gen_db = load_database(seed_gen_path)
    elif isinstance(seed_gen_path, orm.Str):
        seed_gen_db = load_database(seed_gen_path.value)

    if not isinstance(seed_gen_db, list):
        seed_gen_db = seed_gen_path.get_list()

    # Clean delete list
    delete_indices = [uuid for uuid in delete_indices if uuid is not None]

    # Normalize database
    new_db = []
    for struct in seed_gen_db:
        if isinstance(struct, dict):
            struct = Atoms.fromdict(struct)

        info = struct.info

        struct_uuid = info.get('atl_id') or info.get('aiida_uuid')

        # Assign uuid on the fly if missing
        if not struct_uuid:
            struct_uuid = str(uuid4())
            struct.info['atl_id'] = struct_uuid

        if struct_uuid not in delete_indices:
            new_db.append(struct)

    # Write back updated database
    ase_write(
        filename=seed_gen_path.value,
        images=new_db,
        format='extxyz',
    )


@calcfunction
def check_md_seed_agreement(
    return_list_path: str | None,
    md_structs_in_domain: bool | None,
) -> orm.Bool:
    """
    Check if all predictions agree for current seed.

    Parameters
    ----------
    return_list_path : str | None
        Path pointing to the file that contains all calculations
        for predictions where the models disagreed.

    Returns
    -------
    orm.Bool
        True if all the predictions have agreed for the current MD seed
        on the current AL iteration. False if there is no agreement on
        on all structures.
    """
    if not return_list_path or return_list_path == '':
        # If no DFT calculations were found, because all calcs have failed,
        # the predictions are considered to be in disagreement.
        if md_structs_in_domain is False:
            return orm.Bool(False)
        # If all structures were in domain, the predictions
        # are considered to be in agreement.
        else:
            return orm.Bool(True)

    if isinstance(return_list_path, orm.Str):
        return_list_path = return_list_path.value

    # Loading the list of structures
    return_structs = ase_read(filename=return_list_path, format='extxyz', index=':')

    # If there are structures, the predictions are considered to be in
    # disagreement
    if len(return_structs) > 0:
        return orm.Bool(False)
    else:
        return orm.Bool(True)


def read_toml_settings(settings_file: str | Path) -> dict:
    """Read a TOML file containing settings for the active learning workflow."""
    with open(settings_file, 'rb') as f:
        settings = toml.load(f)
    return settings
