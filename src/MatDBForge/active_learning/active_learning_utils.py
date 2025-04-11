"""General utility functions for the active learning workflows."""

import io
import itertools as it
import tempfile
import time
import tomllib as toml
from contextlib import redirect_stdout
from pathlib import Path

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

from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.core import code_utils as mdb_cud
from MatDBForge.core.filtering.structure_filters import (
    apply_filter_exploding_structures,
)
from MatDBForge.workflows import aiida_utils as mdb_aut
from MatDBForge.workflows.aiida_utils import can_submit_calculation


def aiida_wait_submit(builder, computer, calc_count=0):
    # Get the calculation limit, from the computer metadata set to 0
    # if not present.
    # `mdb_calc_limit` is a custom property set with:
    # computer.set_property(name='mdb_calc_limit', value=366)
    try:
        calc_limit = builder.metadata.computer.metadata.get('mdb_calc_limit', 0)
    except AttributeError:
        # If the builder does not have computer metadata, get the limit
        # from the computer
        calc_limit = computer.metadata.get('mdb_calc_limit', 0)
    except Exception:
        # If `mdb_calc_limit` is not set, set the limit to 0
        calc_limit = 0

    # Check if the calculation can be submitted
    if calc_limit == 0:
        can_submit = True
    else:
        can_submit, calc_count_sch = can_submit_calculation(
            computer=computer,
            code=builder.code.label,
            limit=calc_limit,
        )
    mdb_cud.custom_print(f'Can submit: {can_submit}.', 'debug')
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
    #     print('#@# future.get_state(): ', future.get_state())
    #     time.sleep(10)


def md_apply_temperature_ramp(dyn, total_steps, T_start, T_end):
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
    # Update the temperature using the ramp function
    current_temperature = T_start + (T_end - T_start) * dyn.nsteps / total_steps

    # Set the temperature in units of energy
    dyn.set_temperature(temperature_K=current_temperature)

    # Adding T value to info dict
    dyn.atoms.info['md_temperature'] = current_temperature


def md_write_frame_traj(dyn, traj):
    """
    Function to write frames to a trajectory during ASE MD simulations.

    Parameters
    ----------
    dyn : ASE MD object
        ASE MD object used to run the MD simulation.
    traj : ASE trajectory object
        ASE trajectory object to store the MD simulation.

    """
    # Write the frame to the trajectory
    REF_energy = dyn.atoms.get_potential_energy()
    REF_forces = dyn.atoms.get_forces()
    dyn.atoms.info['REF_energy'] = REF_energy
    dyn.atoms.info['REF_forces'] = REF_forces
    traj.write(dyn.atoms, energy=REF_energy, forces=REF_forces)


# TODO: Merge with `generate_descriptor_mace` and `generate_descriptor_soap`
def generate_descriptors(
    model_path: str,
    database,
    device='cpu',
    dtype='float32',
    descriptor_dict: dict = None,
):
    from mace.calculators import MACECalculator

    # Initialize the MACE calculator
    model = torch.load(model_path, map_location=torch.device(device))
    # model = model.to(device=device, dtype=dtype)
    calculator = MACECalculator(models=model, device=device, default_dtype=dtype)

    # Generate descriptors for all structures in the database
    descriptor_dict = {}
    descriptor_list = []

    for struct in database:
        if struct.info.get('mdb_id'):
            struct_key = struct.info.get('mdb_id')
        else:
            struct_key = struct.info.get('aiida_uuid')

        descriptor_dict[struct_key] = {
            'descriptors': [],
            'latent_space': [],
        }
        curr_struct_descriptors = calculator.get_descriptors(struct)
        descriptor_list.append(curr_struct_descriptors)
        descriptor_dict[struct_key]['descriptors'].append(curr_struct_descriptors)

    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr


def generate_descriptors_mace(
    model_path: str,
    database,
    descriptor_settings: dict,
):
    from mace.calculators import MACECalculator

    device = descriptor_settings.get('device', 'cpu')
    dtype = descriptor_settings.get('dtype', 'float32')
    calculator = MACECalculator(
        model_paths=model_path, device=device, default_dtype=dtype
    )

    descriptor_dict = {}
    descriptor_list = []
    for struct in database:
        if struct.info.get('mdb_id'):
            struct_key = struct.info.get('mdb_id')
        else:
            struct_key = struct.info.get('aiida_uuid')

        descriptor_dict[struct_key] = {
            'descriptors': [],
            'latent_space': [],
        }

    for struct in database:
        curr_struct_descriptors = calculator.get_descriptors(struct)
        descriptor_list.append(curr_struct_descriptors)
        descriptor_dict[struct_key]['descriptors'].append(curr_struct_descriptors)

    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr


def generate_descriptors_soap(database, descriptor_settings: dict): ...


def run_mace_md_ase(
    init_conf: Atoms,
    md_params: dict,
    T_start: float,
    traj_obj: TrajectoryWriter | None,
    prepend_path: str | Path = '.',
    explode_filter_dict: dict = None,
    mode='normal',
    md_struct_list=None,
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
    """
    from mace.calculators import MACECalculator

    T_multiplier = md_params.get('max_temp_multiplier', 1.0)
    T_end = T_start * T_multiplier
    timestep = md_params['timestep_duration_ps']

    if not explode_filter_dict:
        explode_filter_dict = {}

    if md_params.get('langevin_friction_ps-1'):
        friction = md_params['langevin_friction_ps-1']
    else:
        friction = md_params['timestep_duration_ps'] * 100

    num_steps = md_params['num_steps']
    write_interval = md_params.get('md_write_interval', 1)
    thermostat = md_params.get('md_thermostat', 'langevin')

    md_params['T_start'] = T_start
    md_params['T_end'] = T_end
    md_params['langevin_friction_ps-1'] = friction
    md_params['write_interval'] = write_interval

    md_type = md_params.get('md_type', 'mace')

    if md_type == 'mace':
        mace_foundation = md_params.get('mace_foundation')

        if mace_foundation:
            from mace.calculators import mace_mp

            calculator = mace_mp(
                device=md_params.get('device', 'cpu'),
                default_dtype=md_params.get('default_dtype', 'float64'),
            )
        else:
            # Load the trained model as an ASE calculator and attach it to the
            # atoms object
            model_path = Path(prepend_path) / 'curr_model.model'

            calculator = MACECalculator(
                model_paths=model_path,
                device=md_params.get('device', 'cpu'),
                default_dtype=md_params.get('default_dtype', 'float64'),
            )
        init_conf.calc = calculator

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

    match thermostat:
        case 'langevin':
            # Define the Langevin dynamics
            dyn = Langevin(
                atoms=init_conf,
                # convert timestep in ps to fs
                timestep=(timestep * 1000) * units.fs,
                temperature_K=T_start,
                # convert friction in ps-1 to fs-1
                friction=(friction / 1000) / units.fs,
                logfile=log_folder / f'md_info-{T_start}K.log',
            )
        case 'nose-hoover':
            # Change the simulation box to remove any small numbers not in the diagonal
            # of the box matrix
            box = init_conf.get_cell()
            box[np.abs(box) < 1e-2] = 0
            init_conf.set_cell(box)

            dyn = NPT(
                atoms=init_conf,
                timestep=(timestep * 1000) * units.fs,
                temperature_K=T_start,
                externalstress=0,
                ttime=100 * units.fs,
                pfactor=None,
                logfile=log_folder / f'md_info-{T_start}K.log',
            )
    mdb_cud.custom_print('Running MD simulation using settings:', 'info')
    rprint(md_params)

    # Attach the thermostat function to increase the temperature
    # linearly from T_start to T_end
    dyn.attach(
        interval=1,
        function=md_apply_temperature_ramp,
        dyn=dyn,
        total_steps=num_steps,
        T_start=T_start,
        T_end=T_end,
    )

    if traj_obj is not None and mode == 'normal':
        dyn.attach(
            md_write_frame_traj,
            # atoms=dyn.atoms,
            # REF_energy=dyn.atoms.get_potential_energy(),
            # REF_forces=dyn.atoms.get_forces(),
            dyn=dyn,
            traj=traj_obj,
            interval=1,
        )

    if explode_filter_dict.get('enable') and mode == 'normal':
        dyn.attach(
            md_stop_explode_filter,
            dyn=dyn,
            interval=int(num_steps * 0.1),
            cov_rad_multiplier_max=explode_filter_dict.get('cov_rad_multiplier_max'),
            cov_rad_multiplier_min=explode_filter_dict.get('cov_rad_multiplier_min'),
            max_T=T_end,
            max_T_multiplier=explode_filter_dict.get('max_T_multiplier', 10),
        )

    if mode != 'normal':
        if not md_struct_list:
            md_struct_list = []

        dyn.attach(
            md_save_gen_structs,
            dyn=dyn,
            struct_list=md_struct_list,
            interval=1,
        )

    # Run the MD simulation
    try:
        dyn.run(num_steps)
    except Exception as e:
        print(f'Error in MD simulation: {e}')

    if mode != 'normal':
        return md_struct_list


def md_stop_explode_filter(
    dyn,
    cov_rad_multiplier_min,
    cov_rad_multiplier_max,
    max_T,
    max_T_multiplier,
):
    has_exploded = apply_filter_exploding_structures(
        dyn.atoms,
        cov_rad_multiplier_min=cov_rad_multiplier_min,
        cov_rad_multiplier_max=cov_rad_multiplier_max,
        max_T=max_T,
        max_T_multiplier=max_T_multiplier,
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

    # Calculate the sample standard deviation of the forces
    # for each structure
    forces_std = np.nanstd(forces_model_list, axis=0, ddof=1)

    return forces_std


def get_model_energies_std(energies_dict: dict) -> np.ndarray:
    """Get the standard deviation of the energies for each structure in the dict."""
    energies_model_list: np.ndarray = model_res_dict_to_arr(
        energies_dict, dict_type='energy'
    )

    # Calculate the sample standard deviation of the energies
    # for each structure
    energies_std = np.nanstd(energies_model_list, axis=0, ddof=1)
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
    struct_type = row['mdb_struct_type']
    struct_type = mdb_aut.CalcType.from_string('single_point_' + struct_type)

    # Gathering row information
    (curr_structure, curr_material_name, curr_unique_id, curr_phase) = (
        mdb_aut.gather_calc_data_from_row(row, curr_structure=struct)
    )

    # Getting default potential mapping
    potential_mapping = mdb_aut.generate_potential_mapping()

    # Updating general INCAR with calc type specific options
    specific_options = dft_settings.get(struct_type)
    if specific_options:
        specific_options = specific_options.get('incar')
        for setting, val in specific_options.items():
            dft_settings['incar'][setting] = val

    builder = mdb_aut.submit_aiida_vasp_calculation(
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


def get_dft_calc_builder_mace_list(
    struct_list: list,
    row,
    dft_settings: dict,
    container_settings: dict,
):
    """Get a MACE calculation builder for a given structure list and row."""
    updated_struct_list = []

    containerized: bool = container_settings.get('use_container', False)

    for idx, curr_struct in enumerate(struct_list):
        curr_struct = struct_list[idx]

        # Gathering row information
        (curr_structure, curr_material_name, curr_unique_id, curr_phase) = (
            mdb_aut.gather_calc_data_from_row(row, curr_structure=curr_struct)
        )

        if not isinstance(curr_struct, Atoms):
            curr_struct = AseAtomsAdaptor().get_atoms(curr_struct)

        curr_struct.info['mdb_md_node'] = row['mdb_md_node']
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

    dft_settings['options'].pop('code_string')

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

    # Adding the final database path and the 'mdb_train_db_' prefix
    # used to identifd the final database.
    final_db_path = curr_run_dir / f'mdb_train_db_{final_db_name}.xyz'
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
            vasprun: Atoms = mdb_conv._gather_mace_req_calc_data_from_node(
                finished_dft_calc
            )

            # Gathering extra DFT calculation information from vasprun.xml
            calc_info_dict = mdb_conv.gather_calc_data_from_node(
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

        # if "stress" not in vasprun.arrays.keys():
        #     vasprun.new_array(
        #         name="stress",
        #         a=np.array(calc_info_dict["stress"]),
        #     )

        # Adding the type of structure to the atoms.info dict
        struct_type = mdb_conv.get_struct_type(
            vasprun=vasprun, dft_calc_node=finished_dft_calc
        )
        calc_info_dict['mdb_struct_type'] = struct_type
        vasprun = vasprun_add_info_dict(vasprun, calc_info_dict)

        # Generate a structure name and gathering the aiida_uuid
        vasprun: Atoms = mdb_conv._add_entry_to_mace_input(
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

        vasprun: dict = serialize_ase(vasprun)

        vasprun_list.append(vasprun)

    # TODO: Checking for outliers
    # result_list, outlier_list = get_outliers_from_calc_list(
    #     curr_struct_res, result_list, outlier_list
    # )

    return_list = orm.List([val for val in vasprun_list])
    return return_list


def write_gathered_dft_calcs_to_file(dft_calc_list: orm.List, results_dir: str):
    # Write the results to a temporary file in the calculation directory
    if isinstance(results_dir, orm.Str):
        results_dir = Path(results_dir.value)
    elif isinstance(results_dir, str):
        results_dir = Path(results_dir)

    results_file_path = results_dir / 'run_tmp_data' / 'gathered_dft_calcs.xyz'
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
) -> orm.Str:
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
            if calc_node.base.extras.all.get('mdb_struct_type'):
                mdb_struct_type = calc_node.base.extras.all.get('mdb_struct_type')
            else:
                mdb_struct_type = structure.info.get('mdb_struct_type', 'unknown')

            if calc_node.base.extras.all.get('mdb_calc_uuid'):
                aiida_uuid = calc_node.base.extras.all.get('mdb_calc_uuid')
            else:
                aiida_uuid = structure.info.get('aiida_uuid', 'unknown')

            calc_info_dict = {
                'struct_name': calc_node.label,
                'dft_calc_uuid': calc_node.uuid,
                'aiida_uuid': aiida_uuid,
                'mdb_struct_type': mdb_struct_type,
                # "mdb_md_node": calc_node.uuid,
            }

            for key, val in calc_info_dict.items():
                structure.info[key] = val

            # Renaming temporary energy key
            if 'mdb_mace_eval_energy' in structure.info:
                structure.info['REF_energy'] = structure.info.pop(
                    'mdb_mace_eval_energy'
                )

            # Renaming forces dict
            if 'mdb_mace_eval_forces' in structure.arrays:
                structure.arrays['REF_forces'] = structure.arrays.pop(
                    'mdb_mace_eval_forces'
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

    results_file_path, results_dir = write_gathered_dft_calcs_to_file(
        dft_calc_list=dft_calc_list,
        results_dir=results_dir,
    )

    # Saving outliers
    if outlier_list:
        outliers_file_path = results_dir / 'outliers.extxyz'
        ase_write(filename=outliers_file_path, images=outlier_list)

    # Return the path to the temporary file
    return orm.Str(str(results_file_path))


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
        'mdb_struct_type',
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
    (`aiida_uuid`) in the info attribute.
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

    for curr_uuid in delete_indices:
        for del_idx, struct in enumerate(seed_gen_db):
            struct: Atoms = struct.todict()

            struct_uuid = struct.get('info', {}).get('mdb_id')
            if not struct_uuid:
                struct_uuid = struct.get('info', {}).get('aiida_uuid')

            if curr_uuid == struct_uuid:
                del seed_gen_db[del_idx]

    ase_write(
        filename=seed_gen_path.value,
        images=seed_gen_db,
        format='extxyz',
    )


@calcfunction
def check_md_seed_agreement(return_list_path: str | None) -> orm.Bool:
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
    # If no DFT calculations were found, because all calcs have failed,
    # the predictions are considered to be in disagreement.
    if not return_list_path or return_list_path == '':
        return orm.Bool(False)

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


def get_concave_hull(latent_space: np.ndarray) -> np.ndarray: ...


def read_toml_settings(settings_file: str | Path) -> dict:
    """Read a TOML file containing settings for the active learning workflow."""
    with open(settings_file, 'rb') as f:
        settings = toml.load(f)
    return settings
