#!/usr/bin/env python3
"""Script to run MACE MD simulations, descriptor generation and extrapolation checks.

This script is part of MatDBForge's active learning loop, and is used to combine
in one single calculation job the acquisition of the MD trajectory, the generation
of the descriptors followed by performing any extrapolation checks, filtering
the MD trajectory if necessary.
"""

import pathlib as pl
import tomllib
import uuid
import warnings

import numpy as np
import torch
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter
from mace.calculators import MACECalculator
from shapely.geometry import Point, Polygon

import MatDBForge.active_learning.active_learning_utils as mdb_al_ut
import MatDBForge.active_learning.extrapolation.concave_hull as mdb_chull
from MatDBForge.active_learning.extrapolation import autoencoder as mdb_ae
from MatDBForge.core import code_utils as mdb_cud
from MatDBForge.core.filtering import structure_filters as mdb_str_filters

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def check_traj_in_domain(
    concave_hull: np.ndarray, descriptor_dict: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check if the descriptors are inside the concave hull.

    Parameters
    ----------
    concave_hull : np.ndarray
        Concave hull of the latent space for the database.
    descriptor_dict : dict
        Descriptor dictionary containing the descriptors for each frame.

    Returns
    -------
    np.ndarray
        Array containing the descriptors that are inside the concave hull.
    np.ndarray
        Array containing the descriptors that are outside the concave hull.
    np.ndarray
        Array containing boolean values showing if the frame is inside the concave
    """
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for c_uuid in descriptor_dict:
        descriptors = descriptor_dict[c_uuid]['latent_space']

        for _, frame_desc in enumerate(descriptors):
            # c_p_in = []
            # c_p_out = []
            c_all_p = []
            for point in frame_desc:
                p = Point(point)
                if polygon.contains(p):
                    point_inside.append(point)
                    c_all_p.append(True)
                else:
                    point_outside.append(point)
                    c_all_p.append(False)

            all_points_in_out.append(np.array(c_all_p))

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out


def limit_md_frames(md_traj, md_params: dict):
    """
    Limit the number of frames in the MD trajectory.

    Parameters
    ----------
    md_traj : ase.Atoms
        MD trajectory.
    md_params : dict
        Dictionary containing the MDB settings for MD.

    Returns
    -------
    list
        List containing the MD trajectory with the limited number of frames.
    list
        Mask containing the indices of the frames that were kept.
    """
    # Limit total number of frames
    total_num_frames = mdb_al_ut.get_total_num_frames(
        len_traj=len(md_traj),
        md_tstep_duration_ps=md_params['timestep_duration_ps'],
        frame_interval=md_params['al_keep_struct_every_n_ps'],
    )
    # Choose the number of frames evenly and create a mask for the arrays
    mask = np.linspace(0, len(md_traj) - 1, int(total_num_frames), dtype=int)

    # Apply mask
    return [md_traj[i] for i in mask], mask


def simple_extrapolation_check(
    curr_it_db_max: np.ndarray, curr_it_db_min: np.ndarray, descriptor_dict: dict
):
    """
    Apply a basic extrapolation check based on the maximum and minimum values
    of the descriptors for the entire database.

    Parameters
    ----------
    curr_it_db_max : np.ndarray
        Maximum value for the descriptors for the entire database.
    curr_it_db_min : np.ndarray
        Minimum value for the descriptors for the entire database.
    descriptor_dict : dict
        Descriptor dictionary containing the descriptors for each frame.

    Returns
    -------
    dict
        Updated descriptor dictionary with the extrapolation check applied.
    """
    mdb_cud.custom_print('Applying basic extrapolation check...', 'info')
    for curr_uuid, val in descriptor_dict.items():
        structure_uuid = curr_uuid
        descr_list = val['descriptors']

    for frame_idx, frame_descriptors in enumerate(descr_list):
        below_min = frame_descriptors < curr_it_db_min
        above_max = frame_descriptors > curr_it_db_max
        is_frame_extrapolating = np.any(np.logical_or(below_min, above_max))
        if is_frame_extrapolating:
            descriptor_dict[structure_uuid]['is_extrapolating'][frame_idx] = True
    return descriptor_dict


if __name__ == '__main__':
    # Initialize random seed
    rng_seed = np.random.randint(0, int(1e15))
    mdb_cud.custom_print(f"Using random seed: '{rng_seed}'")

    mdb_cud.custom_print('Starting process structure script...', 'info')

    # The /mdb_data directory should only exist in the containerized version
    # of the code. This conditional statement will get the correct path for
    # input and output files.
    if pl.Path('/mdb_data').exists():
        prepend_path = pl.Path('/mdb_data')
    else:
        prepend_path = pl.Path('.')

    # Load the rmse_arr.npy file and assign the values to the variables
    rmse_arr = np.load(prepend_path / 'rmse_arr.npy')

    # meV/at
    e_rmse = rmse_arr[0]
    # meV/A
    f_rmse = rmse_arr[1]

    # Define results folder
    res_folder = prepend_path / pl.Path('./results')
    res_folder.mkdir(exist_ok=True)

    # Initialize the logger
    log_folder = prepend_path / pl.Path('./logs')
    log_folder.mkdir(exist_ok=True)
    logger = mdb_cud.init_logger(source='proc_structure', log_path=log_folder)

    # Read TOML file with settings
    with open(prepend_path / 'settings.toml', 'rb') as f:
        settings = tomllib.load(f)

    # Parse settings
    md_params = settings.get('md', {}).get('parameters')
    md_filters = settings.get('md', {}).get('filters', {})
    T_list = md_params['temperature_list_K']

    # Logging CUDA information
    if md_params.get('device') == 'cuda':
        mdb_cud.custom_print(
            (
                f'CUDA INFO - available: {torch.cuda.is_available()}, '
                f'device_count: {torch.cuda.device_count()}, '
                f'current_device: {torch.cuda.current_device()}'
            ),
            'debug',
        )

    # Get the EF disagreement type
    ef_disagreement_type = settings.get('extrapolation', {}).get(
        'disagreement_check_type', 'training'
    )
    # Get the extrapolation type
    extrap_type = settings.get('extrapolation', {}).get(
        'check_extrapolation_type', 'advanced'
    )

    # Get the dimensionality reduction method settings
    dim_red_method = settings.get('descriptors', {}).get(
        'dimensionality_reduction_method', None
    )
    if dim_red_method == 'autoencoder':
        dim_red_settings = settings.get('descriptors', {}).get('autoencoder', {})
    elif dim_red_method == 'pca':
        dim_red_settings = settings.get('descriptors', {}).get('pca', {})

    # Read the initial structure
    init_conf_orig = ase_read(prepend_path / 'curr_structure.xyz', format='extxyz')

    ## Running MD simulations for given temperatures
    for T_start in T_list:
        traj_filename = res_folder / f'md_traj_final_temp-{T_start}.traj'

        # Not repeating MD simulations if the trajectory already exists.
        # Mainly for testing,  this should not happen during a normal run.
        if pl.Path(traj_filename).exists():
            mdb_cud.custom_print(
                f"MD trajectory for 'T={T_start} K' already exists. Skipping...", 'warn'
            )
            continue

        # Instantiating ase trajectory object
        init_conf = init_conf_orig.copy()
        traj_obj = TrajectoryWriter(
            filename=traj_filename,
            mode='w',
            atoms=init_conf,
            properties=['energy', 'forces', 'REF_energy', 'REF_forces', 'MACE_energy'],
        )
        print()
        mdb_cud.custom_print(f"Running MD simulation for 'T={T_start} K'", 'info')
        mdb_al_ut.run_mace_md_ase(
            # temperature_ramp=mdb_al_ut.md_apply_temperature_ramp,
            md_params=md_params,
            T_start=T_start,
            traj_obj=traj_obj,
            init_conf=init_conf,
            prepend_path=prepend_path,
            explode_filter_dict=md_filters.get('exploding_structures', {}),
        )
        mdb_cud.custom_print('MD simulation completed!', 'done')

    # Read MD-generated trajectories for given temperatures
    traj_files = res_folder.glob('*final_temp-*.traj')

    for curr_traj in traj_files:
        print()
        mdb_cud.custom_print(f"Checking extrapolating frames for '{curr_traj}'", 'info')
        extrap_frame_idx = []

        curr_temp = float(str(curr_traj).split('temp-')[1].split('.traj')[0])

        # Read the trajectory
        md_traj = [frame for frame in TrajectoryReader(curr_traj)]
        orig_md_size = len(md_traj)
        mdb_cud.custom_print(f'Trajectory length: {orig_md_size}', 'info')

        # Add frame info and overwrite the trajectory
        for frame_idx, frame in enumerate(md_traj):
            frame.info['frame_idx'] = frame_idx
            frame.info['calc_type'] = 'MACE_MD'
            # frame.info['md_temperature'] = float(curr_temp)
            # frame.info['mdb_id'] = str(init_conf.info['mdb_id'])

        # Apply MD filters and removing these frames from the trajectory
        frames_to_remove = []
        mdb_cud.custom_print('Applying MD filters to remove outliers...', 'info')

        if md_filters.get('layer_distance', {}).get('enable'):
            mdb_cud.custom_print("Running 'layer distance' filter...", 'info')
            later_distance_r_frames = []

            # Getting max distance from input
            max_dist: float = md_filters['layer_distance']['max_layer_distance_ang']

            for idx, frame in enumerate(md_traj):
                is_structure_wrong = mdb_str_filters.apply_filter_layer_distance(
                    struct=frame, max_layer_distance_ang=max_dist
                )
                if is_structure_wrong:
                    later_distance_r_frames.append(idx)
            frames_to_remove.extend(later_distance_r_frames)

            mdb_cud.custom_print(
                f'Marked by layer distance filter: {len(later_distance_r_frames)}',
                'debug',
            )

            exploding_structs = []

        if md_filters.get('exploding_structures', {}).get('enable'):
            mdb_cud.custom_print("Running 'exploding structures' filter...", 'info')

            explod_filt_settings = md_filters.get('exploding_structures', {})
            # Getting multiplier from input
            cov_rad_multiplier_max: float = explod_filt_settings.get(
                'cov_rad_multiplier_max', 10.0
            )
            cov_rad_multiplier_min: float = explod_filt_settings.get(
                'cov_rad_multiplier_min', 0.78
            )
            max_T = curr_temp * md_params.get('max_temp_multiplier', 1)
            max_T_multiplier = explod_filt_settings.get('max_T_multiplier', 10)

            # Applying filter for every frame
            for idx, frame in enumerate(md_traj):
                is_structure_wrong: bool = (
                    mdb_str_filters.apply_filter_exploding_structures(
                        struct=frame,
                        cov_rad_multiplier_max=cov_rad_multiplier_max,
                        cov_rad_multiplier_min=cov_rad_multiplier_min,
                        max_T=max_T,
                        max_T_multiplier=max_T_multiplier,
                    )
                )
                if is_structure_wrong:
                    exploding_structs.append(idx)
            frames_to_remove.extend(exploding_structs)

            mdb_cud.custom_print(
                f'Marked by exploding structures filter: {len(exploding_structs)}',
                'debug',
            )

        if md_filters.get('check_atoms_no_neighbor', {}).get('enable'):
            mdb_cud.custom_print("Running 'no neighbor' filter...", 'info')
            neighbor_r_frames = []

            # Getting multiplier from input
            cov_rad_mult: float = md_filters.get('check_atoms_no_neighbor', {}).get(
                'covalent_radius_multiplier', 1.0
            )
            # Applying filter for every frame
            for idx, frame in enumerate(md_traj):
                is_structure_wrong = mdb_str_filters.apply_filter_no_neighbors(
                    struct=frame, cov_rad_multiplier=cov_rad_mult
                )
                if is_structure_wrong:
                    neighbor_r_frames.append(idx)
            frames_to_remove.extend(neighbor_r_frames)
        mdb_cud.custom_print(
            f'Marked by no neighbor filter: {len(neighbor_r_frames)}', 'debug'
        )

        # Remove duplicate frames
        frames_to_remove = list(set(frames_to_remove))

        # Create a new list excluding the frames to remove
        md_traj_filtered = [
            frame for i, frame in enumerate(md_traj) if i not in frames_to_remove
        ]

        mdb_cud.custom_print(
            f'Trajectory length after MD filters: {len(md_traj_filtered)}', 'info'
        )

        if len(md_traj_filtered) == 0:
            mdb_cud.custom_print(
                (
                    'No MD frames left after filtering. '
                    'This means that probably there are a lot of unrealistic structures'
                    '. Check the training data and models used to run this MD.'
                ),
                'warning',
            )

            # Returning empty frames list
            ase_write(
                res_folder / 'extrapolating_frames.xyz',
                format='extxyz',
                images=[],
                append=True,
            )

            # Skip the rest of the process for the current T.
            continue

        # Save removed frames to a file
        if md_filters.get('save_filtered_structures'):
            mdb_cud.custom_print(
                (
                    f'Saving {len(frames_to_remove)} filtered structures'
                    f" to '{res_folder / 'filtered_frames.xyz'}'"
                ),
                'info',
            )
            filtered_frames = [md_traj[i] for i in frames_to_remove]
            ase_write(
                res_folder / 'filtered_frames.xyz',
                format='extxyz',
                images=filtered_frames,
                append=True,
            )
        else:
            mdb_cud.custom_print('Filtered structures not saved.', 'info')

        # Limit total number of frames
        md_traj_short, short_mask = limit_md_frames(md_traj_filtered, md_params)
        mdb_cud.custom_print(
            f'Limited number of frames to: {len(md_traj_short)}', 'info'
        )

        # Running evaluation of the energies and forces using each commitee model
        mdb_cud.custom_print('Running committee evaluation...', 'info')
        model_file_list = list(prepend_path.glob('*.model'))
        comm_settings = settings.get('committee_eval', {})
        comm_results = {}
        for model in model_file_list:
            comm_results[model.stem] = {'REF_energy': [], 'REF_forces': []}

            calculator = MACECalculator(
                model_paths=prepend_path / model,
                device=comm_settings.get('mace', {}).get('device', 'cpu'),
                default_dtype=comm_settings.get('mace', {}).get(
                    'default_dtype', 'float32'
                ),
                batch_size=comm_settings.get('mace', {}).get('batch_size', 12),
            )

            for _, frame in enumerate(md_traj_short):
                frame.calc = calculator

                # Get the energy [meV/at] and forces [meV/A]
                comm_results[model.stem]['REF_energy'].append(
                    frame.get_potential_energy() * 1000 / len(frame)
                )
                comm_results[model.stem]['REF_forces'].append(frame.get_forces() * 1000)

        ## Apply E/F commitee extrapolation filter
        model_acc_multiplier = settings['active_learning'].get(
            'model_acc_multiplier', 1
        )

        mdb_cud.custom_print(
            f"Using disagreement check type: '{ef_disagreement_type}'", 'info'
        )
        mdb_cud.custom_print('Printing extrapolation statistics for E:', 'debug')

        if ef_disagreement_type == 'training':
            e_error_threshold = model_acc_multiplier * e_rmse  # meV / at
            f_error_threshold = model_acc_multiplier * f_rmse  # meV / A

            mdb_cud.custom_print(
                f'model_acc_multiplier: {model_acc_multiplier}', 'none'
            )

            # Prepare energies and forces dict
            model_names = list(comm_results.keys())
            model_energies_dict = {}
            model_forces_dict = {}
            for model_name in model_names:
                if not model_energies_dict.get(model_name):
                    model_energies_dict[model_name] = []
                if not model_forces_dict.get(model_name):
                    model_forces_dict[model_name] = []

                model_energies_dict[model_name].append(
                    comm_results[model_name]['REF_energy']
                )
                model_forces_dict[model_name].append(
                    comm_results[model_name]['REF_forces']
                )

            # Checking if the energies are over the error threshold
            energies_stat = mdb_al_ut.get_model_energies_std(model_energies_dict)  # meV
            maximum_value_e = np.average(energies_stat) * 10  # meV
            mdb_cud.custom_print(f'e_error_threshold: {e_error_threshold}', 'none')
            mdb_cud.custom_print(f'e_maximum_value: {maximum_value_e}', 'none')
            mdb_cud.custom_print(f'energies_stat: {energies_stat}', 'none')

            error_e_structures_sm = np.ma.make_mask(
                energies_stat >= e_error_threshold,
                shrink=False,
            )
            error_e_structures_bg = np.ma.make_mask(
                energies_stat < maximum_value_e,
                shrink=False,
            )

            # Any True value in this array is over the energy error threshold
            # and must be sent to calculate with DFT.
            error_e_structures = np.logical_and(
                error_e_structures_sm, error_e_structures_bg
            )
            mdb_cud.custom_print(
                f'Extrapolating structures according to E: {error_e_structures}', 'none'
            )

            mdb_cud.custom_print('Printing extrapolation statistics for F...', 'none')

            # model_forces_dict shape: (1, n_frames, n_atoms, 3, n_models)
            # Shape: (1, n_frames, n_atoms, 3)
            forces_std = mdb_al_ut.get_model_forces_std(model_forces_dict)

            # Getting the magnitude for the force vector (Euclidean norm)
            # Shape: (1, n_frames, n_atoms)
            forces_std_norm = np.linalg.norm(forces_std, axis=3)

            # Keeping only the maximum force for every structure
            # Shape: (1, n_frames)
            forces_std_norm_max = np.amax(forces_std_norm, axis=2)

            maximum_value_f = np.average(forces_std_norm_max) * 10  # meV
            mdb_cud.custom_print(f'f_error_threshold: {f_error_threshold}', 'none')
            mdb_cud.custom_print(f'f_maximum_value: {maximum_value_f}', 'none')
            mdb_cud.custom_print(f'forces_std_norm_max: {forces_std_norm_max}', 'none')

            # Checking if the forces are over the error threshold
            err_f_struct_sm = np.ma.make_mask(
                forces_std_norm_max >= f_error_threshold,
                shrink=False,
            )
            err_f_struct_bg = np.ma.make_mask(
                forces_std_norm_max < maximum_value_f,
                shrink=False,
            )

            # Any True value in this array is over the force error threshold
            # and must be sent to calculate with DFT.
            error_f_structures = np.logical_and(err_f_struct_sm, err_f_struct_bg)

            mdb_cud.custom_print(
                f'Extrapolating structures according to F: {error_f_structures}', 'none'
            )

            # Adding extrapolating indices to list
            e_f_extrapol = []
            if isinstance(error_e_structures, np.ndarray):
                for err_idx, error in enumerate(error_e_structures[0]):
                    if error:
                        e_f_extrapol.append(short_mask[err_idx])
            if isinstance(error_f_structures, np.ndarray):
                for err_idx, error in enumerate(error_f_structures[0]):
                    if error:
                        e_f_extrapol.append(short_mask[err_idx])

            # Any index in this array is extrapolating and must
            # be sent to calculate with DFT.
            extrap_frame_idx.extend(set(e_f_extrapol))

        elif ef_disagreement_type == 'md_threshold':
            mdb_cud.custom_print(
                f'model_acc_multiplier: {model_acc_multiplier}', 'none'
            )

            # Get the energy and forces and save into arrays
            # shape: (n_frames, n_models)
            all_energies_array = np.array(
                [comm_results[model]['REF_energy'] for model in comm_results]
            ).T

            # Getting the standard deviation of the energies across
            # all models for each frame.
            energies_stat = np.nanstd(all_energies_array, axis=1, ddof=1)
            mdb_cud.custom_print(f'energies_stat: {energies_stat}', 'none')

            # Average variability in energy predictions across models for all frames
            e_mean_error = np.mean(energies_stat)
            mdb_cud.custom_print(f'e_mean_error: {e_mean_error}', 'none')

            # Spread of variability in the energy predictions across frames
            e_std_error = np.std(energies_stat)
            mdb_cud.custom_print(f'e_std_error: {e_std_error}', 'none')

            # Threshold for identifying frames with unusually high variability
            e_error_threshold = e_mean_error + 3 * e_std_error  # meV
            mdb_cud.custom_print(f'e_error_threshold: {e_error_threshold}', 'none')

            # Threshold for outliers
            e_maximum_value = e_mean_error + 10 * e_std_error  # meV
            mdb_cud.custom_print(f'e_maximum_value: {e_maximum_value}', 'none')

            # Checking if the energies are over the error threshold
            error_e_structures_sm = np.ma.make_mask(
                energies_stat >= e_error_threshold,
                shrink=False,
            )
            # Structures need to be below the maximum value,
            # otherwise they are considered as outliers and should be ignored.
            error_e_structures_bg = np.ma.make_mask(
                energies_stat < e_maximum_value,
                shrink=False,
            )

            # Marked structures should be both above the error threshold
            # and below the maximum value.
            # True values are the structures that are considered as extrapolation.
            error_e_structures = np.logical_and(
                error_e_structures_sm, error_e_structures_bg
            )
            mdb_cud.custom_print(
                f'Extrapolating structures according to E: {error_e_structures}', 'none'
            )

            mdb_cud.custom_print(
                'Printing extrapolation statistics for forces...', 'none'
            )
            # Array containing the forces for each model
            # shape: (3, n_at, n_frames, n_models)
            all_forces_array = np.array(
                [comm_results[model]['REF_forces'] for model in comm_results]
            ).T

            # Shape: (n_at, n_frames, n_models)
            f_magnitudes = np.linalg.norm(all_forces_array, axis=0)

            # Shape: (n_at, n_frames)
            f_std_devs = np.nanstd(f_magnitudes, axis=-1)

            # Shape: (n_frames)
            f_std_norm_max = np.amax(f_std_devs, axis=0)
            mdb_cud.custom_print(f'f_std_norm_max: {f_std_norm_max}', 'none')

            # Average variability in forces predictions across models for all frames
            f_mean_error = np.mean(f_std_norm_max)
            mdb_cud.custom_print(f'f_mean_error: {f_mean_error}', 'none')

            # Spread of variability in the forces predictions across frames
            f_std_error = np.std(f_std_norm_max)
            mdb_cud.custom_print(f'f_std_error: {f_std_error}', 'none')

            # Threshold for identifying frames with unusually high variability
            f_error_threshold = f_mean_error + 3 * f_std_error  # meV
            mdb_cud.custom_print(f'f_error_threshold meV: {f_error_threshold}', 'none')

            # Threshold for outliers
            f_maximum_value = f_mean_error + 10 * f_std_error  # meV
            mdb_cud.custom_print(f'f_maximum_value meV: {f_maximum_value}', 'none')

            # Checking if the forces are over the error threshold
            err_f_struct_sm = np.ma.make_mask(
                f_std_norm_max >= f_error_threshold,
                shrink=False,
            )
            err_f_struct_bg = np.ma.make_mask(
                f_std_norm_max < f_maximum_value,
                shrink=False,
            )

            # Marked structures should be both above the error threshold
            # and below the maximum value.
            # True values are the structures that are considered as extrapolation.
            error_f_structures = np.logical_and(err_f_struct_sm, err_f_struct_bg)
            mdb_cud.custom_print(
                f'Extrapolating structures according to F: {error_f_structures}', 'none'
            )

            # Adding extrapolating indices to list
            e_f_extrapol = []
            if isinstance(error_e_structures, np.ndarray):
                for err_idx, error in enumerate(error_e_structures):
                    if error:
                        e_f_extrapol.append(short_mask[err_idx])
            if isinstance(error_f_structures, np.ndarray):
                for err_idx, error in enumerate(error_f_structures):
                    if error:
                        e_f_extrapol.append(short_mask[err_idx])

            # Any index in this array is extrapolating and must
            # be sent to calculate with DFT.
            extrap_frame_idx.extend(set(e_f_extrapol))

        mdb_cud.custom_print(
            f'Extrapolating frames found by committee check: {len(extrap_frame_idx)}',
            'info',
        )

        ## Get the descriptors if extrapolation enabled
        if extrap_type != 'none':
            # Only MACE is supported for now
            mdb_cud.custom_print('Generating descriptors...', 'info')
            if settings['descriptors'].get('descriptor_type', 'mace') == 'soap':
                descriptor_dict, descriptor_arr = mdb_al_ut.generate_descriptors_soap(
                    database=md_traj_short,
                    descriptor_settings=settings['descriptors'],
                )
            else:
                descriptor_dict, descriptor_arr = mdb_al_ut.generate_descriptors_mace(
                    model_path=prepend_path / 'curr_model.model',
                    database=md_traj_short,
                    descriptor_settings=settings['descriptors'],
                )

            # Add is_extrapolating list which contains boolean values showing
            # if the frame is extrapolating or not
            for structure_uuid in descriptor_dict:
                descriptor_dict[structure_uuid]['is_extrapolating'] = np.zeros(
                    len(md_traj_short), dtype=bool
                )

        print()

        # Advanced extrapolation (concave hull) check
        if extrap_type == 'advanced':
            mdb_cud.custom_print('Applying advanced extrapolation check...', 'info')

            # Read the concave hull
            concave_hull = np.load(prepend_path / 'concave_hull.npy')

            # Get latent space for the trajectory
            if dim_red_method == 'pca':
                # latent_space = get_latent_space_pca(database=descriptor_dict)
                raise NotImplementedError('PCA not implemented yet.')
            else:
                aut_t_params = settings['descriptors']['autoencoder'].get(
                    'train_settings'
                )

                if (prepend_path / 'autoencoder_model.pth').exists():
                    model_path = prepend_path / 'autoencoder_model.pth'
                    model = torch.load(model_path)
                    mdb_cud.custom_print(f"Model loaded from:'{model_path}'")
                else:
                    model_path = prepend_path / aut_t_params.get('model_path')
                    model = torch.load(model_path)
                    mdb_cud.custom_print(f"Model loaded from:'{model_path}'")

                model.to(dtype=torch.float32)

                descriptor_dict = mdb_ae.get_latent_space_autoencoder(
                    model=model,
                    descriptor_dict=descriptor_dict,
                )

            point_inside, point_outside, all_points_in_out = check_traj_in_domain(
                concave_hull=concave_hull, descriptor_dict=descriptor_dict
            )

            mdb_chull.plot_concave_hull(
                concave_hull=concave_hull,
                point_inside=point_inside,
                point_outside=point_outside,
                filename=res_folder / f'concave_hull_temp-{curr_temp}.png',
            )

            for idx, frame in enumerate(all_points_in_out):
                if np.all(frame):
                    descriptor_dict[structure_uuid]['is_extrapolating'][idx] = False
                else:
                    descriptor_dict[structure_uuid]['is_extrapolating'][idx] = True

        # Simple extrapolation check (descriptor min/max)
        elif extrap_type == 'basic':
            # Read the minimum and maximum values for each descriptor
            # for the entire database
            curr_it_db_max = np.load(prepend_path / 'curr_it_db_max.npy')
            curr_it_db_min = np.load(prepend_path / 'curr_it_db_min.npy')

            descriptor_dict = simple_extrapolation_check(
                curr_it_db_max, curr_it_db_min, descriptor_dict
            )
        # Dont apply any further extrapolation check. Use the EF commitee check
        # already applied.
        elif extrap_type == 'none':
            mdb_cud.custom_print(
                ('No extrapolation check applied. Only EF commitee check applied.'),
                'warn',
            )

        elif extrap_type != 'none':
            for idx, is_extrapolating in enumerate(
                descriptor_dict[structure_uuid]['is_extrapolating']
            ):
                if is_extrapolating:
                    extrap_frame_idx.append(md_traj_short[idx].info['frame_idx'])

        # Saving all the frames that are extrapolating to a file
        extrap_frame_idx = set(extrap_frame_idx)
        extrapol_frames_final = [md_traj[i] for i in extrap_frame_idx]
        mdb_cud.custom_print(
            f'Total count of extrapolating frames: {len(extrapol_frames_final)}', 'info'
        )

        # Renaming result keys
        mod_extrap_frames = []
        for structure in extrapol_frames_final:
            # Setting the main model calculator for the current structure
            structure.calc = calculator

            # Getting the energy and forces for the structure
            structure.info['REF_energy'] = structure.get_potential_energy()
            structure.arrays['REF_forces'] = structure.get_forces()
            structure.info['mdb_id'] = uuid.uuid4()

            mod_extrap_frames.append(structure)

        ase_write(
            res_folder / 'extrapolating_frames.xyz',
            format='extxyz',
            images=mod_extrap_frames,
            append=True,
        )

    mdb_cud.custom_print('Structure processed!', 'done')
