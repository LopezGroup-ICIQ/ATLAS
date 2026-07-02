#!/usr/bin/env python3
"""Script to run MACE MD simulations, descriptor generation and extrapolation checks.

This script is part of ATLAS's active learning loop, and is used to combine
in one single calculation job the acquisition of the MD trajectory, the generation
of the descriptors followed by performing any extrapolation checks, filtering
the MD trajectory if necessary.
"""

import json
import logging
import pathlib as pl
import pickle
import tomllib
import uuid
import warnings

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter
from ase.neighborlist import natural_cutoffs
from mace.calculators import MACECalculator
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Point, Polygon

import atlas.active_learning.active_learning_utils as atl_al_ut
from atlas.active_learning.extrapolation import autoencoder as atl_ae
from atlas.active_learning.extrapolation.concave_hull import plot_concave_hull
from atlas.core import code_utils as atl_cut
from atlas.core.filtering import structure_filters as atl_str_filters

# Silencing specific warnings and log messages
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*weights_only.*')
warnings.filterwarnings('ignore')

# Force third party loggers to only show errors and critical messages
logging.getLogger('mace').setLevel(logging.ERROR)
logging.getLogger('e3nn').setLevel(logging.ERROR)


def check_traj_in_domain(
    concave_hull: np.ndarray,
    descriptor_dict: dict,
    hull_scale_factor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list | np.ndarray | None]:
    """
    Check if the generated descriptors are inside the precomputed concave hull.

    Parameters
    ----------
    concave_hull : np.ndarray
        Concave hull of the latent space for the database, corresponding
        to its convex or concave hull.
    descriptor_dict : dict
        Descriptor dictionary containing the descriptors for each frame.
        The structure is as follows:
        ```python
        {
            uuid: {
                'latent_space': np.ndarray,
                'descriptors': np.ndarray,
                'is_extrapolating': np.ndarray,
            }
        }
        ```
    hull_scale_factor : float, optional
        Tolerance percentage to enlarge the concave hull.
        For example, 0.1 adds 10% tolerance. Default is 0.0.

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
    # If the concave hull has multiple parts, create a MultiPolygon
    # from all the parts.
    if isinstance(concave_hull, list) and len(concave_hull) == 1:
        polygon = Polygon(concave_hull[0])
    elif isinstance(concave_hull, np.ndarray) and concave_hull.ndim == 2:
        polygon = Polygon(concave_hull)
    else:
        polygons = []
        for part in concave_hull:
            if len(part) >= 3:
                polygons.append(Polygon(part))
        polygon = MultiPolygon(polygons)

    # Scale the polygon if tolerance is provided
    scaled_hull_coords = None
    if hull_scale_factor > 0:
        scaling_factor = 1.0 + hull_scale_factor
        atl_cut.custom_print(
            f'Scaling concave hull by a factor of {scaling_factor} '
            'to apply tolerance...',
            'info',
        )
        polygon = scale(
            polygon, xfact=scaling_factor, yfact=scaling_factor, origin='center'
        )

        if isinstance(polygon, Polygon):
            scaled_hull_coords = np.array(polygon.exterior.coords)
        elif isinstance(polygon, MultiPolygon):
            scaled_hull_coords = [np.array(p.exterior.coords) for p in polygon.geoms]
    else:
        atl_cut.custom_print('No tolerance applied to concave hull.', 'warn')

    for c_uuid in descriptor_dict:
        descriptors = descriptor_dict[c_uuid]['latent_space']

        c_all_p = []
        for point in descriptors:
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
    return point_inside, point_outside, all_points_in_out, scaled_hull_coords


def define_allowed_stages(
    md_stages: dict,
    current_al_step: int,
    init_structure: Atoms,
) -> list[int]:
    """
    Define allowed MD stages based on the current active learning step.

    Parameters
    ----------
    md_stages : dict
        Dictionary containing the MD stages settings.
    current_al_step : int
        Current active learning iteration step.
    init_structure : Atoms
        Initial structure object, used to determine the type
        from its info dict, one of: 'bulk', 'surface' or 'cluster'

    Returns
    -------
    list[int]
        List of allowed stage names.
    """
    allowed_stages_names = []

    # Checking iteratively for every stage
    for stage_name, stage_settings in md_stages.items():
        # Get the two filtering types
        use_during_al_steps: str = stage_settings.get('use_during_al_steps')
        use_for_structure_types: list[str] = stage_settings.get(
            'use_for_structure_types'
        )

        steps_to_check_list = []

        # Only check stage if the `use_during_al_steps` key is defined
        if use_during_al_steps:
            # Parsing the string to get the steps or ranges of steps
            use_steps_list = use_during_al_steps.strip().split(',')
            for step_range in use_steps_list:
                step_range = step_range.strip()

                # Checking if it's a range of steps
                if '-' in step_range:
                    start_str, end_str = step_range.split('-')
                    start_it = int(start_str)
                    end_it = int(end_str)
                    step_range = range(start_it, end_it + 1)
                # Single step
                else:
                    start_it = int(step_range)
                    end_it = start_it
                    step_range = range(start_it, end_it + 1)

                steps_to_check_list.extend(step_range)

            # Converting to set to avoid duplicates
            steps_to_check_list = list(set(steps_to_check_list))

            # Checking if the current AL step is in the list of steps to check
            if current_al_step in steps_to_check_list:
                allowed_stages_names.append(stage_name)

        # Only check stage if the `use_for_structure_types` key is defined
        elif use_for_structure_types:
            # Check the keys corresponding to the given structure types, which will
            # be a bool marking if the structure is of that type
            for struct_type in use_for_structure_types:
                current_structure_type: bool = init_structure.info.get(struct_type)
                if current_structure_type:
                    allowed_stages_names.append(stage_name)

    return allowed_stages_names


def limit_md_frames(md_traj: Atoms, md_params: dict):
    """
    Limit the number of frames in the MD trajectory.

    Parameters
    ----------
    md_traj : Atoms
        MD trajectory.
    md_params : dict
        Dictionary containing the ATL settings for MD.

    Returns
    -------
    list
        List containing the MD trajectory with the limited number of frames.
    list
        Mask containing the indices of the frames that were kept.
    """
    # Limit total number of frames
    total_num_frames = atl_al_ut.get_total_num_frames(
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
    atl_cut.custom_print('Applying basic extrapolation check...', 'info')
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
    # The /atl_data directory should only exist in the containerized version
    # of the code. This conditional statement will get the correct path for
    # input and output files.
    if pl.Path('/atl_data').exists():
        prepend_path = pl.Path('/atl_data')
    else:
        prepend_path = pl.Path('.')

    # Initialize the logger
    log_folder = prepend_path / pl.Path('./logs')
    log_folder.mkdir(exist_ok=True)
    logger, log_filename = atl_cut.init_logger(
        source='proc_structure', log_path=log_folder
    )

    # Initialize random seed
    rng_seed = np.random.randint(0, ((2**32) - 1))
    np.random.seed(rng_seed)
    atl_cut.custom_print(f"Using random seed: '{rng_seed}'", logger=logger)

    atl_cut.custom_print('Starting process structure script...', 'info', logger=logger)

    # Load the rmse_arr.npy file and assign the values to the variables
    rmse_arr = np.load(prepend_path / 'rmse_arr.npy')

    # Best model RMSE for the E in meV/at
    e_rmse = rmse_arr[0]
    # Best model RMSE for the F in meV/A
    f_rmse = rmse_arr[1]

    atl_cut.custom_print(f'e_rmse: {e_rmse}, f_rmse: {f_rmse}', logger=logger)

    # Define results folder
    res_folder = prepend_path / pl.Path('./results')
    res_folder.mkdir(exist_ok=True)

    # Read TOML file with settings
    with open(prepend_path / 'settings.toml', 'rb') as f:
        settings = tomllib.load(f)

    # Parse settings
    md_params = settings.get('md', {}).get('parameters')

    # Adding key explicitly to display it in the log
    if not md_params.get('sample_frames_during_md'):
        md_params['sample_frames_during_md'] = False

    md_filters = settings.get('md', {}).get('filters', {})
    T_list = md_params['temperature_list_K']

    # Logging CUDA information
    enable_cueq = False
    if md_params.get('device') == 'cuda':
        atl_cut.custom_print(
            (
                f'CUDA INFO - available: {torch.cuda.is_available()}, '
                f'device_count: {torch.cuda.device_count()}, '
                f'current_device: {torch.cuda.current_device()}'
            ),
            'info',
            logger=logger,
        )

        if md_params.get('enable_cueq'):
            atl_cut.custom_print(
                'Using CUEQ to accelerate MD simulations...', 'info', logger=logger
            )
            enable_cueq = True

    # Get the EF disagreement type for interpolation
    ef_disagreement_type = settings.get('interpolation', {}).get(
        'disagreement_check_type', 'training'
    )

    # Get measure of chemical accuracy from settings
    target_acc_e = settings.get('interpolation', {}).get(
        'target_accuracy_e_meV_per_at', 43.0
    )
    target_acc_f = settings.get('interpolation', {}).get(
        'target_accuracy_f_meV_per_A', 50.0
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

    # Read the settings related to MD stages
    md_stages = md_params.get('stages', {})
    current_al_step = settings.get('active_learning', {}).get('current_iteration', 0)

    # Define allowed stages based on current AL step
    allowed_stage_names = define_allowed_stages(
        md_stages=md_stages,
        current_al_step=current_al_step,
        init_structure=init_conf_orig,
    )

    md_stages_allowed = {}
    if allowed_stage_names:
        for stage in allowed_stage_names:
            md_stages_allowed[stage] = md_stages[stage]
    md_stages = md_stages_allowed

    md_stage_order = None
    if md_stages:
        md_stage_order = md_params.get('md_stage_order', [])

    if not md_stage_order:
        md_stage_order = list(md_stages_allowed.keys())
        atl_cut.custom_print(
            'No MD stage order defined. Using default order', 'info', logger=logger
        )

    ## Running MD simulations for given temperatures
    for T_start in T_list:
        traj_filename = res_folder / f'md_traj_final_temp-{T_start}.traj'

        # Not repeating MD simulations if the trajectory already exists.
        # Mainly for testing, this should not happen during a normal run.
        if pl.Path(traj_filename).exists():
            atl_cut.custom_print(
                f"MD trajectory for 'T={T_start} K' already exists. Skipping...",
                'warn',
                logger=logger,
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

        if not md_stages:
            atl_cut.custom_print('No MD stages found!', 'info', logger=logger)
            atl_cut.custom_print(
                f"Running MD simulation starting at 'T={T_start} K'",
                'info',
                logger=logger,
            )
            atl_al_ut.run_mace_md_ase(
                md_params=md_params,
                T_start=T_start,
                traj_obj=traj_obj,
                init_conf=init_conf,
                prepend_path=prepend_path,
                explode_filter_dict=md_filters.get('exploding_structures', {}),
                enable_cueq=enable_cueq,
            )
            atl_cut.custom_print('MD simulation completed!', 'done', logger=logger)
        else:
            orig_md_params = md_params.copy()
            for stage_name in md_stage_order:
                # Update current MD parameters
                curr_stage = md_stages_allowed.get(stage_name, {})

                if curr_stage:
                    md_params = orig_md_params.copy()
                    for key, value in curr_stage.items():
                        md_params[key] = value

                    T_list_stage = md_params['temperature_list_K']
                    for T_start_stage in T_list_stage:
                        atl_cut.custom_print(
                            f"Running MD stage '{stage_name}' simulation starting at "
                            f'T={T_start_stage} K',
                            'info',
                            logger=logger,
                        )

                        # Running MD.
                        # Since init_conf is updated during the MD run, the run
                        # can be continued from the last frame of the previous stage
                        # without any modification
                        atl_al_ut.run_mace_md_ase(
                            md_params=md_params,
                            T_start=T_start_stage,
                            traj_obj=traj_obj,
                            init_conf=init_conf,
                            prepend_path=prepend_path,
                            explode_filter_dict=md_filters.get(
                                'exploding_structures', {}
                            ),
                            enable_cueq=enable_cueq,
                            stage_name=stage_name,
                        )

                else:
                    atl_cut.custom_print(
                        f"MD stage '{stage_name}' not found in allowed stages. "
                        'Skipping...',
                        'warn',
                        logger=logger,
                    )
                print()

    # Read MD-generated trajectories for given temperatures
    traj_files = res_folder.glob('*final_temp-*.traj')

    for curr_traj in traj_files:
        print()
        atl_cut.custom_print(
            f"Checking out of domain frames for '{curr_traj}'", 'info', logger=logger
        )
        out_of_domain_frame_idx = []

        curr_temp = float(str(curr_traj).split('temp-')[1].split('.traj')[0])

        # Read the trajectory
        md_traj = [frame for frame in TrajectoryReader(curr_traj)]
        orig_md_size = len(md_traj)
        atl_cut.custom_print(
            f'Trajectory length: {orig_md_size}', 'info', logger=logger
        )

        # Add frame info and overwrite the trajectory
        for frame_idx, frame in enumerate(md_traj):
            frame.info['frame_idx'] = frame_idx
            frame.info['calc_type'] = 'MACE_MD'

        # Check for cached short trajectory indices
        cache_indices_file = res_folder / f'md_traj_short_indices_temp-{curr_temp}.txt'
        md_traj_short = None
        short_mask = None

        if cache_indices_file.exists():
            atl_cut.custom_print(
                f"Found cached short trajectory indices at '{cache_indices_file}'. "
                'Skipping filtering and sampling...',
                'info',
                logger=logger,
            )
            with open(cache_indices_file) as f:
                indices = [int(line.strip()) for line in f if line.strip()]

            md_traj_short = [md_traj[i] for i in indices]
            # Use absolute frame indices
            short_mask = indices
            # Define md_traj_filtered for logging purposes
            md_traj_filtered = md_traj_short

        if md_traj_short is None:
            # Apply MD filters and removing these frames from the trajectory
            frames_to_remove = []
            atl_cut.custom_print(
                'Applying MD filters to remove outliers...', 'info', logger=logger
            )

            if md_filters.get('layer_distance', {}).get('enable'):
                atl_cut.custom_print(
                    "Running 'layer distance' filter...", 'info', logger=logger
                )
                later_distance_r_frames = []

                # Getting max distance from input
                max_dist: float = md_filters['layer_distance']['max_layer_distance_ang']

                for idx, frame in enumerate(md_traj):
                    is_structure_wrong = atl_str_filters.apply_filter_layer_distance(
                        struct=frame, max_layer_distance_ang=max_dist
                    )
                    if is_structure_wrong:
                        later_distance_r_frames.append(idx)
                frames_to_remove.extend(later_distance_r_frames)

                atl_cut.custom_print(
                    f'Marked by layer distance filter: {len(later_distance_r_frames)}',
                    'debug',
                )

            exploding_structs = []
            if md_filters.get('exploding_structures', {}).get('enable'):
                atl_cut.custom_print(
                    "Running 'exploding structures' filter...", 'info', logger=logger
                )

                explod_filt_settings = md_filters.get('exploding_structures', {})
                # Getting multiplier from input
                cov_rad_multiplier_max: float = explod_filt_settings.get(
                    'cov_rad_multiplier_max', 10.0
                )
                cov_rad_multiplier_min: float = explod_filt_settings.get(
                    'cov_rad_multiplier_min', 0.8
                )
                max_T = curr_temp * md_params.get('max_temp_multiplier', 1)
                max_T_multiplier = explod_filt_settings.get('max_T_multiplier', 10)
                remove_positive_E = explod_filt_settings.get('remove_positive_E', False)
                max_F: float = explod_filt_settings.get('max_F', 25.0)
                max_V: float = explod_filt_settings.get('max_V', 2.0)

                # Precomputing cutoffs once before the loop
                base_structure = md_traj[0]
                base_cutoffs = np.array(natural_cutoffs(base_structure))
                cutoffs_max_base = base_cutoffs * cov_rad_multiplier_max
                cutoffs_min_base = base_cutoffs * cov_rad_multiplier_min

                # Applying filter for every frame
                for idx, frame in enumerate(md_traj):
                    is_structure_wrong: bool = (
                        atl_str_filters.apply_filter_exploding_structures(
                            struct=frame,
                            max_F=max_F,
                            max_V=max_V,
                            max_T=max_T,
                            T_list=[frame.info.get('md_temperature')],
                            max_T_multiplier=max_T_multiplier,
                            remove_positive_E=remove_positive_E,
                        )
                    )
                    if is_structure_wrong:
                        exploding_structs.append(idx)

                frames_to_remove.extend(exploding_structs)

                atl_cut.custom_print(
                    f'Marked by exploding structures filter: {len(exploding_structs)}',
                    'debug',
                    logger=logger,
                )

            if md_filters.get('check_atoms_no_neighbor', {}).get('enable'):
                neighbor_r_frames = []
                atl_cut.custom_print(
                    "Running 'no neighbor' filter...", 'info', logger=logger
                )

                # Getting multiplier from input
                cov_rad_mult: float = md_filters.get('check_atoms_no_neighbor', {}).get(
                    'covalent_radius_multiplier', 1.0
                )

                # Precompute the maximum allowed Z-thickness
                base_structure = md_traj[0]
                initial_z_thickness = np.ptp(base_structure.positions[:, 2])
                expansion_buffer = 10.0  # Set your 5-10 Å buffer here
                max_allowed_thickness = initial_z_thickness + expansion_buffer

                # Applying filter for every frame
                for idx, frame in enumerate(md_traj):
                    is_structure_wrong = atl_str_filters.apply_filter_evaporation(
                        struct=frame,
                        max_allowed_thickness=max_allowed_thickness,
                    )
                    if is_structure_wrong:
                        neighbor_r_frames.append(idx)

                atl_cut.custom_print(
                    f'Marked by no neighbor filter: {len(neighbor_r_frames)}',
                    'debug',
                    logger=logger,
                )

            # Remove duplicate frames
            frames_to_remove = list(set(frames_to_remove))

            # Create a new list excluding the frames to remove
            md_traj_filtered = [
                frame for i, frame in enumerate(md_traj) if i not in frames_to_remove
            ]

            atl_cut.custom_print(
                f'Trajectory length after MD filters: {len(md_traj_filtered)}',
                'info',
                logger=logger,
            )

            if len(md_traj_filtered) == 0:
                atl_cut.custom_print(
                    (
                        'No MD frames left after filtering. '
                        'This means that probably there are a lot of '
                        'unrealistic structures. '
                        'Check the training data and models used to run this MD.'
                    ),
                    'warning',
                    logger=logger,
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
                atl_cut.custom_print(
                    (
                        f'Saving {len(frames_to_remove)} filtered structures'
                        f" to '{res_folder / 'filtered_frames.xyz'}'"
                    ),
                    'info',
                    logger=logger,
                )
                filtered_frames = [md_traj[i] for i in frames_to_remove]
                ase_write(
                    res_folder / 'filtered_frames.xyz',
                    format='extxyz',
                    images=filtered_frames,
                    append=True,
                )
            else:
                atl_cut.custom_print(
                    'Filtered structures not saved.', 'info', logger=logger
                )

            # Limit total number of frames if sampling during md is disabled
            if not md_params.get('sample_frames_during_md'):
                md_traj_short, short_mask = limit_md_frames(md_traj_filtered, md_params)
                atl_cut.custom_print(
                    f'Limited number of frames to: {len(md_traj_short)}',
                    'info',
                    logger=logger,
                )
            else:
                md_traj_short = md_traj_filtered
                short_mask = np.arange(len(md_traj_short))
                atl_cut.custom_print(
                    'Trajectory already shortened during MD by save interval. '
                    f'Length unchanged. Current length: {len(md_traj_short)}',
                    'info',
                    logger=logger,
                )

            # Save cache
            short_mask = [f.info['frame_idx'] for f in md_traj_short]
            with open(cache_indices_file, 'w') as f:
                for idx in short_mask:
                    f.write(f'{idx}\n')

        # Running evaluation of the energies and forces using each commitee model
        atl_cut.custom_print('Running committee evaluation...', 'info', logger=logger)
        model_file_list = list(prepend_path.glob('*.model'))
        comm_settings = settings.get('committee_eval', {})
        comm_results = {}
        for model in model_file_list:
            comm_results[model.stem] = {'REF_energy': [], 'REF_forces': []}

            # Use torch.load with map_location to ensure model loads
            # on the correct device
            device_str = comm_settings.get('mace', {}).get('device', 'cpu')
            model_path = prepend_path / model
            model_loaded = torch.load(model_path, map_location=torch.device(device_str))

            calculator = MACECalculator(
                models=[model_loaded],
                device=device_str,
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
        model_acc_multiplier = settings.get('interpolation', {}).get(
            'model_acc_multiplier', 1
        )

        atl_cut.custom_print(
            f"Using disagreement check type: '{ef_disagreement_type}'",
            'info',
            logger=logger,
        )
        atl_cut.custom_print(
            'Printing interpolation statistics for E:', 'debug', logger=logger
        )

        atl_cut.custom_print(f'E RMSE: {e_rmse}', 'debug', logger=logger)
        atl_cut.custom_print(f'F RMSE: {f_rmse}', 'debug', logger=logger)

        if ef_disagreement_type == 'training':
            # model_acc_multiplier is equivalent to lambda in our reference

            atl_cut.custom_print(f'λ = {model_acc_multiplier}', 'none', logger=logger)
            atl_cut.custom_print(
                f'λ · RMSE_E = {model_acc_multiplier * e_rmse}', 'none', logger=logger
            )
            atl_cut.custom_print(
                f'λ · RMSE_F = {model_acc_multiplier * f_rmse}', 'none', logger=logger
            )

            # These error thresholds will decide if a MD frame is interpolating or not.
            # meV/at
            e_error_threshold = max(model_acc_multiplier * e_rmse, target_acc_e)

            # meV / A
            f_error_threshold = max(model_acc_multiplier * f_rmse, target_acc_f)

            atl_cut.custom_print(
                f'θ_E = max(λ · RMSE_E, target_acc_e) = {e_error_threshold}',
                'none',
                logger=logger,
            )
            atl_cut.custom_print(
                f'θ_F = max(λ · RMSE_F, target_acc_f) = {f_error_threshold}',
                'none',
                logger=logger,
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
            energies_stat = atl_al_ut.get_model_energies_std(model_energies_dict)  # meV

            # maximum_value_e = np.average(energies_stat) * 10  # meV
            maximum_value_e = e_error_threshold * 10  # meV

            atl_cut.custom_print(
                f'e_error_threshold: {e_error_threshold}', 'none', logger=logger
            )
            atl_cut.custom_print(
                f'e_maximum_value: {maximum_value_e}', 'none', logger=logger
            )
            atl_cut.custom_print(
                f'energies_stat: {energies_stat}', 'none', logger=logger
            )

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
            atl_cut.custom_print(
                f'Interpolation error structures according to E: {error_e_structures}',
                'none',
                logger=logger,
            )

            atl_cut.custom_print(
                'Printing interpolation statistics for F...', 'none', logger=logger
            )

            # model_forces_dict shape: (1, n_frames, n_atoms, 3, n_models)
            # Shape: (1, n_frames, n_atoms, 3)
            forces_std = atl_al_ut.get_model_forces_std(model_forces_dict)

            # Getting the magnitude for the force vector (Euclidean norm)
            # Shape: (1, n_frames, n_atoms)
            forces_std_norm = np.linalg.norm(forces_std, axis=3)

            # Keeping only the maximum force for every structure
            # Shape: (1, n_frames)
            forces_std_norm_max = np.amax(forces_std_norm, axis=2)

            # maximum_value_f = np.average(forces_std_norm_max) * 10  # meV
            maximum_value_f = f_error_threshold * 10  # meV

            atl_cut.custom_print(
                f'f_error_threshold: {f_error_threshold}', 'none', logger=logger
            )
            atl_cut.custom_print(
                f'f_maximum_value: {maximum_value_f}', 'none', logger=logger
            )
            atl_cut.custom_print(
                f'forces_std_norm_max: {forces_std_norm_max}', 'none', logger=logger
            )

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

            atl_cut.custom_print(
                f'Interpolation error structures according to F: {error_f_structures}',
                'none',
                logger=logger,
            )

            # Adding extrapolating indices to list
            e_f_interpolation = []
            if isinstance(error_e_structures, np.ndarray):
                for err_idx, error in enumerate(error_e_structures[0]):
                    if error:
                        e_f_interpolation.append(short_mask[err_idx])
            if isinstance(error_f_structures, np.ndarray):
                for err_idx, error in enumerate(error_f_structures[0]):
                    if error:
                        e_f_interpolation.append(short_mask[err_idx])

            # Any index in this array is extrapolating and must
            # be sent to calculate with DFT.
            num_interpolating_frames = len(set(e_f_interpolation))
            out_of_domain_frame_idx.extend(set(e_f_interpolation))

        elif ef_disagreement_type == 'md_threshold':
            atl_cut.custom_print(
                f'model_acc_multiplier: {model_acc_multiplier}', 'none', logger=logger
            )

            # Get the energy and forces and save into arrays
            # shape: (n_frames, n_models)
            all_energies_array = np.array(
                [comm_results[model]['REF_energy'] for model in comm_results]
            ).T

            # Getting the standard deviation of the energies across
            # all models for each frame.
            energies_stat = np.nanstd(all_energies_array, axis=1, ddof=1)
            atl_cut.custom_print(
                f'energies_stat: {energies_stat}', 'none', logger=logger
            )

            # Average variability in energy predictions across models for all frames
            e_mean_error = np.mean(energies_stat)
            atl_cut.custom_print(f'e_mean_error: {e_mean_error}', 'none', logger=logger)

            # Spread of variability in the energy predictions across frames
            e_std_error = np.std(energies_stat)
            atl_cut.custom_print(f'e_std_error: {e_std_error}', 'none', logger=logger)

            # Threshold for identifying frames with unusually high variability
            e_error_threshold = e_mean_error + 3 * e_std_error  # meV
            atl_cut.custom_print(
                f'e_error_threshold: {e_error_threshold}', 'none', logger=logger
            )

            # Threshold for outliers
            e_maximum_value = e_mean_error + 10 * e_std_error  # meV
            atl_cut.custom_print(
                f'e_maximum_value: {e_maximum_value}', 'none', logger=logger
            )

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
            atl_cut.custom_print(
                f'Interpolation error structures according to E: {error_e_structures}',
                'none',
                logger=logger,
            )

            atl_cut.custom_print(
                'Printing interpolation statistics for forces...', 'none', logger=logger
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
            atl_cut.custom_print(
                f'f_std_norm_max: {f_std_norm_max}', 'none', logger=logger
            )

            # Average variability in forces predictions across models for all frames
            f_mean_error = np.mean(f_std_norm_max)
            atl_cut.custom_print(f'f_mean_error: {f_mean_error}', 'none', logger=logger)

            # Spread of variability in the forces predictions across frames
            f_std_error = np.std(f_std_norm_max)
            atl_cut.custom_print(f'f_std_error: {f_std_error}', 'none', logger=logger)

            # REMOVE
            # f_error_threshold = 103.71869367415634  # meV

            # Threshold for identifying frames with unusually high variability
            f_error_threshold = f_mean_error + 3 * f_std_error  # meV
            atl_cut.custom_print(
                f'f_error_threshold meV: {f_error_threshold}', 'none', logger=logger
            )

            # Threshold for outliers
            f_maximum_value = f_mean_error + 10 * f_std_error  # meV
            atl_cut.custom_print(
                f'f_maximum_value meV: {f_maximum_value}', 'none', logger=logger
            )

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
            atl_cut.custom_print(
                f'Interpolation error structures according to F: {error_f_structures}',
                'none',
                logger=logger,
            )

            # Adding extrapolating indices to list
            e_f_interpolation = []
            if isinstance(error_e_structures, np.ndarray):
                for err_idx, error in enumerate(error_e_structures):
                    if error:
                        e_f_interpolation.append(short_mask[err_idx])
            if isinstance(error_f_structures, np.ndarray):
                for err_idx, error in enumerate(error_f_structures):
                    if error:
                        e_f_interpolation.append(short_mask[err_idx])

            # Any index in this array is extrapolating and must
            # be sent to calculate with DFT.
            num_interpolating_frames = len(set(e_f_interpolation))
            out_of_domain_frame_idx.extend(set(e_f_interpolation))

        atl_cut.custom_print(
            'Frames with committee disagreement found by committee check: '
            f'{num_interpolating_frames}',
            'info',
            logger=logger,
        )

        # EXTRAPOLATION CHECKING #
        ## Get the descriptors if any type of extrapolation is enabled
        extrapolating_frames = []
        if extrap_type != 'none':
            # Only MACE is supported for now
            atl_cut.custom_print('Generating descriptors...', 'info', logger=logger)
            if settings['descriptors'].get('descriptor_type', 'mace') == 'soap':
                descriptor_dict, descriptor_arr, uuids = (
                    atl_al_ut.generate_descriptors_soap(
                        database=md_traj_short,
                        descriptor_settings=settings['descriptors'],
                    )
                )
            else:
                descriptor_dict, descriptor_arr, uuids = (
                    atl_al_ut.generate_descriptors_mace(
                        model_path=prepend_path / 'curr_model.model',
                        database=md_traj_short,
                        descriptor_settings=settings['descriptors'],
                    )
                )

            # Add is_extrapolating list which contains boolean values showing
            # if the frame is extrapolating or not
            for structure_uuid in descriptor_dict:
                descriptor_dict[structure_uuid]['is_extrapolating'] = np.zeros(
                    len(md_traj_short), dtype=bool
                )

        print()
        # Advanced extrapolation (concave hull) check
        if extrap_type in ['advanced', 'alpha-shape']:
            atl_cut.custom_print(
                'Applying advanced extrapolation check...', 'info', logger=logger
            )

            # Read the concave hull. If it fails, notify the user and proceed
            # without extrapolation check.
            if (prepend_path / 'concave_hull.npy').exists():
                concave_hull = np.load(prepend_path / 'concave_hull.npy')
            elif (prepend_path / 'concave_hulls.pkl').exists():
                with open(prepend_path / 'concave_hulls.pkl', 'rb') as f:
                    concave_hull = []
                    concave_hull_tuples = pickle.load(f)
                    for hull in concave_hull_tuples:
                        concave_hull.append(np.array(hull))
            else:
                atl_cut.custom_print(
                    (
                        'Concave hull file not found! '
                        'Please make sure that the extrapolation check has finished '
                        'correctly during the active learning loop. Proceeding '
                        'without extrapolation check...'
                    ),
                    'error',
                    logger=logger,
                )
                extrap_type = 'none'
                break

            # Get latent space for the trajectory
            match dim_red_method:
                case 'pca':
                    # latent_space = get_latent_space_pca(database=descriptor_dict)
                    raise NotImplementedError('PCA not implemented yet.')
                case 'autoencoder':
                    aut_t_params = settings['descriptors']['autoencoder'].get(
                        'train_settings'
                    )

                    if (prepend_path / 'autoencoder_model.pth').exists():
                        model_path = prepend_path / 'autoencoder_model.pth'
                        model = torch.load(model_path)
                        atl_cut.custom_print(
                            f"Model loaded from:'{model_path}'", logger=logger
                        )
                    else:
                        model_path = prepend_path / aut_t_params.get('model_path')
                        model = torch.load(model_path)
                        atl_cut.custom_print(
                            f"Model loaded from:'{model_path}'", logger=logger
                        )

                    model.to(dtype=torch.float32)

                    descriptor_dict = atl_ae.get_latent_space_autoencoder(
                        model=model,
                        descriptor_dict=descriptor_dict,
                    )
                case 'none' | _:
                    atl_cut.custom_print(
                        'Dimensionality method not specified or not recognized!',
                        'warn',
                        logger=logger,
                    )

            # Scale factor for the concave hull tolerance
            concave_hull_scale_factor = settings['extrapolation'].get(
                'concave_hull_tolerance_scale_factor', 0.0
            )

            hull_scale_factor = (
                settings.get('extrapolation', {})
                .get('concave_hull', {})
                .get('concave_hull_scale_factor', 0.0)
            )
            (
                point_inside,
                point_outside,
                all_points_in_out,
                scaled_hull,
            ) = check_traj_in_domain(
                concave_hull=concave_hull,
                descriptor_dict=descriptor_dict,
                hull_scale_factor=hull_scale_factor,
            )

            plot_concave_hull(
                concave_hull=concave_hull,
                point_inside=point_inside,
                point_outside=point_outside,
                scaled_hull=scaled_hull,
                filename=res_folder / f'concave_hull_temp-{curr_temp}.png',
            )

            for uuid, frame in zip(uuids, all_points_in_out, strict=False):
                descriptor_dict[uuid]['is_extrapolating'] = not np.all(frame)

        # Simple extrapolation check (descriptor min/max)
        elif extrap_type in ['basic', 'min-max']:
            # Read the minimum and maximum values for each descriptor
            # for the entire database
            curr_it_db_max = np.load(prepend_path / 'curr_it_db_max.npy')
            curr_it_db_min = np.load(prepend_path / 'curr_it_db_min.npy')

            descriptor_dict = simple_extrapolation_check(
                curr_it_db_max, curr_it_db_min, descriptor_dict
            )
        # Dont apply any further extrapolation check. Use the EF commitee check
        # already applied.
        elif extrap_type in ['none', 'disabled', None]:
            atl_cut.custom_print(
                (
                    'No extrapolation check applied. Only interpolation (EF commitee) '
                    'check applied.'
                ),
                'warn',
                logger=logger,
            )

        # Gathering indices of extrapolating frames
        # according to the extrapolation check results
        if extrap_type != 'none':
            for idx, uuid in enumerate(uuids):
                if descriptor_dict[uuid]['is_extrapolating']:
                    curr_frame_idx = md_traj_short[idx].info['frame_idx']
                    extrapolating_frames.append(curr_frame_idx)
                    out_of_domain_frame_idx.append(curr_frame_idx)

        # Saving all the frames that are out of domain to a file
        out_of_domain_frame_idx = set(out_of_domain_frame_idx)
        out_of_domain_frames_struct = [md_traj[i] for i in out_of_domain_frame_idx]
        atl_cut.custom_print(
            f'Total count of extrapolating frames: {len(extrapolating_frames)}',
            'info',
            logger=logger,
        )

        # Renaming result keys
        out_of_domain_frames_final = []
        for structure in out_of_domain_frames_struct:
            # Setting the main model calculator for the current structure
            structure.calc = calculator

            # Getting the energy and forces for the structure
            structure.info['REF_energy'] = structure.get_potential_energy()
            structure.arrays['REF_forces'] = structure.get_forces()
            structure.info['atl_id'] = str(uuid.uuid4())

            out_of_domain_frames_final.append(structure)

        ase_write(
            res_folder / 'extrapolating_frames.xyz',
            format='extxyz',
            images=out_of_domain_frames_final,
            append=True,
        )

        # Creating empty concave hull plot if it does not exist
        # to avoid gathering errors with aiida
        plot_path = res_folder / f'concave_hull_temp-{curr_temp}.png'
        if not plot_path.exists():
            plot_path.touch()

        uq_stats_dict: dict = {
            'total_frames': orig_md_size,
            'frames_after_filters': len(md_traj_filtered),
            'extrapolation_error_frames': len(set(extrapolating_frames)),
            'interpolation_error_frames': num_interpolating_frames,
            'out_of_domain_frames': len(out_of_domain_frames_final),
        }

        atl_cut.custom_print(f'UQ statistics: {uq_stats_dict}', 'info', logger=logger)

        with open(res_folder / 'uq_stats.json', 'w+') as f:
            json.dump(
                obj=uq_stats_dict,
                fp=f,
                indent=4,
            )

    atl_cut.custom_print('Structure processed!', 'done', logger=logger)
