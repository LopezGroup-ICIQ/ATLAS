#!/usr/bin/env python3
"""Script to run safeguard MD, descriptor generation and extrapolation checks.

This script is part of ATLAS's active learning loop, and is used to check
if the current iteration of the model is robust enough before early stopping.
"""

import gc
import logging
import pathlib as pl
import pickle
import tomllib
import uuid
import warnings

import numpy as np
import torch
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter
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
    np.ndarray or list
        Scaled hull coordinates if hull_scale_factor > 0, otherwise None.
    """
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    # If the concave hull has multiple parts, create a MultiPolygon
    # from all the parts.
    if len(concave_hull) == 1:
        polygon = Polygon(concave_hull[0])
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

    for c_uuid in descriptor_dict:
        # Get the [x, y] point directly.
        # .squeeze() ensures it's a flat 1D array even if the model outputs shape (1, 2)
        # latent_point.shape: (1, 2) -> (2,)
        latent_point = np.array(descriptor_dict[c_uuid]['latent_space']).squeeze()

        # Create the Shapely point directly
        p = Point(latent_point)

        # Check if the point is inside the hull
        if polygon.contains(p):
            point_inside.append(latent_point)
            all_points_in_out.append(True)
        else:
            point_outside.append(latent_point)
            all_points_in_out.append(False)

    # array shape: (n_points, 2)
    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)

    # 1D array of booleans
    # e.g., array([ True,  False,  True,  False,  True, ...])
    all_points_in_out = np.array(all_points_in_out)

    return point_inside, point_outside, all_points_in_out, scaled_hull_coords


def limit_md_frames(md_traj, frames_to_keep: list, md_params: dict):
    """
    Limit the number of frames in the MD trajectory.

    Parameters
    ----------
    md_traj : ase.Atoms | TrajectoryReader
        ASE Atoms object or TrajectoryReader containing the MD trajectory.
    frames_to_keep : list
        List containing the indices of the frames that were kept after applying
        the MD filters.
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
        len_traj=len(frames_to_keep),
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
        # Get the descriptors for this specific frame
        frame_descriptors = val['descriptors']

        # Check bounds
        below_min = frame_descriptors < curr_it_db_min
        above_max = frame_descriptors > curr_it_db_max
        is_frame_extrapolating = np.any(np.logical_or(below_min, above_max))

        # Assign the boolean directly
        if is_frame_extrapolating:
            descriptor_dict[curr_uuid]['is_extrapolating'] = True
        else:
            descriptor_dict[curr_uuid]['is_extrapolating'] = False

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
    logger, log_filename = atl_cut.init_logger(source='safeguard', log_path=log_folder)

    # Initialize random seed
    rng_seed = np.random.randint(0, ((2**32) - 1))
    np.random.seed(rng_seed)
    atl_cut.custom_print(f"Using random seed: '{rng_seed}'", logger=logger)

    atl_cut.custom_print('Starting safeguard MD script...', 'info', logger=logger)

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

    # Initialize the logger
    log_folder = prepend_path / pl.Path('./logs')
    log_folder.mkdir(exist_ok=True)

    # Read TOML file with settings
    with open(prepend_path / 'settings.toml', 'rb') as f:
        settings = tomllib.load(f)

    # Parse settings
    safe_md_params = settings.get('safeguard', {}).get('md', {}).get('parameters')

    if safe_md_params is None:
        safe_md_params = settings.get('safeguard', {}).get('md_parameters')

    # Adding key explicitly to display it in the log
    if not safe_md_params.get('sample_frames_during_md'):
        safe_md_params['sample_frames_during_md'] = False

    md_filters = settings.get('md', {}).get('filters', {})
    T_list = safe_md_params['temperature_list_K']

    # Logging CUDA information
    enable_cueq = False
    if safe_md_params.get('device') == 'cuda':
        atl_cut.custom_print(
            (
                f'CUDA INFO - available: {torch.cuda.is_available()}, '
                f'device_count: {torch.cuda.device_count()}, '
                f'current_device: {torch.cuda.current_device()}'
            ),
            'info',
            logger=logger,
        )

        if safe_md_params.get('enable_cueq'):
            atl_cut.custom_print(
                'Using CUEQ to accelerate MD simulations...', 'info', logger=logger
            )
            enable_cueq = True

    # Get the EF disagreement type
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
    try:
        init_conf_orig = ase_read(prepend_path / 'curr_structure.xyz', format='extxyz')
    except Exception as e:
        atl_cut.custom_print(
            f'Error reading initial structure: {e}', 'error', logger=logger
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
            properties=['energy', 'forces', 'REF_energy', 'REF_forces', 'MLIP_energy'],
        )
        print()
        atl_cut.custom_print(
            f"Running MD simulation for 'T={T_start} K'", 'info', logger=logger
        )
        atl_al_ut.run_mace_md_ase(
            md_params=safe_md_params,
            T_start=T_start,
            traj_obj=traj_obj,
            init_conf=init_conf,
            prepend_path=prepend_path,
            explode_filter_dict=md_filters.get('exploding_structures', {}),
            enable_cueq=enable_cueq,
            model_name='sampler_model.model',
        )
        atl_cut.custom_print('MD simulation completed!', 'done', logger=logger)

    # Read MD-generated trajectories for given temperatures
    traj_files = res_folder.glob('*final_temp-*.traj')

    for curr_traj in traj_files:
        print()
        atl_cut.custom_print(
            f"Checking extrapolating frames for '{curr_traj}'", 'info', logger=logger
        )

        curr_temp = float(str(curr_traj).split('temp-')[1].split('.traj')[0])

        # Read the trajectory
        md_traj = TrajectoryReader(curr_traj)
        # md_traj = [frame for frame in TrajectoryReader(curr_traj)]

        orig_md_size = len(md_traj)
        atl_cut.custom_print(
            f'Trajectory length: {orig_md_size}', 'info', logger=logger
        )

        # Apply MD filters and removing these frames from the trajectory
        layer_dist_enabled = md_filters.get('layer_distance', {}).get('enable', False)
        explode_enabled = md_filters.get('exploding_structures', {}).get(
            'enable', False
        )
        evap_enabled = md_filters.get('check_atoms_no_neighbor', {}).get(
            'enable', False
        )

        # Precompute values needed for filters
        if layer_dist_enabled:
            atl_cut.custom_print(
                'Layer distance filter enabled.', 'info', logger=logger
            )
            max_dist: float = md_filters['layer_distance']['max_layer_distance_ang']

        if explode_enabled:
            explod_filt_settings = md_filters.get('exploding_structures', {})
            max_T = curr_temp * safe_md_params.get('max_temp_multiplier', 1)
            max_T_multiplier = explod_filt_settings.get('max_T_multiplier', 10)
            remove_positive_E = explod_filt_settings.get('remove_positive_E', False)
            max_F: float = explod_filt_settings.get('max_F', 25.0)
            max_V: float = explod_filt_settings.get('max_V', 2.0)

        if evap_enabled:
            initial_z_thickness = np.ptp(md_traj[0].positions[:, 2])
            expansion_buffer = 10.0
            max_allowed_thickness = initial_z_thickness + expansion_buffer

        # Single loop with early exits, fastest path for each frame
        atl_cut.custom_print(
            'Applying MD filters to remove outliers...', 'info', logger=logger
        )
        frames_to_remove = set()
        explode_count = 0
        layer_dist_count = 0
        evap_count = 0

        iterable = atl_cut.atl_show_progress(
            enumerate(md_traj),
            total=orig_md_size,
            interval=1000,
            prepend='Filter:',
        )

        for idx, frame in iterable:
            # Add frame info and overwrite the trajectory
            # for frame_idx, frame in enumerate(md_traj):
            frame.info['frame_idx'] = idx
            frame.info['calc_type'] = 'MACE_MD'

            # Check exploding structures first (fast: dict lookups + array ops)
            if explode_enabled and atl_str_filters.apply_filter_exploding_structures(
                struct=frame,
                max_F=max_F,
                max_V=max_V,
                max_T=max_T,
                T_list=[frame.info.get('md_temperature')],
                max_T_multiplier=max_T_multiplier,
                remove_positive_E=remove_positive_E,
            ):
                frames_to_remove.add(idx)
                explode_count += 1
                continue

            # Check evaporation (fast: np.ptp)
            if evap_enabled and atl_str_filters.apply_filter_evaporation(
                struct=frame,
                max_allowed_thickness=max_allowed_thickness,
            ):
                frames_to_remove.add(idx)
                evap_count += 1
                continue

            # Check layer distance (moderate: .copy() + .wrap())
            if layer_dist_enabled and atl_str_filters.apply_filter_layer_distance(
                struct=frame, max_layer_distance_ang=max_dist
            ):
                frames_to_remove.add(idx)
                layer_dist_count += 1
                continue

        atl_cut.custom_print(
            f'Marked by filters - exploding: {explode_count}, '
            f'layer distance: {layer_dist_count}, no neighbor: {evap_count}',
            'info',
            logger=logger,
        )

        # Create a new list excluding the frames to remove
        frames_to_keep = [frame_idx for frame_idx in range(len(md_traj))]
        frames_to_keep = [
            frame for frame in frames_to_keep if frame not in frames_to_remove
        ]
        # md_traj_filtered = [md_traj[frame] for frame in frames_to_keep]

        atl_cut.custom_print(
            f'Trajectory length after MD filters: {len(frames_to_keep)}',
            'info',
            logger=logger,
        )

        if len(frames_to_keep) == 0:
            atl_cut.custom_print(
                (
                    'No MD frames left after filtering. '
                    'This means that probably there are a lot of unrealistic structures'
                    '. Check the training data and models used to run this MD.'
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
        if not safe_md_params.get('sample_frames_during_md'):
            md_traj_short, short_mask = limit_md_frames(
                md_traj=md_traj,
                frames_to_keep=frames_to_keep,
                md_params=safe_md_params,
            )
            atl_cut.custom_print(
                f'Limited number of frames to: {len(md_traj_short)}',
                'info',
                logger=logger,
            )
        else:
            # md_traj_short = md_traj_filtered
            md_traj_short = [md_traj[frame] for frame in frames_to_keep]
            short_mask = np.arange(len(md_traj_short))
            atl_cut.custom_print(
                'Trajectory already shortened during MD by save interval. '
                f'Length unchanged. Current length: {len(md_traj_short)}',
                'info',
                logger=logger,
            )

        # Free the full and filtered trajectories to release memory
        del md_traj
        gc.collect()

        # # Running evaluation of the energies and forces using each commitee model
        # atl_cut.custom_print('Running committee evaluation...', 'info', logger=logger)
        # model_file_list = list(prepend_path.glob('*.model'))
        # comm_settings = settings.get('committee_eval', {})
        # comm_results = {}
        # for model in model_file_list:
        # comm_results[model.stem] = {'REF_energy': [], 'REF_forces': []}

        from atlas.active_learning.backends import get_backend

        backend_name = settings.get('mlip', {}).get('training_backend', 'mace')
        backend = get_backend(backend_name)
        model_ext = backend.model_file_extension

        device_str = safe_md_params.get('device', 'cpu')

        calculator = backend.create_calculator(
            model_path=pl.Path(prepend_path) / f'sampler_model{model_ext}',
            device=device_str,
            dtype=safe_md_params.get('default_dtype', 'float32'),
        )

        atl_cut.custom_print(
            'Checking extrapolating frames...',
            'info',
            logger=logger,
        )

        ## Get the descriptors if extrapolation enabled
        if extrap_type != 'none':
            # Only MACE is supported for now
            atl_cut.custom_print('Generating descriptors...', 'info', logger=logger)
            if settings['descriptors'].get('descriptor_type', 'mace') == 'soap':
                descriptor_dict, descriptor_arr, uuids = (
                    atl_al_ut.generate_descriptors_soap(
                        database=md_traj_short,
                        descriptor_settings=settings['descriptors'],
                        verbose=True,
                    )
                )
            else:
                descriptor_type = settings['descriptors'].get('descriptor_type', 'mace')
                descriptor_dict, descriptor_arr, uuids = atl_al_ut.generate_descriptors(
                    database=md_traj_short,
                    descriptor_type=descriptor_type,
                    descriptor_settings=settings['descriptors'],
                    model_path=prepend_path / f'sampler_model{model_ext}',
                    verbose=True,
                )

            # Add is_extrapolating list which contains boolean values showing
            # if the frame is extrapolating or not
            # for structure_uuid in descriptor_dict:
            #     descriptor_dict[structure_uuid]['is_extrapolating'] = np.zeros(
            #         len(md_traj_short), dtype=bool
            #     )
            # Add is_extrapolating boolean showing if the frame is extrapolating
            for structure_uuid in descriptor_dict:
                descriptor_dict[structure_uuid]['is_extrapolating'] = False

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

            hull_tolerance = settings.get('extrapolation', {}).get(
                'hull_tolerance', 0.0
            )
            (
                point_inside,
                point_outside,
                all_points_in_out,
                scaled_hull,
            ) = check_traj_in_domain(
                concave_hull=concave_hull,
                descriptor_dict=descriptor_dict,
                hull_scale_factor=hull_tolerance,
            )

            plot_concave_hull(
                concave_hull=concave_hull,
                point_inside=point_inside,
                point_outside=point_outside,
                scaled_hull=scaled_hull,
                filename=res_folder / f'concave_hull_temp-{curr_temp}.png',
            )

            # True means it is INSIDE the domain (NOT extrapolating)
            # False means it is OUTSIDE the domain (IS extrapolating)
            for idx, c_uuid in enumerate(uuids):
                is_inside_hull = all_points_in_out[idx]

                if is_inside_hull:
                    descriptor_dict[c_uuid]['is_extrapolating'] = False
                else:
                    descriptor_dict[c_uuid]['is_extrapolating'] = True

        # Simple extrapolation check (descriptor min/max)
        elif extrap_type in ['basic', 'min-max']:
            # Read the minimum and maximum values for each descriptor
            # for the entire database
            try:
                curr_it_db_max = np.load(prepend_path / 'curr_it_db_max.npy')
            except ValueError:
                curr_it_db_max = np.fromfile(
                    '/atl_data/curr_it_db_min.npy', dtype=np.float32
                )

            try:
                curr_it_db_min = np.load(prepend_path / 'curr_it_db_min.npy')
            except ValueError:
                curr_it_db_min = np.fromfile(
                    '/atl_data/curr_it_db_min.npy', dtype=np.float32
                )

            descriptor_dict = simple_extrapolation_check(
                curr_it_db_max, curr_it_db_min, descriptor_dict
            )

        # Dont apply any further extrapolation check. Use the EF commitee check
        # already applied.
        elif extrap_type in ['none', 'disabled', None]:
            atl_cut.custom_print(
                ('No extrapolation check applied.'),
                'warn',
                logger=logger,
            )

        if extrap_type != 'none':
            extrapol_frames_final = [
                frame
                for frame, uuid in zip(md_traj_short, uuids, strict=False)
                if descriptor_dict[uuid]['is_extrapolating']
            ]
        else:
            extrapol_frames_final = []

        atl_cut.custom_print(
            f'Total count of extrapolating frames: {len(extrapol_frames_final)}',
            'info',
            logger=logger,
        )

        # Renaming result keys
        mod_extrap_frames = []
        for structure in extrapol_frames_final:
            # Setting the main model calculator for the current structure
            structure.calc = calculator

            # Getting the energy and forces for the structure
            structure.info['REF_energy'] = structure.get_potential_energy()
            structure.arrays['REF_forces'] = structure.get_forces()
            structure.info['atl_id'] = str(uuid.uuid4())

            mod_extrap_frames.append(structure)

        ase_write(
            res_folder / 'extrapolating_frames.xyz',
            format='extxyz',
            images=mod_extrap_frames,
            append=True,
        )

    atl_cut.custom_print('Structure processed!', 'done', logger=logger)
