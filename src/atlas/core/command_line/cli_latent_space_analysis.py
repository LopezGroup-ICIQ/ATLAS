"""Launch latent space analysis from a TOML configuration file."""

import argparse
import pathlib as pl
import pickle
import time
import tomllib

import numpy as np
import torch
from aiida.common.extendeddicts import AttributeDict
from ase.io import read

from atlas.active_learning import active_learning_utils as atl_al_ut
from atlas.active_learning.extrapolation import morphological_closing as atl_morph
from atlas.active_learning.extrapolation import train_autoencoder as atl_train_ae
from atlas.active_learning.extrapolation.concave_hull import (
    check_traj_in_domain,
    get_optimized_concave_hull,
    plot_concave_hull,
)
from atlas.core import code_utils as atl_cut


def run_latent_space_analysis():
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='atl_latent_space_analysis',
        description='Analyze latent space of a database using '
        'settings from a TOML file.',
    )
    parser.add_argument(
        'config',
        type=pl.Path,
        nargs='?',
        default=pl.Path('latent_space_settings.toml'),
        help='Path to the TOML configuration file.',
    )
    parser.add_argument(
        '--database',
        type=pl.Path,
        default=None,
        help='Path to the database file (e.g. .xyz, .traj).',
    )
    parser.add_argument(
        '--model-dir',
        type=pl.Path,
        default=pl.Path.cwd(),
        help='Directory containing models and concave hull files. '
        'Defaults to current directory.',
    )
    parser.add_argument(
        '--output',
        type=pl.Path,
        default=pl.Path.cwd(),
        help='Output folder for results.',
    )

    args = parser.parse_args()

    # Setup paths and logger
    config_path = args.config
    prepend_path = args.model_dir
    res_folder = args.output

    if not res_folder.exists():
        res_folder.mkdir(parents=True)

    logger, _ = atl_cut.init_logger(source='latent_space_analysis', log_path=res_folder)

    # Load settings
    with open(config_path, 'rb') as f:
        settings = tomllib.load(f)

    db_path = args.database or pl.Path(settings['descriptors']['database_path'])

    # Load database
    atl_cut.custom_print(f'Loading database from {db_path}...', 'info', logger=logger)
    md_traj_short = read(db_path, index=':')
    atl_cut.custom_print(
        f'Loaded {len(md_traj_short)} structures.', 'info', logger=logger
    )

    # Extract settings
    extrap_settings = settings.get('extrapolation', {})
    concave_hull_settings = extrap_settings.get('concave_hull', {})
    concave_hull_scale_factor = concave_hull_settings.get(
        'concave_hull_scale_factor', 0.0
    )

    # Boundary determination algorithm and morphological closing settings
    boundary_method = concave_hull_settings.get('boundary_method', 'concave_hull')
    morph_disk_size = concave_hull_settings.get('morph_disk_size', 10)
    morph_threshold = concave_hull_settings.get('morph_threshold', 250)
    morph_dpi = concave_hull_settings.get('morph_dpi', 100)

    extrap_type = extrap_settings.get('check_extrapolation_type', 'none')

    # Autoencoder specific settings
    dim_red_method = settings.get('descriptors', {}).get(
        'dimensionality_reduction_method', 'none'
    )

    # Get latent space data
    if extrap_type != 'none':
        # Only MACE is supported for now

        descriptor_settings = settings.get('descriptors', {})

        if not (res_folder / 'curr_it_db_descriptors.pkl').exists():
            if (prepend_path / 'all_descriptors.npz').exists():
                atl_cut.custom_print(
                    'Reading descriptors from npz file...', 'info', logger=logger
                )
                descriptor_arr = np.load(prepend_path / 'all_descriptors.npz')
                if descriptor_arr.get('arr_0') is not None:
                    descriptor_arr = descriptor_arr['arr_0']
                elif descriptor_arr.get('descriptor') is not None:
                    descriptor_arr = descriptor_arr['descriptor']

                descriptor_dict = {}
                for idx, struct in enumerate(md_traj_short):
                    if struct.info.get('atl_id'):
                        curr_struct_id = struct.info.get('atl_id')
                    else:
                        curr_struct_id = struct.info.get('aiida_uuid')

                    descriptor_dict[curr_struct_id] = {
                        'descriptors': [descriptor_arr[idx]]
                    }
            else:
                atl_cut.custom_print(
                    'Computing descriptors from database...', 'info', logger=logger
                )
                descriptor_type = descriptor_settings.get('descriptor_type', 'mace')

                if descriptor_type == 'soap':
                    atl_cut.custom_print(
                        'Generating SOAP descriptors...', 'info', logger=logger
                    )
                    atl_cut.custom_print(
                        f'Using following settings: '
                        f'{descriptor_settings.get("soap", {})}'
                    )
                    descriptor_dict, descriptor_arr, uuid_list = (
                        atl_al_ut.generate_descriptors_soap(
                            database=md_traj_short,
                            descriptor_settings=descriptor_settings.get('soap', {}),
                        )
                    )
                else:
                    descriptor_dict = descriptor_settings.get('mace', {})

                    # Check for model path
                    model_path = prepend_path / descriptor_dict.get(
                        'model_path', 'curr_model.model'
                    )
                    if not model_path.exists():
                        atl_cut.custom_print(
                            f'MACE model not found at {model_path}.',
                            'warn',
                            logger=logger,
                        )

                    descriptor_dict, descriptor_arr, uuids = (
                        atl_al_ut.generate_descriptors_mace(
                            model_path=str(model_path),
                            database=md_traj_short,
                            descriptor_settings=descriptor_settings,
                        )
                    )

                # Minimum and maximum values for each of the descriptors
                min_val = np.min(descriptor_arr, axis=0)
                max_val = np.max(descriptor_arr, axis=0)

                # Storing arrays into a numpy file to be later gathered by the workchain
                np.save(file=res_folder / 'curr_it_db_max', arr=max_val)
                np.save(file=res_folder / 'curr_it_db_min', arr=min_val)

                # Saving descriptor array
                np.savez_compressed(
                    prepend_path / 'all_descriptors.npz', descriptor_arr
                )
                atl_cut.custom_print(
                    'Descriptors generated and saved.', 'info', logger=logger
                )

        else:
            atl_cut.custom_print(
                'Reading descriptors from file...', 'info', logger=logger
            )
            try:
                descriptor_arr = np.load(prepend_path / 'all_descriptors.npz')
                if descriptor_arr.get('arr_0') is not None:
                    descriptor_arr = descriptor_arr['arr_0']
                elif descriptor_arr.get('descriptor') is not None:
                    descriptor_arr = descriptor_arr['descriptor']

            except FileNotFoundError:
                descriptor_arr = np.load(prepend_path / 'all_descriptors.npy')

            with open(res_folder / 'curr_it_db_descriptors.pkl', 'rb') as f:
                descriptor_dict = pickle.load(f)

            atl_cut.custom_print('Descriptors loaded from file.', 'info', logger=logger)

        atl_cut.custom_print(
            (
                'Gathered a descriptor array with the following shape: '
                f'{descriptor_arr.shape}'
            ),
            'done',
            logger=logger,
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

        # Get latent space for the trajectory
        match dim_red_method:
            case 'pca':
                # latent_space = get_latent_space_pca(database=descriptor_dict)
                raise NotImplementedError('PCA not implemented yet.')
            case 'autoencoder':
                aut_t_params = (
                    settings['descriptors']
                    .get('autoencoder', {})
                    .get('train_settings', {})
                )

                # Determine model path
                model_filename = aut_t_params.get('model_path', 'autoencoder_model.pth')
                model_path = prepend_path / model_filename

                if not model_path.exists():
                    atl_cut.custom_print(
                        f'Autoencoder model not found at {model_path}. '
                        'Training new model...',
                        'info',
                        logger=logger,
                    )
                    # Update params for training
                    aut_t_params['model_path'] = model_path
                    # We use the descriptors calculated above as dataset
                    # aut_t_params['dataset'] =

                    # Train the autoencoder model
                    atl_train_ae.run_training(AttributeDict(aut_t_params))
                    atl_cut.custom_print(
                        'Autoencoder model trained.', 'info', logger=logger
                    )

                if model_path.exists():
                    model = torch.load(
                        model_path,
                        weights_only=False,
                    )
                    atl_cut.custom_print(
                        f"Model loaded from:'{model_path}'", logger=logger
                    )
                else:
                    atl_cut.custom_print(
                        f'Autoencoder model not found at {model_path}',
                        'error',
                        logger=logger,
                    )
                    return

                print()
                model.to(dtype=torch.float32)
                model.eval()

                # Reduce the dimensionality of the input points to 2D
                atl_cut.custom_print(
                    'Computing latent space for all structures...', logger=logger
                )
                latent_space_all = []

                autoenc_dev = next(model.parameters()).device
                atl_cut.custom_print(
                    f"Autoencoder model is currently on: '{autoenc_dev}'"
                )

                with torch.no_grad():  # No need to compute gradients for inference
                    for idx, struct in enumerate(md_traj_short):
                        atl_cut.custom_print(
                            f'Structure {idx}/{len(md_traj_short)}',
                            'debug',
                            logger=logger,
                        )

                        # Get descriptors
                        if struct.info.get('atl_id'):
                            curr_struct_id = struct.info.get('atl_id')
                        else:
                            curr_struct_id = struct.info.get('aiida_uuid')

                        if (
                            'latent_space' in descriptor_dict[curr_struct_id]
                            and len(descriptor_dict[curr_struct_id]['latent_space']) > 0
                        ):
                            latent_space = descriptor_dict[curr_struct_id][
                                'latent_space'
                            ]
                        else:
                            curr_descriptors = descriptor_dict[curr_struct_id][
                                'descriptors'
                            ][0]

                            # Get latent space
                            latent_space = (
                                model.encoder(
                                    torch.Tensor(curr_descriptors).to(autoenc_dev)
                                )
                                .cpu()
                                .numpy()
                            )

                            descriptor_dict[curr_struct_id]['latent_space'] = (
                                latent_space
                            )

                        latent_space_all.append(latent_space)

                # Saving latent space
                latent_space_all = np.vstack(latent_space_all)
                np.save(res_folder / 'latent_space.npy', latent_space_all)
            case 'none' | _:
                atl_cut.custom_print(
                    'Dimensionality method not specified or not recognized!',
                    'warn',
                    logger=logger,
                )

        # Save descriptor dictionary with latent space
        with open(res_folder / 'curr_it_db_descriptors.pkl', 'wb') as f:
            pickle.dump(descriptor_dict, f)

        # Read the concave hull. If it fails, calculate it.
        concave_hull = None
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
                'Concave hull file not found! Calculating it from descriptors...',
                'info',
                logger=logger,
            )
            # Create array from dict
            # Check if latent space is available (from autoencoder step)
            try:
                latent_space = np.vstack(
                    [
                        descriptor_dict[structure_uuid]['latent_space']
                        for structure_uuid in descriptor_dict
                    ]
                )
            except KeyError:
                # Fallback to descriptors if latent space not present
                # (e.g. no reduction)
                latent_space = np.array(
                    [
                        descriptor_dict[structure_uuid]['descriptors']
                        for structure_uuid in descriptor_dict
                    ]
                )

            # Flatten the array if it has shape (N, 1, D) to (N, D)
            if len(latent_space.shape) == 3 and latent_space.shape[1] == 1:
                latent_space = latent_space.reshape(latent_space.shape[0], -1)

            if boundary_method == 'morphological_closing':
                atl_cut.custom_print(
                    'Using morphological closing for boundary determination.',
                    'info',
                    logger=logger,
                )
                morph_results = atl_morph.process_morphological_closing(
                    latent_space[:, 0],
                    latent_space[:, 1],
                    disk_size=morph_disk_size,
                    threshold=morph_threshold,
                    dpi=morph_dpi,
                )
                # data_boundaries is already a list of (N, 2) coordinate arrays,
                # the format check_traj_in_domain / plot_concave_hull consume.
                concave_hull = morph_results['data_boundaries']
            else:
                concave_hull, _ = get_optimized_concave_hull(latent_space=latent_space)

        # Add a timestamp
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        plot_filename = res_folder / f'latent_space_analysis_{timestamp}.png'

        (
            point_inside,
            point_outside,
            all_points_in_out,
            scaled_hull,
        ) = check_traj_in_domain(
            concave_hull=concave_hull,
            descriptor_dict=descriptor_dict,
            hull_scale_factor=concave_hull_scale_factor,
        )

        plot_concave_hull(
            concave_hull=concave_hull,
            point_inside=point_inside,
            point_outside=point_outside,
            scaled_hull=None,
            filename=plot_filename,
        )

        atl_cut.custom_print(
            f'Analysis complete. Plot saved to {plot_filename}',
            'info',
            logger=logger,
        )
