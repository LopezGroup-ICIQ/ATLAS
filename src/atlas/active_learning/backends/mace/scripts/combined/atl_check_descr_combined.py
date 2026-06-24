#!/usr/bin/env python3
"""Script to handle descriptor gathering for an AL Loop iteration."""

import logging
import pathlib as pl
import pickle
import tomllib
import uuid
import warnings

import numpy as np
from aiida.common.extendeddicts import AttributeDict
from ase.io import read as ase_read
from shapely.geometry import Point, Polygon

from atlas.active_learning.active_learning_utils import generate_descriptors
from atlas.active_learning.extrapolation import quadtree as atl_qt
from atlas.active_learning.extrapolation import train_autoencoder as atl_train_ae
from atlas.active_learning.extrapolation.concave_hull import (
    get_optimized_concave_hull,
)
from atlas.core import code_utils as atl_cut

# Silencing specific warnings and log messages
warnings.filterwarnings('ignore', category=UserWarning, message='.*weights_only.*')

# Force third party loggers to only show errors and critical messages
logging.getLogger('mace').setLevel(logging.ERROR)
logging.getLogger('e3nn').setLevel(logging.ERROR)


def is_advanced_extrapolation_set(input_dict: dict):
    """Check if the advanced extrapolation check is enabled."""
    extr_type = input_dict.get('extrapolation', {}).get('check_extrapolation_type')
    if extr_type in ['advanced', 'alpha-shape']:
        return True
    return


def has_latent_space(settings_dict: dict):
    """Check if the latent space was computed for the current iteration."""
    return settings_dict.get('latent_space', False)


def can_do_advanced_extrapolation():
    """Check if the advanced extrapolation can be done."""
    return has_latent_space() and is_advanced_extrapolation_set()


def check_atom_in_domain(
    concave_hull: np.ndarray, descriptors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for point in descriptors:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
            all_points_in_out.append(True)
        else:
            point_outside.append(point)
            all_points_in_out.append(False)

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out


if __name__ == '__main__':
    # The /atl_data directory should only exist in the containerized version
    # of the code. This conditional statement will get the correct path for
    # input and output files.
    if pl.Path('/atl_data').exists():
        prepend_path = pl.Path('/atl_data')
    else:
        prepend_path = pl.Path('.')

    # Define results folder
    res_folder = pl.Path(prepend_path) / pl.Path('./results')
    res_folder.mkdir(exist_ok=True)

    # Initialize the logger
    log_folder = prepend_path / pl.Path('./logs')
    log_folder.mkdir(exist_ok=True)

    logger, log_filename = atl_cut.init_logger(
        source='proc_structure', log_path=log_folder
    )

    atl_cut.custom_print(
        'Initializing descriptor generation, dimensionality reduction, '
        'and concave hull computation for the structure database.'
    )
    atl_cut.custom_print(f"Storing results in '{res_folder.resolve()}'.")

    # Read TOML file with settings
    with open(prepend_path / 'settings.toml', 'rb') as f:
        settings = tomllib.load(f)

    descriptor_settings = settings.get('descriptors', {})
    device = descriptor_settings.get('device', 'cpu')
    dtype = descriptor_settings.get('dtype', 'float32')
    extrapolation_settings = settings.get('extrapolation', {})
    auto_settings = descriptor_settings.get('autoencoder', {})
    auto_train_settings = auto_settings.get('train_settings', {})
    auto_path = auto_train_settings.get('model_path', 'autoencoder_model.pth')

    atl_cut.custom_print(f"Using device '{device}' and dtype '{dtype}'", 'info')

    # Get settings for concave hull if available
    concave_hull_settings = extrapolation_settings.get('concave_hull', {})
    target_alpha_range_min = concave_hull_settings.get('target_alpha_range_min', 3.0)
    target_alpha_range_max = concave_hull_settings.get('target_alpha_range_max', 8.0)
    default_alpha_if_issues = concave_hull_settings.get('default_alpha_if_issues', 5.0)
    nn_dist_scale_factor = concave_hull_settings.get('nn_dist_scale_factor', 1.5)
    frac_points_allowed_out = concave_hull_settings.get(
        'frac_points_allowed_out', 0.002
    )
    qt_offset_frac = concave_hull_settings.get('qt_offset_frac', 0.1)
    qt_data_frac_capacity = concave_hull_settings.get('qt_data_frac_capacity', 0.015)
    qt_subdivision_factor = concave_hull_settings.get('qt_subdivision_factor', 8)

    # Ensure target alpha range is a tuple
    target_alpha_range = (target_alpha_range_min, target_alpha_range_max)

    # Load database structures. Assuming extxyz format.
    atl_cut.custom_print('Reading structure database...')
    structs_database = ase_read(
        prepend_path / 'training_db.xyz', index=':', format='extxyz'
    )

    # Ensure that each structure has an 'atl_id' in its info dictionary.
    # This shouldn't normally happen, except when debugging with custom
    # databases. In normal operation, each structure should already have
    # an 'atl_id' assigned during database creation.
    # There's an additional fallback in the descriptor generation function,
    # which generates a UUID if 'atl_id' is not found. However, using it
    # would require saving a file mapping the generated UUIDs to the structures,
    # and add code to read that file later.
    for struct in structs_database:
        if 'atl_id' not in struct.info:
            struct.info['atl_id'] = str(uuid.uuid4())

    # Generate descriptors
    if not pl.Path(res_folder / 'curr_it_db_descriptors.pkl').exists():
        atl_cut.custom_print('Computing descriptors from database...')
        descriptor_type = descriptor_settings.get('descriptor_type', 'mace')
        descriptor_dict, descriptor_arr, _ = generate_descriptors(
            descriptor_type=descriptor_type,
            database=structs_database,
            descriptor_settings=descriptor_settings,
            model_path=prepend_path / 'curr_iter_best.model',
        )

    else:
        atl_cut.custom_print('Reading descriptors from file...')
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

    atl_cut.custom_print(
        f'Gathered a descriptor array with the following shape: {descriptor_arr.shape}',
        'done',
    )

    # Minimum and maximum values for each of the descriptors
    min_val = np.min(descriptor_arr, axis=0)
    max_val = np.max(descriptor_arr, axis=0)

    # Storing arrays into a numpy file to be later gathered by the workchain
    np.save(file=res_folder / 'curr_it_db_max', arr=max_val)
    np.save(file=res_folder / 'curr_it_db_min', arr=min_val)

    # Saving descriptor array
    np.savez_compressed(prepend_path / 'all_descriptors.npz', descriptor_arr)
    atl_cut.custom_print('Descriptors generated.')

    latent_space = None
    latent_space_all = None

    # Proceed with advanced extrapolation if enabled
    if is_advanced_extrapolation_set(input_dict=settings):
        # Get latent space
        latent_space_file = res_folder / 'latent_space.npy'

        if not pl.Path(prepend_path / auto_path).exists():
            atl_cut.custom_print('Training the autoencoder model...')

            # Check if the dataset path exists, if not prepend the prepend_path
            if not pl.Path(auto_train_settings['dataset']).exists():
                auto_train_settings['dataset'] = (
                    prepend_path / auto_train_settings['dataset']
                )
            # Check if the model path exists, if not prepend the prepend_path
            if not pl.Path(auto_train_settings['model_path']).exists():
                auto_path = prepend_path / auto_train_settings['model_path']
                auto_train_settings['model_path'] = auto_path

            # Train the autoencoder model
            atl_train_ae.run_training(AttributeDict(auto_train_settings))
            atl_cut.custom_print('Autoencoder model trained.')
        else:
            atl_cut.custom_print('Loading Autoencoder model...')

        # Load autoencoder model
        import torch

        model = torch.load(auto_path, weights_only=False)

        # Save autoencoder model to `autoencoder_model.pth` if it does not exist
        if not pl.Path(res_folder / 'autoencoder_model.pth').exists():
            torch.save(model, res_folder / 'autoencoder_model.pth')

        # Changing device to CPU
        model.to('cpu')

        # Remember that you must call model.eval() to set dropout and batch
        # normalization layers to evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        model.eval()

        # Reduce the dimensionality of the input points to 2D
        atl_cut.custom_print('Computing latent space for all structures...')
        latent_space_all = []
        with torch.no_grad():  # No need to compute gradients for inference
            for idx, struct in enumerate(structs_database):
                atl_cut.custom_print(
                    f'Structure {idx}/{len(structs_database)}', 'debug'
                )

                # Get descriptors
                if struct.info.get('atl_id'):
                    curr_struct_id = struct.info.get('atl_id')
                else:
                    curr_struct_id = struct.info.get('aiida_uuid')

                curr_descriptors = descriptor_dict[curr_struct_id]['descriptors'][0]

                # Get latent space
                latent_space = (
                    model.encoder(torch.Tensor(curr_descriptors)).cpu().numpy()
                )

                # Save latent space
                descriptor_dict[curr_struct_id]['latent_space'].extend(latent_space)
                latent_space_all.append(latent_space)

        # Saving latent space
        latent_space_all = np.vstack(latent_space_all)
        np.save(latent_space_file, latent_space_all)

        # No way to store all of the descriptors in a single array,
        # as the n_atom dimension will change according to the structure
        # This pickle object will contain a list of length n_struct,
        # that will have n_at lists inside, each containing model_size
        # lists of descriptor values.
        with open(res_folder / 'curr_it_db_descriptors.pkl', 'wb') as f:
            pickle.dump(descriptor_dict, f)

    # If latent space was successfully calculated
    # Check for the existence of the latent_space variable
    if latent_space is not None and latent_space_all is not None:
        # Get concave hull
        match descriptor_settings.get('dimensionality_reduction_method'):
            case 'autoencoder':
                atl_cut.custom_print(
                    f'Read concave hull settings. '
                    f'Target alpha range: {target_alpha_range}, '
                    f'default alpha if issues: {default_alpha_if_issues}, '
                    f'NN distance scale factor: {nn_dist_scale_factor}, '
                    f'Fraction of points allowed outside: {frac_points_allowed_out}, '
                    f'QuadTree offset fraction: {qt_offset_frac}, '
                    f'QuadTree leaf capacity fraction: {qt_data_frac_capacity}, '
                    f'QuadTree leaf subdivision factor: {qt_subdivision_factor}, ',
                    'info',
                )

                # Preparing quadtree
                atl_cut.custom_print('Preparing QuadTree...', 'info')
                all_points = [Point(p) for p in latent_space_all]

                # Setup QuadTree
                qt = atl_qt.setup_quadtree(
                    all_points=all_points,
                    offset_frac=qt_offset_frac,
                    data_frac_capacity=qt_data_frac_capacity,
                )
                atl_cut.custom_print('QuadTree prepared.', 'done')

                # Preparing clusters
                atl_cut.custom_print('Preparing clusters...', 'info')
                dense_boxes = qt.find_dense_leaves(
                    max_width_threshold=qt.data_range_x / qt_subdivision_factor
                )
                clusters = atl_qt.separate_clusters(dense_boxes)
                atl_cut.custom_print('Clusters obtained!', 'done')

                atl_cut.custom_print(
                    f'Number of distinct clusters found: {len(clusters)}'
                )
                print()

                alpha_shapes = []
                alpha_val = None
                if len(clusters) > 1:
                    for i, cluster_dense_boxes in enumerate(clusters):
                        try:
                            atl_cut.custom_print(
                                f'Cluster {i + 1} has {len(cluster_dense_boxes)} '
                                'dense boxes.',
                                'debug',
                            )
                            atl_cut.custom_print(
                                f'Computing concave hull for Cluster {i + 1}...', 'info'
                            )

                            # Box-aware point filtering
                            latent_space_pts = []
                            for box in cluster_dense_boxes:
                                latent_space_pts.extend(
                                    [p for p in all_points if box.contains(p)]
                                )
                            coords = np.array([(p.x, p.y) for p in latent_space_pts])

                            if len(coords) < 3:
                                raise ValueError(
                                    'Not enough points after QuadTree filtering'
                                )

                            alpha_shape_arr, alpha_val = get_optimized_concave_hull(
                                latent_space=coords,
                                target_alpha_range=(1, 500.0),
                                frac_points_allowed_out=0.05,
                            )

                            # Create polygon from shell
                            alpha_shape_polygon = Polygon(alpha_shape_arr)

                            points_inside_hull = [
                                p
                                for p in latent_space_pts
                                if alpha_shape_polygon.intersects(p)
                            ]
                            frac_inside_hull = (
                                len(points_inside_hull) / len(latent_space_pts)
                                if latent_space_pts
                                else 0.0
                            )

                            # Store alpha shapes into alpha_shapes list
                            alpha_shapes.append(
                                {
                                    'cluster_id': i + 1,
                                    'dense_boxes': cluster_dense_boxes,
                                    'alpha_shape': alpha_shape_polygon,
                                    'used_alpha': alpha_val,
                                    'frac_inside_hull': frac_inside_hull,
                                }
                            )

                        except Exception as e:
                            atl_cut.custom_print(
                                f'Could not compute concave hull due to error: {e}. '
                                'Skipping concave hull computation.',
                                'error',
                            )
                            concave_hull = None
                else:
                    try:
                        atl_cut.custom_print(
                            'Only one cluster found. '
                            'Computing concave hull for the entire dataset...',
                            'info',
                        )

                        coords = np.array([(p.x, p.y) for p in all_points])

                        alpha_shape_arr, alpha_val = get_optimized_concave_hull(
                            latent_space=coords,
                            target_alpha_range=target_alpha_range,
                            frac_points_allowed_out=frac_points_allowed_out,
                        )

                        # Create polygon from shell
                        alpha_shape_polygon = Polygon(alpha_shape_arr)

                        points_inside_hull = [
                            p for p in all_points if alpha_shape_polygon.intersects(p)
                        ]
                        frac_inside_hull = len(points_inside_hull) / len(all_points)

                        # Store alpha shapes into alpha_shapes list
                        alpha_shapes.append(
                            {
                                'cluster_id': 1,
                                'dense_boxes': None,
                                'alpha_shape': alpha_shape_polygon,
                                'used_alpha': alpha_val,
                                'frac_inside_hull': frac_inside_hull,
                            }
                        )

                    except Exception as e:
                        atl_cut.custom_print(
                            f'Could not compute concave hull due to error: {e}. '
                            'Skipping concave hull computation.',
                            'error',
                        )
                        concave_hull = None

                concave_hull_data_path = res_folder / 'concave_hulls_data.pkl'
                with open(concave_hull_data_path, 'wb') as f:
                    pickle.dump(alpha_shapes, f)

                shape_coordinates = []
                for shape in alpha_shapes:
                    curr_shape_coord = shape['alpha_shape'].exterior.coords.xy
                    curr_shape_coord = [
                        coord
                        for coord in zip(
                            curr_shape_coord[0], curr_shape_coord[1], strict=True
                        )
                    ]
                    shape_coordinates.append(curr_shape_coord)

                concave_hull_path = res_folder / 'concave_hulls.pkl'
                with open(concave_hull_path, 'wb') as f:
                    pickle.dump(shape_coordinates, f)

                atl_cut.custom_print(
                    f"Concave hull(s) computed, saved to '{concave_hull_path}'.",
                    'info',
                )
                print()

                points_inside, points_outside, frac_outside = (
                    atl_qt.check_if_points_in_polygons(alpha_shapes, all_points)
                )

                atl_cut.custom_print(
                    f'Points inside alpha shapes: {len(points_inside)} '
                    f', outside: {len(points_outside)}',
                    'info',
                )
                atl_cut.custom_print(
                    f'Percentage of total datapoints points outside '
                    f'alpha shapes: {frac_outside * 100:.2f}%',
                    'info',
                )

                plot_hull = True

                if plot_hull:
                    plot_img_path = res_folder / 'concave_hull.png'
                    atl_qt.visualize_quadtree(
                        qt=qt,
                        points=all_points,
                        clusters=clusters,
                        alpha_shapes=alpha_shapes,
                        filename=plot_img_path,
                        frac_outside=frac_outside,
                    )

                    atl_cut.custom_print(
                        f"Concave hull plotted, saved to '{plot_img_path}'.", 'info'
                    )

            case 'pca':
                raise NotImplementedError('PCA not implemented yet')
            case 'none':
                atl_cut.custom_print(
                    'No dimensionality reduction method specified. '
                    'Skipping concave hull computation.',
                    'warn',
                )
                concave_hull = None
    else:
        atl_cut.custom_print(
            'Latent space was not computed. Skipping concave hull computation.',
            'warn',
        )
    atl_cut.custom_print('Calculation done.', 'done')
