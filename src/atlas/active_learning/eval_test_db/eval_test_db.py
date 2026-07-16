#!/usr/bin/env python3
"""
Module for evaluating a test database using the sampler model from
a ATLAS active learning workflow.
"""

import json
import os
import pathlib as pl
import tomllib

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from matplotlib.ticker import MultipleLocator

from atlas.core import code_utils as atl_cut

if __name__ == '__main__':
    # The /atl_data directory should only exist in the containerized variant
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
        source='test_db_eval', log_path=log_folder
    )
    atl_cut.custom_print(
        'Loading settings for database evaluation...', 'info', logger=logger
    )

    # Initialize random seed
    rng_seed = np.random.randint(0, ((2**32) - 1))
    np.random.seed(rng_seed)
    atl_cut.custom_print(f"Using random seed: '{rng_seed}'", logger=logger)

    # Load settings
    with open(prepend_path / 'settings.toml', 'rb') as f:
        settings = tomllib.load(f)
    eval_test_db_settings = settings.get('test_db', {})
    model_settings = eval_test_db_settings.get('model_settings', {})

    # Load test database
    test_db = ase_read(prepend_path / 'test_db.xyz', index=':', format='extxyz')
    atl_cut.custom_print(
        f'Loaded test database with {len(test_db)} structures.', 'info', logger=logger
    )

    # Get precomputed energies and forces from structures
    true_energies = np.array([struc.info['REF_energy'] for struc in test_db])
    true_forces = [struc.arrays['REF_forces'] for struc in test_db]

    # Initialize calculator using the MLIP backend
    from atlas.active_learning.backends import find_inference_model, get_backend

    backend_name = settings.get('mlip', {}).get('training_backend', 'mace')
    backend = get_backend(backend_name)

    # Prefer a pre-compiled inference artifact if the compile-once step shipped
    # one; otherwise fall back to the raw model (unchanged for MACE).
    mace_calc = backend.create_calculator(
        model_path=find_inference_model(prepend_path, 'curr_iter_best', backend),
        device=model_settings.get('device', 'cpu'),
        dtype=model_settings.get('default_dtype', 'float32'),
    )

    # Evaluate test database
    atl_cut.custom_print('Starting test database evaluation...', 'info', logger=logger)
    pred_energies = []
    pred_forces = []
    num_atoms_list = []
    for structure in test_db:
        structure.calc = mace_calc
        pred_energies.append(structure.get_potential_energy())
        pred_forces.append(structure.get_forces())
        num_atoms_list.append(len(structure))

    num_at_array = np.array(num_atoms_list)
    pred_energies = np.array(pred_energies)

    # Energy metrics (meV/atom)
    # Calculate error per atom first to ensure units match meV/atom
    energy_diff_per_atom = ((pred_energies - true_energies) / num_at_array) * 1000.0

    energy_mae = np.mean(np.abs(energy_diff_per_atom))
    energy_rmse = np.sqrt(np.mean(energy_diff_per_atom**2))

    # Force metrics (meV/Å)
    # Concatenate lists of ragged arrays into single flat arrays (N_total_atoms, 3)
    flat_pred_forces = np.concatenate(pred_forces)
    flat_true_forces = np.concatenate(true_forces)

    forces_mae = np.mean(np.abs((flat_pred_forces - flat_true_forces) * 1000.0))
    forces_rmse = np.sqrt(
        np.mean(((flat_pred_forces - flat_true_forces) * 1000.0) ** 2)
    )

    atl_cut.custom_print(
        f'Test DB Energy MAE: {energy_mae:.6f} meV/atom', 'info', logger=logger
    )
    atl_cut.custom_print(
        f'Test DB Forces MAE: {forces_mae:.6f} meV/Å', 'info', logger=logger
    )
    atl_cut.custom_print(
        f'Test DB Energy RMSE: {energy_rmse:.6f} meV/atom', 'info', logger=logger
    )
    atl_cut.custom_print(
        f'Test DB Forces RMSE: {forces_rmse:.6f} meV/Å', 'info', logger=logger
    )

    results_file = prepend_path / 'test_db_eval_results.json'
    file_size = os.path.getsize(results_file)

    if file_size == 0:
        # If the file is empty, initialize an empty results dictionary
        results_dict = {'current_iteration': 0}
    else:
        # Load results json file
        with open(results_file) as f:
            results_dict = json.load(f)

    # Save results to json file
    current_iter = results_dict['current_iteration']
    results_dict[f'step_{current_iter}'] = {}

    results_dict[f'step_{current_iter}']['mae_e'] = energy_mae
    results_dict[f'step_{current_iter}']['mae_f'] = forces_mae
    results_dict[f'step_{current_iter}']['rmse_e'] = energy_rmse
    results_dict[f'step_{current_iter}']['rmse_f'] = forces_rmse
    results_dict[f'step_{current_iter}']['pred_energies'] = pred_energies.tolist()
    results_dict[f'step_{current_iter}']['pred_forces'] = [
        force_arr.tolist() for force_arr in pred_forces
    ]

    # Plot results
    # Create 4 figures arranged in 2 vertical columns for the errors
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot RMSE_E
    iters = []
    rmse_e_values = []
    for key in results_dict:
        if key.startswith('step_'):
            iters.append(int(key.split('_')[1]))
            rmse_e_values.append(results_dict[key]['rmse_e'])
    axs[0, 0].plot(iters, rmse_e_values, marker='o', color='#458588')
    axs[0, 0].set_title('Test DB Energy RMSE over Iterations')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Energy RMSE (meV/atom)')
    axs[0, 0].grid(True)
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(1))

    # Plot RMSE_F
    rmse_f_values = []
    for key in results_dict:
        if key.startswith('step_'):
            rmse_f_values.append(results_dict[key]['rmse_f'])
    axs[0, 1].plot(iters, rmse_f_values, marker='o', color='#cc241d')
    axs[0, 1].set_title('Test DB Forces RMSE over Iterations')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Forces RMSE (meV/Å)')
    axs[0, 1].grid(True)
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(1))

    # Plot MAE_E
    mae_e_values = []
    for key in results_dict:
        if key.startswith('step_'):
            mae_e_values.append(results_dict[key]['mae_e'])
    axs[1, 0].plot(iters, mae_e_values, marker='o', color='#98971a')
    axs[1, 0].set_title('Test DB Energy MAE over Iterations')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Energy MAE (meV/atom)')
    axs[1, 0].grid(True)
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(1))

    # Plot MAE_F
    mae_f_values = []
    for key in results_dict:
        if key.startswith('step_'):
            mae_f_values.append(results_dict[key]['mae_f'])
    axs[1, 1].plot(iters, mae_f_values, marker='o', color='#d65d0e')
    axs[1, 1].set_title('Test DB Forces MAE over Iterations')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Forces MAE (meV/Å)')
    axs[1, 1].grid(True)
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(1))

    # Save figure to file
    plt.tight_layout()
    plt.savefig(prepend_path / 'test_db_eval_plots.png', dpi=300)
    plt.savefig(prepend_path / 'test_db_eval_plots.svg')

    # Write updated results to json file
    with open(prepend_path / 'test_db_eval_results_updated.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    atl_cut.custom_print(
        'Test database evaluation completed.',
        'done',
        logger=logger,
    )
