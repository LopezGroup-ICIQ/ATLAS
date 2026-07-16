#!/usr/bin/env python3
"""Script to get the MACE descriptors from a structure database."""

import json
import pathlib as pl
import pickle

import numpy as np
import torch
from aiida.common.extendeddicts import AttributeDict
from ase.io import read as ase_read

from atlas.active_learning.extrapolation import train_autoencoder as atl_train_ae


def generate_descriptors(model_path: str, database):
    from atlas.active_learning.backends import get_backend

    backend = get_backend('mace')
    descriptor_dict, descriptor_arr, _ = backend.generate_descriptors(
        database=database, model_path=model_path, settings={}
    )
    return descriptor_dict, descriptor_arr


if __name__ == '__main__':
    # As this code will be run as a script,
    # we can keep these variables as constant.
    DATABASE_PATH = 'current_db.xyz'
    MACE_PATH = 'current_model_mace.model'
    DESCR_PATH = 'all_descriptors.npz'
    AUTO_PATH = 'autoencoder_model.pth'
    AUTO_SETTINGS_PATH = 'settings_dict.json'

    # Read database using ase into an Atoms object
    database = ase_read(DATABASE_PATH, format='extxyz', index=':')

    # Generate descriptors using MACE
    print('Generating descriptors using MACE...')
    descriptor_dict, descriptor_arr = generate_descriptors(MACE_PATH, database)

    # Saving descriptor array
    np.save(DESCR_PATH, descriptor_arr)

    print(f"Descriptors generated, saved as '{DESCR_PATH}'.")

    # Minimum and maximum values for each of the descriptors from MACE
    min_val = np.min(descriptor_arr, axis=0)
    max_val = np.max(descriptor_arr, axis=0)

    # Storing arrays into a numpy file to be later gathered by the workchain
    np.save(file='curr_it_db_max', arr=max_val)
    np.save(file='curr_it_db_min', arr=min_val)

    if not pl.Path(AUTO_PATH).exists():
        print('Training the autoencoder model...')

        # Read the training arguments from a toml file
        with open(AUTO_SETTINGS_PATH) as f:
            settings_dict = AttributeDict(json.load(f))

        # Train the autoencoder model
        atl_train_ae.run_training(settings_dict)
        print('Autoencoder model trained.')
    else:
        print('Loading Autoencoder model...')

    # Load autoencoder model
    model = torch.load(AUTO_PATH, weights_only=False)

    # Changing device to CPU
    model.to('cpu')

    # Remember that you must call model.eval() to set dropout and batch
    # normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Reduce the dimensionality of the input points to 2D
    print('Computing latent space for all structures...')
    with torch.no_grad():  # No need to compute gradients for inference
        for struct in database:
            # Get descriptors
            curr_struct_id = struct.info['aiida_uuid']
            curr_descriptors = descriptor_dict[curr_struct_id]['descriptors']

            # Get latent space
            latent_space = model.encoder(torch.Tensor(descriptor_arr))

            # Save latent space
            descriptor_dict[curr_struct_id]['latent_space'].append(
                latent_space.cpu().numpy()
            )

    print('Computed latent space!')

    # This pickle object will contain a dict of length n_struct,
    # that will have two keys inside inside, `descriptors` and `latent_space`.
    # The `descriptors` will contain model_size lists of descriptor values.
    # The `latent_space` will contain the 2D latent space representation
    # of the descriptors.
    with open('curr_it_db_descriptors.pkl', 'wb') as f:
        pickle.dump(descriptor_dict, f)

    # Saving latent space
    np.save('latent_space.npy', latent_space)

    print("Latent space saved in 'curr_it_db_descriptors.pkl' and 'latent_space.npy'.")
    print('Calculation finished correctly!')
