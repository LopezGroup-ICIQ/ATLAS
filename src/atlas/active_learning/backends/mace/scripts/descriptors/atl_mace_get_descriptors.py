#!/usr/bin/env python3
"""Script to get the MACE descriptors from a structure database."""

import pickle

import numpy as np
from ase.io import read as ase_read

from atlas.active_learning.active_learning_utils import generate_descriptors

if __name__ == '__main__':
    print('Running MACE descriptor generation script...')

    # As this code will be run as a script,
    # we can keep these variables as constant.
    DATABASE_PATH = 'current_db.xyz'
    MODEL_PATH = 'current_model_mace.model'

    # Read database using ase into an Atoms object
    database = ase_read(DATABASE_PATH, format='extxyz', index=':')

    # Generate descriptors using MACE
    descriptor_dict, descriptor_arr, uuid_list = generate_descriptors(
        MODEL_PATH, database
    )

    # Minimum and maximum values for each of the descriptors from MACE
    min_val = np.min(descriptor_arr, axis=0)
    max_val = np.max(descriptor_arr, axis=0)

    # Storing arrays into a numpy file to be later gathered by the workchain
    np.save(file='curr_it_db_max', arr=max_val)
    np.save(file='curr_it_db_min', arr=min_val)

    # No way to store all of the descriptors in a single array,
    # as the n_atom dimension will change according to the structure
    # This pickle object will contain a list of length n_struct,
    # that will have n_at lists inside, each containing model_size
    # lists of descriptor values.
    with open('curr_it_db_descriptors.pkl', 'wb') as f:
        pickle.dump(descriptor_dict, f)

    print('MACE descriptor generation script finished!')
