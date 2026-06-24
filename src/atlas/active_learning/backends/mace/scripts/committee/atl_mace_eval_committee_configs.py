#!/usr/bin/env python3
"""
Script to be used as an aiida PortableCode for running MACE evaluation configurations.

Functions
---------
parse_arguments():
    Parses command-line arguments for MACE evaluation configuration.

run_mace_evals(args):
    Runs MACE evaluations for each model using the provided arguments.
"""

import argparse
import os
import subprocess as sb
import time
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='MACE evaluation configuration script')
    parser.add_argument(
        '--configs', required=True, help='Path to the configuration file'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        required=True,
        help='Device to run the evaluation on',
    )
    parser.add_argument(
        '--default_dtype',
        choices=['float32', 'float64'],
        required=True,
        help='Default data type',
    )
    parser.add_argument(
        '--batch_size', type=int, required=True, help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_threads', type=int, required=True, help='Number of OpenMP threads'
    )
    parser.add_argument(
        '--compute_stress', action='store_true', help='Flag to compute stress'
    )
    parser.add_argument(
        '--return_contributions',
        action='store_true',
        help='Flag to return contributions',
    )
    parser.add_argument('--info_prefix', help='Prefix for info output')

    args = parser.parse_args()

    # Access the arguments using args.<argument_name>
    return args


def run_mace_evals(args):
    formatted_params_list = []

    # Formatting input arguments for mace_eval_configs argument parser
    arg_dict = vars(args)
    for key, val in arg_dict.items():
        if key == 'train_file' or key == 'configs':
            val = Path(val).resolve().absolute()

        if key == 'num_threads':
            num_threads = val
            continue

        if isinstance(val, str):
            curr_key = f'--{key}={val}'
        elif isinstance(val, bool):
            if val:
                curr_key = f'--{key}'
        else:
            curr_key = f'--{key}={val}'
        formatted_params_list.append(curr_key)

    # Running mace_eval_configs for each model
    model_file_list = list(Path.cwd().glob('*.model'))
    for model in model_file_list:
        execute_line_list = [
            'mace_eval_configs',
            '--model',
            str(model),
            '--output',
            f'{model.name.split(".")[0]}_output.out',
        ]
        execute_line_list.extend(formatted_params_list)

        # Using Popen to launch the command in a new process,
        # which allows to submit several evaluations at once.
        my_env = os.environ.copy()
        my_env['OMP_NUM_THREADS'] = f'{num_threads}'
        curr_proc = sb.Popen(args=execute_line_list, stdout=sb.PIPE, stderr=sb.PIPE)

        # Short wait between submisions to avoid overloading the system
        time.sleep(5)

    curr_proc.wait()


if __name__ == '__main__':
    args = parse_arguments()
    run_mace_evals(args)
