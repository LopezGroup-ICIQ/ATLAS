"""Generate an initial database from TOML config."""

import argparse
import pathlib as pl
import tomllib
import warnings
from argparse import RawTextHelpFormatter

from rich.traceback import install as traceback_install

# TODO: Remove this once the deprecation warnings are fixed
warnings.filterwarnings('ignore', category=DeprecationWarning, module='spglib')


def gen_initial_database(config_dict: dict):
    """
    Generate a initial database from TOML config.

    A ATL database can be generated from settings
    specified in a TOML file.


    Parameters
    ----------
    toml_path : str
        Path for the TOML configuration file
    """
    from atlas.core.initial_db import cli_run_gen_initial_database

    # Extract parameters from toml
    database_dict = config_dict['database']
    db_path = database_dict['database_path']
    phase_diagram_dict = config_dict['phase_diagram']
    gen_dict = config_dict['generation']
    selected_phases = list(phase_diagram_dict['phase'].keys())

    # Generating the database
    cli_run_gen_initial_database(
        db_path,
        database_dict,
        phase_diagram_dict,
        gen_dict,
        selected_phases,
        config_dict,
    )


def _read_config_file(config_file) -> dict:
    """Load a TOML configuration, with a readable error when it is missing."""
    try:
        with open(config_file, 'rb') as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        error_message = (
            f"The config file '{config_file}' does not exist. "
            'Please make sure that is the correct name or input a different path.'
        )
        raise FileNotFoundError(error_message) from e


def gen_boundary(config_dict: dict, db_path: str):
    """Recompute the latent-space map and boundary of an existing database.

    Unlike ``generate`` this does not rebuild the database: it loads the
    stored ``.xz`` and re-runs descriptor generation, autoencoder training and
    boundary detection with the ``[concave_hull]`` block of the configuration.
    """
    from atlas.core.initial_db import (
        InitialDatabase,
        generate_latent_space_from_config,
    )

    db_file = pl.Path(db_path).resolve(strict=True)
    database = InitialDatabase.load_database(db_file)

    database_dict = config_dict.get('database') or {}
    rng_seed = database_dict.get('rng_seed')
    if rng_seed is not None:
        rng_seed = int(rng_seed)

    generate_latent_space_from_config(
        database,
        config_dict.get('concave_hull') or {},
        db_dir=db_file.parent,
        rng_seed=rng_seed,
    )


def run_gen_initial_database():
    parser = argparse.ArgumentParser(
        prog='atl_gen_init_db',
        description='Generate and manage ATL initial databases.',
        formatter_class=RawTextHelpFormatter,
    )

    # Create a subparsers object
    subparsers = parser.add_subparsers(
        dest='command', help='List of available commands'
    )

    # Create the subparser for the 'report' command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate reports for atlas.',
        usage=('atl_gen_init_db report [-h]\nGenerate a report for a ATLAS database'),
    )
    report_parser.add_argument(
        '--db_path',
        '-d',
        help=('Path to the database file.'),
        metavar='<PATH>',
        default=None,
        required=True,
    )

    # Create the subparser for the 'boundary' command
    boundary_parser = subparsers.add_parser(
        'boundary',
        help='Recompute the latent-space map of an existing database.',
        usage=(
            'atl_gen_init_db boundary [-h] -c CONFIG -d DB_PATH\n'
            'Recompute descriptors, autoencoder and boundary of an existing '
            'database without rebuilding it.'
        ),
    )
    boundary_parser.add_argument(
        '-c',
        '--config_file',
        help=(
            'path pointing to a TOML settings file.\n'
            'By default `database_generation_settings.toml` will be '
            'searched in the CWD.\n'
            'Only its `[concave_hull]` section is used.'
        ),
        type=pl.Path,
        default='./database_generation_settings.toml',
        metavar='PATH',
    )
    boundary_parser.add_argument(
        '-d',
        '--db_path',
        help='Path to the database file (.xz) to recompute the map for.',
        metavar='<PATH>',
        required=True,
    )

    # Create the subparser for the 'report' command
    init_db_parser = subparsers.add_parser(
        'generate',
        help='Generate an initial database for atlas.',
        usage=(
            'atl_gen_init_db generate [-h]\nGenerate an initial database for atlas.'
        ),
    )

    init_db_parser.add_argument(
        '-c',
        '--config_file',
        help=(
            'path pointing to a TOML settings file.\n'
            'By default `database_generation_settings.toml` will be '
            'searched in the CWD.'
        ),
        type=pl.Path,
        default='./database_generation_settings.toml',
        metavar='PATH',
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Generate an initial database
    if args.command == 'generate':
        # Loading TOML config file and generating the database
        gen_initial_database(config_dict=_read_config_file(args.config_file))

    # Recompute the latent-space map of an existing database
    elif args.command == 'boundary':
        gen_boundary(_read_config_file(args.config_file), args.db_path)

    # Generate an initial database report
    elif args.command == 'report':
        db_path = pl.Path(args.db_path).resolve(strict=True)
        if db_path.exists():
            from atlas.core.initial_db import cli_gen_db_report

            # Generating the report
            cli_gen_db_report(database_path=args.db_path)
        else:
            raise FileNotFoundError(
                f"The database '{db_path}' does not exist. "
                'Please make sure that is the correct name or input a different path.'
            )


if __name__ == '__main__':
    # Use rich formatter for tracebacks
    traceback_install(
        width=88,
    )

    run_gen_initial_database()
