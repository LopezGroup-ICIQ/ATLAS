#!/usr/bin/env python
"""Run an active-learning procedure based on ML-MD using aiida."""

import argparse
import pathlib as pl
import shutil
import sys
import warnings
from argparse import RawTextHelpFormatter

import tomli
from aiida import load_profile
from aiida.engine import run, submit
from aiida.orm import Dict, Int
from aiida.plugins import WorkflowFactory
from gunicorn.app.wsgiapp import WSGIApplication

from MatDBForge.active_learning.dashboard.training_dashboard_flask import (
    run_training_dashboard,
)
from MatDBForge.core import MDB_DATA_DIR

warnings.filterwarnings("ignore")


class StandaloneApplication(WSGIApplication):
    def __init__(self, app_uri, options=None):
        self.options = options or {}
        self.app_uri = app_uri
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)


def run_dashboard_app(process_id, port, update_interval, debug, online):
    print(
        f"Running dashboard to monitor process: {process_id}."
        f"Access: http://127.0.0.1:{port}."
    )
    print("Pres Ctrl+C to stop the dashboard.")
    if debug:
        app = run_training_dashboard(
            workchain_node_id=process_id,
            refresh_interval={update_interval},
            port={port},
        )
        if online:
            app.run(debug=True, port=port, host="0.0.0.0")
        else:
            app.run(debug=True, port=port, host="0.0.0.0")
    else:
        app = StandaloneApplication(
            f"MatDBForge.active_learning.dashboard.training_dashboard_flask"
            f":run_training_dashboard(workchain_node_id={process_id}, "
            f"refresh_interval={update_interval}, port={port})",
        )
        if online:
            app.options["bind"] = f"0.0.0.0:{port}"
            app.load_config()
            app.run()
        else:
            app.options["bind"] = f"127.0.0.1:{port}"
            app.load_config()
            app.run()


def create_active_learning_builder(toml_dict: dict):
    """
    Create builder object for the ActiveLearningWorkChain.

    Parameters
    ----------
    toml_dict : dict
        Dictionary coming from parsing an MDB TOML settings file.

    Returns
    -------
    ProcessBuilder
        A process builder that helps setting up the inputs for
        an ActiveLearningWorkChain.
    """
    # Getting builder for workchain
    al_calculation = WorkflowFactory("mdb-active-learning-base")
    builder = al_calculation.get_builder()

    ## General AL settings
    al_conf = toml_dict["active_learning"]
    builder.active_learning.run_name = al_conf["run_name"]
    builder.active_learning.data_path = al_conf["data_path"]
    builder.active_learning.load_init_models = al_conf.get("load_init_models")
    builder.active_learning.init_db_path = al_conf["init_db_path"]
    builder.active_learning.results_dir = al_conf["results_dir"]
    builder.active_learning.final_db_name = al_conf["final_db_name"]
    builder.active_learning.max_iterations = Int(int(al_conf["max_iterations"]))
    builder.active_learning.check_extrapolation = al_conf["check_extrapolation"]

    ## AL seed settings
    builder.active_learning.seed_size_frac = float(
        toml_dict["al_seed"]["seed_size_frac"]
    )
    builder.active_learning.seed_max_num_structs = int(
        toml_dict["al_seed"]["seed_max_num_structs"]
    )

    builder.active_learning.seed_select_settings = toml_dict["al_seed"][
        "seed_select_settings"
    ]

    builder.active_learning.committee_num_models = int(
        toml_dict["committee_eval"]["committee_num_models"]
    )
    builder.active_learning.model_acc_multiplier = float(
        al_conf["model_acc_multiplier"]
    )
    builder.active_learning.al_keep_struct_every_n_ps = float(
        al_conf["al_keep_struct_every_n_ps"]
    )

    ## MD settings
    md_params = toml_dict["md"]["parameters"]
    builder.active_learning.md_temperature_list_K = md_params["temperature_list_K"]
    builder.active_learning.md_max_temp_multiplier = md_params["max_temp_multiplier"]
    builder.active_learning.md_num_steps = int(md_params["num_steps"])
    builder.active_learning.md_timestep_duration_ps = float(
        md_params["timestep_duration_ps"]
    )
    builder.active_learning.gather_traj_cnt_lattice = md_params[
        "gather_traj_cnt_lattice"
    ]
    builder.active_learning.use_kokkos = md_params["use_kokkos"]

    # LAMMPS-MACE MD Settings
    builder.active_learning.lammps_mace = Dict(value=toml_dict["md"]["queue"])

    # MD filters
    builder.active_learning.md_filters = Dict(value=toml_dict["md"].get("filters"))

    ## MACE training settings
    builder.active_learning.mace_train = Dict(value=toml_dict["mace_train"])

    ## Committee Evaluation Settings
    builder.active_learning.committee_eval = Dict(value=toml_dict["committee_eval"])

    ## DFT method selection and settings
    builder.active_learning.dft_method = al_conf["dft_method"]
    if al_conf["dft_method"] == "vasp":
        builder.active_learning.dft_settings = Dict(value=toml_dict["dft"]["vasp"])
    elif al_conf["dft_method"] == "mace":
        builder.active_learning.dft_settings = Dict(value=toml_dict["dft"]["mace"])

    ## Descriptor settings
    builder.active_learning.descriptor_settings = Dict(value=toml_dict["descriptors"])

    return builder


def run_active_learning():
    parser = argparse.ArgumentParser(
        prog="run_active_learning",
        description="Launch a MDB active learning loop.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help=(
            "path pointing to a TOML settings file.\n"
            "By default `active_learning_settings.toml` will be searched in the CWD."
        ),
        type=pl.Path,
        default="./active_learning_settings.toml",
        # required=True,
        metavar="PATH",
    )

    # Create a subparsers object
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Create the subparser for the 'gui' command
    gui_parser = subparsers.add_parser(
        "gui", help="Launch a dashboard to keep track of the active learning loop"
    )

    # Add arguments specific to the 'gui' subcommand
    gui_parser.add_argument(
        "--update_interval",
        help=("Refresh time interval in seconds"),
        type=int,
        default=60,
        metavar="n_sec",
    )
    gui_parser.add_argument(
        "--port",
        help=("Port to use for the webapp"),
        type=int,
        default=8000,
        metavar="port",
    )

    gui_parser.add_argument(
        "--debug",
        help=("Enable Flask debug"),
        action="store_const",
        const=True,
        default=False,
    )
    gui_parser.add_argument(
        "--online",
        help=("Enable online"),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    # Loading TOML config file
    try:
        with open(args.config_file, "rb") as f:
            toml_dict = tomli.load(f)
    except FileNotFoundError as e:
        error_message = (
            f"The config file '{args.config_file}' does not exist. "
            "Please make sure that is the correct name or input a different path."
        )
        raise FileNotFoundError(error_message) from e

    # Loading default aiida profile
    load_profile(profile=toml_dict["active_learning"]["aiida_profile"])

    # Parsing settings from TOML and creating builder for aiida
    builder = create_active_learning_builder(toml_dict)

    if args.command != "gui":
        node = run(builder)
    else:
        node = submit(builder)

        run_dashboard_app(
            process_id=node,
            port=args.port,
            update_interval=args.update_interval,
            debug=args.debug,
            online=args.online,
        )


def gen_default_config():
    parser = argparse.ArgumentParser(
        prog="gen_default_config",
        description="Generate MDB default configuration files in the TOML format.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--config_type",
        help=(
            "Type of the configuration file to be generated. Available types are:\n"
            "\t- active_learning: Configuration file for active learning loop.\n"
            "\t- initial_db: Configuration file for initial database generation.\n"
        ),
        type=str,
        required=True,
        choices=["active_learning", "initial_db"],
        # default="./active_learning_settings.toml",
        metavar="TYPE",
    )

    parser.add_argument(
        "-p",
        "--path",
        help=(
            "Path in which to store the file.\n"
            "Will use the CWD by default. Folders will be created if necessary."
        ),
        type=pl.Path,
        default=".",
        metavar="PATH",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help=("Whether to overwrite the destination file, if existent."),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    default_config_name = "active_learning_settings.toml"

    # Choosing config file to write.
    if args.config_type == "active_learning":
        config_file_path = pl.Path(MDB_DATA_DIR) / "input_files" / default_config_name

    # Copying file to path.
    final_path: pl.Path = args.path / default_config_name

    if not final_path.exists() or args.overwrite:
        shutil.copy(config_file_path, args.path) #if not active_learning in line 303, no config_file_path are generated.
        print(
            f"Saved file {default_config_name} in path '{final_path.absolute()}'",
        )
    else:
        print(
            "File already exists. Not overwriting as flag -o / --overwrite not set.",
        )
        sys.exit(1)


def monitor_al_loop():
    parser = argparse.ArgumentParser(
        prog="monitor_al_loop",
        description="Monitor a MDB active learning loop.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--process_id",
        help=("Process id (pk/uuid) of the WorkChain to monitor.\n"),
        type=str,
        metavar="UUID/PK",
    )
    parser.add_argument(
        "--update_interval",
        help=("Refresh time interval in seconds"),
        type=int,
        default=60,
        metavar="n_sec",
    )
    parser.add_argument(
        "--port",
        help=("Port to use for the webapp"),
        type=int,
        default=8000,
        metavar="port",
    )
    parser.add_argument(
        "--debug",
        help=("Enable Flask debug"),
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--online",
        help=("Enable online"),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    run_dashboard_app(
        process_id=args.process_id,
        port=args.port,
        update_interval=args.update_interval,
        debug=args.debug,
        online=args.online,
    )


def parse_input_toml(toml_dict: dict, type: str):
    """
    Parses and validates the input TOML dictionary based on the specified type.

    Parameters
    ----------
    toml_dict : dict
        The input dictionary parsed from a TOML file.
    type : str
        The type of configuration to validate. Currently supports "active_learning".

    Raises
    ------
    MissingMandatoryParameterError
        If any mandatory keys are missing from the input TOML dictionary.
    """
    if type == "active_learning":
        mandatory_keys_list = ["active_learning", "md", "committee_eval", "dft"]

        for key in mandatory_keys_list:
            if key not in list(toml_dict.keys()):
                raise MissingMandatoryParameterError(
                    f"Input toml file missing mandatory key: {key}."
                )
    elif type == "generate_database":
        raise NotImplementedError("Database generation toml not implemented yet.")


class MissingMandatoryParameterError(Exception):
    """Raised when a mandatory parameter is missing in the toml dictionary."""
