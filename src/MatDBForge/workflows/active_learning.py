"""Definition of an aiida workchain for MACE active learning loops using MD."""

import io
import pickle
import re
import shutil
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pymatgen.io.ase as pmg_ase
import torch
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessSpec,
    WorkChain,
    append_,
    if_,
    while_,
)
from aiida.orm import (
    Bool,
    CalcJobNode,
    Dict,
    Float,
    FolderData,
    Group,
    Int,
    List,
    PortableCode,
    SinglefileData,
    Str,
    load_code,
    load_computer,
    load_group,
    load_node,
    to_aiida_type,
)
from aiida.plugins import CalculationFactory
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from mace import data as mace_data
from mace import tools as mace_tools
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData

from MatDBForge import ROOT_DIR
from MatDBForge.active_learning import active_learning_utils as mdb_al_ut
from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.core import DATA_DIR


class ActiveLearningWorkChain(WorkChain):
    """
    WorkChain to run an active learning loop for a MACE potential using MD
    simulations to generate training data.
    """

    @classmethod
    def define(cls, spec: ProcessSpec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.input("init_db_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("final_db_name", valid_type=Str, serializer=to_aiida_type)
        spec.input("run_name", valid_type=Str, serializer=to_aiida_type)
        spec.input("data_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("results_dir", valid_type=Str, serializer=to_aiida_type)
        spec.input("al_loop_iteration", valid_type=Int, serializer=to_aiida_type)
        spec.input("seed_size_frac", valid_type=Float, serializer=to_aiida_type)
        spec.input("commitee_num_models", valid_type=Int, serializer=to_aiida_type)
        spec.input("model_acc_multiplier", valid_type=Float, serializer=to_aiida_type)

        # MD settings
        spec.input("md_temperature_list_K", valid_type=List, serializer=to_aiida_type)
        spec.input("md_num_steps", valid_type=Int, serializer=to_aiida_type)
        spec.input("md_max_temp_multiplier", valid_type=Float, serializer=to_aiida_type)
        spec.input(
            "md_timestep_duration_ps",
            valid_type=Float,
            serializer=to_aiida_type,
        )
        spec.input(
            "al_keep_struct_every_n_ps",
            valid_type=Float,
            serializer=to_aiida_type,
        )
        spec.input("current_md_seed_structs", valid_type=List, serializer=to_aiida_type)
        spec.input("gather_traj_cnt_lattice", valid_type=Bool, serializer=to_aiida_type)

        # Safeguard settings
        spec.input(
            "safeguard_enable",
            required=True,
            valid_type=Bool,
            serializer=to_aiida_type,
        )
        spec.input(
            "structure_selection_method",
            valid_type=Str,
            serializer=to_aiida_type,
        )
        spec.input(
            "safeguard_user_defined_struct_path",
            valid_type=Str,
            serializer=to_aiida_type,
        )
        spec.input(
            "safeguard_temperature_K", valid_type=Float, serializer=to_aiida_type
        )
        spec.input(
            "safeguard_max_temp_multiplier",
            valid_type=Float,
            serializer=to_aiida_type,
        )
        spec.input("safeguard_num_steps", valid_type=Int, serializer=to_aiida_type)
        spec.input(
            "safeguard_timestep_duration_ps",
            valid_type=Float,
            serializer=to_aiida_type,
        )
        spec.input(
            "safeguard_gather_traj_cnt_lattice",
            valid_type=Bool,
            serializer=to_aiida_type,
        )

        # Path settings
        spec.input("seed_db_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("training_db_path", valid_type=Str, serializer=to_aiida_type)
        spec.input(
            "current_md_seed_structs_idx",
            valid_type=List,
            serializer=to_aiida_type,
        )
        spec.input(
            "train_seed_group",
            valid_type=Str,
            serializer=to_aiida_type,
        )
        spec.input(
            "mace_train",
            valid_type=Dict,
            serializer=to_aiida_type,
        )
        spec.input("lammps_mace", valid_type=Dict)
        spec.input("dft_settings", valid_type=Dict)
        spec.input("committee_eval", valid_type=Dict)
        spec.input("check_extrapolation", valid_type=Bool, serializer=to_aiida_type)

        spec.outline(
            # Training the main mace model (M0) and the commitee models
            # using the training database (Dt).
            cls.train_mace_model,
            # Gathering results from mace training.
            cls.get_mace_train_output,
            if_(cls.check_extrapolation_enabled)(
                # Generate MACE descriptors for the current seed.
                cls.generate_descriptors,
                # Gather the descriptors from the calcjob and store them
                # in the workchain context.
                cls.get_mace_descriptors_output,
            ),
            # All of the structures in the seed will be run using the MD
            # code selected, using the main model (M0)
            cls.run_md_seed,
            # Structures and energy predictions will be gathered and prepared
            # into a dataframe
            cls.gather_m0_md_results,
            # The structures from M0 will be evaluated using M1, M2 and M3.
            cls.check_commitee_results,
            if_(cls.check_extrapolation_enabled)(
                # Getting MACE descriptors for the structures obtained with MD.
                cls.get_descriptors_from_md_results,
            ),
            # According to the difference in error between the models either:
            # The original structure will be removed from D0, or
            # The problematic structure will be calcualated using DFT
            cls.send_calc_or_remove_structures,
            # Return the generated DFT calculations to the workchain as an output
            cls.return_seed_dft,
            # TODO: If high error (define this) is found on a training seed,
            # do not change seed until the error is decreased
            # cls.choose_next_seed,
        )
        spec.output("dft_calcs", valid_type=List, required=False)
        spec.output("m0_model_file", valid_type=SinglefileData)
        spec.output("stop_md_seed_no_disagreement", valid_type=Bool)

        spec.exit_code(
            420, "ERROR_SCHEDULER_MACE", "error when submitting a MACE calculation."
        )

    def train_mace_model(self):
        """
        Setup and submit TrainMACEModelCalculation for MACE model training.

        This function generates a new training database file by appending the
        workchain UUID to the original database file name. It then proceeds to train
        a specified number of MACE models using settings defined in the workflow
        inputs. The function updates the workchain context with the futures of the
        submitted calculation jobs, allowing for the tracking of these jobs. The most
        accurate model from these trainings is selected in a later step to drive MD
        simulations, while the others serve as committee models for energy evaluation.

        Returns
        -------
        None
            The function does not return a value but updates the workchain context with
            futures of the submitted TrainMACEModelCalculation jobs for later reference.
        """
        self.report("Generating new training database file.")

        # Adding workchain uuid input.data filename to path
        updated_path, _ = mdb_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=mdb_al_ut.process_call_root(load_node(self.uuid)),
        )

        database_training = mdb_al_ut.load_database(self.inputs.training_db_path.value)

        # Generate new training data file
        mdb_conv.gen_mace_train_structure_list(
            path=updated_path,
            structure_list=database_training,
        )

        # Train n models (M0-Mn)
        # The most accurate model (during validation) will be chosen as the main model,
        # and used to drive the MD simulations. The remaining models will act as
        # commitee models and will only be used to evaluate energies.
        self.report(
            f"Training {self.inputs.commitee_num_models.value} models using "
            "current iteration data."
        )

        for _ in range(self.inputs.commitee_num_models.value):
            model_name = mdb_al_ut.generate_model_name()

            # Load training settings from inputs and update path and model names.
            mace_train_settings: Dict = mdb_al_ut.update_mace_train_settings_dict(
                settings_dict=self.inputs.mace_train.get("train_settings"),
                train_data_path=str(updated_path),
                curr_model=model_name,
                curr_iter=self.inputs.al_loop_iteration.value,
            )

            # Run training and save new model file
            mace_train = CalculationFactory("mace-train")
            mace_builder = mace_train.get_builder()

            mace_builder.model_name = model_name
            mace_builder.mace_settings_dict = Dict(mace_train_settings)

            mace_train_file_path, _ = mdb_al_ut.get_final_db_path(
                result_dir_path=self.inputs.results_dir.value,
                final_db_name=self.inputs.final_db_name.value,
                node=self.node,
            )
            mace_builder.mace_train_file_path = str(mace_train_file_path)

            mace_builder.code = load_code(self.inputs.mace_train.dict.code)
            mace_builder.metadata.options.withmpi = True
            mace_builder.metadata.options = self.inputs.mace_train.dict.metadata.get(
                "options"
            )
            mace_builder.metadata.options.output_filename = (
                f"train_{model_name}_iter-{self.inputs.al_loop_iteration.value}"
            )
            mace_builder.metadata.label = model_name

            future = self.submit(mace_builder)
            self.to_context(mace_training_results=append_(future))

    def get_mace_train_output(self):
        """
        Retrieve and process MACE training output for model selection.

        This function evaluates MACE training results to identify the model with the
        best performance based on a weighted sum of RMSE values for energy (E) and
        forces (F). It selects the model with the lowest weighted E+F, considers
        the importance of forces via a weighting factor, and processes the best model
        to create a LAMMPS-compatible potential file. Information about the selected
        model and committee models is updated in the context for further use.

        Returns
        -------
        None
            The function updates the workchain context with the best model's RMSE values, the
            LAMMPS potential file, and the committee models' information but does not
            return any value directly.
        """
        mace_training_results = self.ctx.mace_training_results

        model_name_list = []
        weighted_E_F_sum_list = []

        # Iterate over all models and get weighted sum of E and F
        for calc in mace_training_results:
            # Loading calculation node
            curr_calc: CalcJobNode = load_node(calc.uuid)

            # Skipping model if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                continue

            # Getting model name
            model_name_list.append(curr_calc.inputs.model_name.value)

            # Adding E + F multiplied by a weight value, in order to consider
            # forces when deciding which model to keep
            force_weight = self.inputs.mace_train.get(
                "result_force_weight",
                0.1,
            )
            weighted_E_F_sum = curr_calc.outputs.m_rmse_e.value + (
                force_weight * curr_calc.outputs.m_rmse_f
            )
            weighted_E_F_sum_list.append(weighted_E_F_sum)

        # Get the most accurate model name during the validation by checking
        # the lowest E+F*weight.
        self.ctx.best_model_name = model_name_list[np.argmin(weighted_E_F_sum_list)]

        commitee_models_tupl_name_uuid = []
        for calc in mace_training_results:
            # Loading calculation node
            curr_calc: CalcJobNode = load_node(calc.uuid)

            # Skipping model if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                self.report("Skipping CalcJob with errors.")
                continue

            # Loading model name
            model_name = curr_calc.inputs.model_name.value

            # Getting model file
            model_file = curr_calc.outputs.model_file

            # Use checking if the currentt case is the best model.
            # Saving model results
            if model_name == self.ctx.best_model_name:
                # Overwriting m0_rmse values with actual training values
                self.ctx.m0_rmse_e = curr_calc.outputs.m_rmse_e
                self.ctx.m0_rmse_f = curr_calc.outputs.m_rmse_f

                # Convert model to LAMMPS compatible format
                # and return it to workchain context
                self.ctx.lammps_potential_file = mdb_al_ut.create_mace_lammps_model(
                    model_file, self.ctx.m0_rmse_e, self.ctx.m0_rmse_f
                )

                self.report(
                    f"Generated LAMMPS potential using '{model_name}' as M0 - "
                    f"RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV/at, "
                    f"RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV/Å"
                )
                self.out("m0_model_file", model_file)
            else:
                self.report(
                    f"Trained commitee model '{model_name}' - "
                    f"RMSE E: {curr_calc.outputs.m_rmse_e.value:.3f} meV/at, "
                    f"RMSE F: {curr_calc.outputs.m_rmse_f.value:.3f} meV/Å"
                )
                commitee_models_tupl_name_uuid.append((model_name, model_file.uuid))

        # Sending commitee model paths to current context
        self.ctx.commitee_models_tupl_name_uuid = commitee_models_tupl_name_uuid

    def check_extrapolation_enabled(self):
        return self.inputs.check_extrapolation

    def generate_descriptors(self):
        self.report("Generating descriptors...")

        for _, calc in enumerate(self.ctx.mace_training_results):
            # Loading calculation node
            curr_calc = load_node(calc.uuid)

            # Getting model name
            model_name = curr_calc.inputs.model_name.value

            # Using the best model
            if model_name == self.ctx.best_model_name:
                best_calc = curr_calc
                break

        # Getting model file
        self.ctx.best_model_file = best_calc.outputs.model_file

        # Prepare GetMACEDescriptorsCalculation
        mace_descr_calc = CalculationFactory("mace-get-descriptors")
        mace_builder = mace_descr_calc.get_builder()
        mace_builder.model_file = self.ctx.best_model_file

        mace_train_file_path, _ = mdb_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=self.node,
        )
        mace_builder.mace_train_file_path = str(mace_train_file_path)
        descriptor_code_path = Path(f"{ROOT_DIR}/active_learning/mace_code")
        code = PortableCode(
            label="mace_get_descriptors",
            filepath_files=descriptor_code_path,
            filepath_executable="./mace_get_descriptors.py",
            # TODO: Add to TOML
            prepend_text="source /gpuscratch/psanz/mace/mace-venv/bin/activate",
        )
        mace_builder.code = code

        # TODO: Add to TOML
        mace_builder.metadata.options = {
            "resources": {
                "parallel_env": "c128m1024ib_mpi_32slots",
                "tot_num_mpiprocs": 4,
            },
            "queue_name": "c128m1024ibgpu4.q",
            "max_memory_kb": 102400000,
            "parser_name": "mace-descriptors-parser",
            "max_wallclock_seconds": 117280000,
            "withmpi": False,
            "custom_scheduler_commands": "#$ -l gpu=1",
        }
        mace_builder.metadata.label = model_name + "_descriptors"
        mace_builder.metadata.computer = load_computer("tekla2-new-test")

        mace_builder.metadata.options.output_filename = (
            f"descriptors_{model_name}_iter-{self.inputs.al_loop_iteration.value}"
        )

        future = self.submit(mace_builder)
        self.to_context(mace_descriptor_results=append_(future))

    def get_mace_descriptors_output(self):
        mace_descriptor_results = self.ctx.mace_descriptor_results
        for calc in mace_descriptor_results:
            # Loading calculation node
            curr_calc = load_node(calc.uuid)

            # Storing results in context
            self.ctx.descriptors_min_array = (
                curr_calc.outputs.descriptors_min_array.get_array()
            )

            self.ctx.descriptors_max_array = (
                curr_calc.outputs.descriptors_max_array.get_array()
            )

        self.report("Gathered descriptor ranges.")

    def gen_md_input(
        self,
        structure: Structure,
        potential_path: str,
        current_temp: float,
        temp_coeff: float,
        timestep_val: float,
        num_timestep: int,
    ) -> str:
        """
        Generate a MACE-LAMMPS input file for MD simulations using a template.

        This function creates a customized MACE-LAMMPS input file for molecular dynamics
        simulations by reading and modifying a template file. The modifications include
        setting the pair style to a MACE-based style, defining pair coefficients using
        the provided potential file, specifying simulation parameters such as timestep
        size, temperature, initial velocities, and the number of timesteps. The function
        utilizes the structure's composition to dynamically adjust the input file's
        content, catering to the specifics of the simulation's atomic species and the
        applied potential.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            The structure object containing composition information used to
            set species-specific parameters in the LAMMPS input file.
        potential_path : str
            Path to the potential file to be used in the pair_coeff directive.
        current_temp : float
            The temperature value (in Kelvin) to be used for setting initial velocities
            and maintaining simulation temperature
        temp_coeff : float
            The temperature coefficient to calculate the maximum temperature.
        timestep_val : float
            The length of the timestep for the simulation (in picoseconds).
        num_tsteps : int
            The total number of timesteps to be executed in the simulation.

        Returns
        -------
        str
            The modified LAMMPS input file content as a string, ready for use in
            simulation.

        Notes
        -----
        - The template file is located in the `DATA_DIR/input_files` directory.
        - This function replaces placeholders in the template with actual values from
        the function's inputs and the structure's composition.
        - Future versions may include more dynamic options based on LAMMPS's extensive
        configurability.
        """
        with open(f"{DATA_DIR}/input_files/input.lammps") as f:
            lammps_template = f.read()

        lammps_template = lammps_template.replace(
            "$MACESTYLE", "mace no_domain_decomposition"
        )

        species = structure.composition.elements

        # Setting MACE potential as the potential to use
        pair_coeff_str = "* * "
        pair_coeff_str += f"{Path(potential_path).name} "

        # Adding species from given structure
        for spec in species:
            pair_coeff_str += f"{spec} "
        lammps_template = lammps_template.replace("$PAIRCOEFF", pair_coeff_str)

        # Setting elements
        elem_str = ""
        for elem in species:
            elem_str += f"{elem} "

        lammps_template = lammps_template.replace("$ELEMS", elem_str)

        # Setting timestep size
        lammps_template = lammps_template.replace("$TSTEP_SIZE", str(timestep_val))

        # Setting start and end temperature and the damping parameter.
        # The max T is calculated using a multiplier applied to the initial T.
        # The damping coefficient is computed as 100*dt as by the lammps docs,
        # see note in: https://docs.lammps.org/fix_nh.html#description
        temp_arr = f"{current_temp} {current_temp*temp_coeff} {100 * timestep_val}"
        lammps_template = lammps_template.replace("$TEMPARR", temp_arr)

        # Setting intial velocities.
        seed = np.random.randint(low=1, high=1000000)
        vel_str = f"{current_temp} {seed}"
        lammps_template = lammps_template.replace("$VELOCITY", vel_str)

        # Setting number of timesteps
        lammps_template = lammps_template.replace("$NSTEPS", str(num_timestep))

        return lammps_template

    def run_md_seed(self):
        """
        Run MD simulations for all structures in the current training seed using M0.

        This function initiates molecular dynamics simulations for each structure
        within the current training seed, utilizing a predefined main model (M0).
        It generates and configures MACE-LAMMPS calculation jobs, sets up the necessary
        input files and parameters, and submits these jobs for execution.
        Calculation nodes are stored and managed within the workflow's context for
        later retrieval and analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The function does not return a value but updates the workflow context with
            futures of the submitted MD simulation jobs, facilitating tracking and
            subsequent analysis of these simulations.

        Notes
        -----
        - This function assumes the availability of a trained MACE-LAMMPS potential file
        within the workflow's context.
        - Submitted calculation jobs are added to an AiiDA group for organization and
        are tagged with additional information to link them back to their respective
        positions in the database.
        """
        self.report("Running MD (using M0) for all structures in the current seed...")

        # Creating a list in the context to store the nodes
        self.ctx.current_train_seed = []

        # This string with the label used in the code setup.
        code_str = self.inputs.lammps_mace.get("code")
        builder = CalculationFactory("lammps.raw").get_builder()
        builder.code = load_code(code_str)

        # Getting the lammps potential file in a temporary folder
        with self.ctx.lammps_potential_file.as_path() as lmp_pot_path:
            lmp_pot_filename = Path(lmp_pot_path).name
            lmp_pot_path = str(lmp_pot_path)

            # Setting the trajectory to be retrieved and the
            # potential file to be copied into the calculation folder
            builder_settings = {
                "additional_retrieve_list": ["structure.lammpstrj"],
                "local_copy_list": [
                    (
                        self.ctx.lammps_potential_file.uuid,
                        lmp_pot_path,
                        lmp_pot_filename,
                    )
                ],
            }

            builder.settings = Dict(builder_settings)

            for idx, curr_structure in enumerate(self.inputs.current_md_seed_structs):
                # Structures are stored as a dict in order to be json-serializable
                for key in ["pbc", "cell", "numbers", "positions", "forces"]:
                    curr_structure[key] = np.array(curr_structure[key])

                curr_structure = Atoms.fromdict(curr_structure)

                # Converting structure to pymatgen
                curr_structure = pmg_ase.AseAtomsAdaptor.get_structure(curr_structure)
                struct_properties = curr_structure.properties

                # Running a MD calculation for every T specified by the user
                for temp_val in self.inputs.md_temperature_list_K:
                    # TODO: Add to TOML. Check how to include parameters needed here as
                    # initial parameters (TOML)
                    curr_input = self.gen_md_input(
                        structure=curr_structure,
                        potential_path=lmp_pot_filename,
                        current_temp=temp_val,
                        temp_coeff=self.inputs.md_max_temp_multiplier.value,
                        timestep_val=self.inputs.md_timestep_duration_ps.value,
                        num_timestep=self.inputs.md_num_steps.value,
                    )

                    script = SinglefileData(io.StringIO(curr_input))
                    builder.script = script

                    lammps_struct_str = LammpsData.from_structure(
                        curr_structure, atom_style="atomic"
                    ).get_str()

                    data = SinglefileData(io.StringIO(lammps_struct_str))
                    builder.files = {
                        "data": data,
                        "mace_potential": self.ctx.lammps_potential_file,
                    }
                    builder.filenames = {
                        "data": "structure.lammps",
                        "mace_potential": lmp_pot_filename,
                    }

                    index_in_db = self.inputs.current_md_seed_structs_idx[idx]

                    # Loading metadata settings from workchain inputs
                    builder.metadata = self.inputs.lammps_mace.get("metadata")
                    builder.metadata.label = (
                        f"struct_{index_in_db}_mace_lammps_md_{temp_val}_K"
                    )

                    # Submitting current calculation
                    future = self.submit(builder)

                    # Add calculation to the workchain's aiida group.
                    self.ctx.current_train_seed.append(future)
                    curr_group = load_group(uuid=self.inputs.train_seed_group.value)
                    curr_group.add_nodes(future)

                    # Writing extra information that helps associating the calculation
                    # with its position on the database.
                    for key, val in struct_properties.items():
                        future.base.extras.set(key, val)
                    future.base.extras.set("index_in_db", index_in_db)
                    future.base.extras.set("md_temperature", temp_val)

                    # Telling the work chain to wait for the md to finish
                    # before continuing the workflow.
                    # We append the future to a list of workflows.
                    self.to_context(md_seed_workchains=append_(future))

    def gather_m0_md_results(self):
        """
        Gather MD simulation results for all structures in the current seed.

        This function collects the results from MD simulations of each structure within
        the current training seed.
        It extracts trajectories, energies, and forces from each simulation's output and
        aggregates these into a structured format. The collected data is then organized
        into a pandas DataFrame, which is stored in the workflow's context.
        """
        self.report("Gathering best model MD results for the current seed...")
        new_rows = []

        # Gathering all results
        for workchain in self.ctx.md_seed_workchains:
            workchain_results = workchain.outputs.retrieved
            steps_E_F_arr = self.gather_energies_from_workchain(workchain_results)
            traj, forces = self.gather_traj_from_workchain(workchain_results)

            # Instead of keeping all frames, select some of them
            # Get 1 frame every n picoseconds of MD simulation
            traj, steps_E_F_arr, forces = mdb_al_ut.select_md_frames_to_keep(
                frame_interval=self.inputs.al_keep_struct_every_n_ps,
                total_n_frames=self.inputs.md_num_steps.value,
                md_tstep_duration_ps=self.inputs.md_timestep_duration_ps.value,
                traj=traj,
                steps_E_F_arr=steps_E_F_arr,
                forces=forces,
            )

            new_rows.append(
                {
                    "trajectory": traj,
                    "energy": {"m0": steps_E_F_arr[:, 1]},
                    "forces": {"m0": forces},
                    "al_step": self.inputs.al_loop_iteration.value,
                    "index_in_db": workchain.base.extras.all["index_in_db"],
                    "mdb_struct_type": workchain.base.extras.all["mdb_struct_type"],
                    "material_name": workchain.base.extras.all["struct_name"],
                    "unique_id": workchain.base.extras.all["aiida_uuid"],
                    "md_temperature": workchain.base.extras.all["md_temperature"],
                    "extrapolation": np.nan,
                }
            )

        # Creating a DataFrame with all the results
        self.ctx.md_seed_results_df = pd.DataFrame(new_rows)

    def gather_energies_from_workchain(self, workchain_results):
        """
        Extracts energy, force, and step information from a 'lammps.out' output file
        contained within a workchain result.

        This function parses the 'lammps.out' output file from a given set of workchain
        results. It specifically looks for lines starting with 'thermo ', then extracts
        step number, energy, and force values from these lines. The extracted values are
        compiled into a numpy array with each column representing steps, energies, and
        forces, respectively.

        Parameters
        ----------
        workchain_results : object
            An object containing the results of a workchain. It is expected to have
            a method `get_object_content` which can retrieve the content of 'lammps.out'.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array where each row corresponds to a step in the workchain.
            The columns represent step number, energy, and force,respectively.

        Notes
        -----
        - The function assumes that the 'lammps.out' file has a specific format where
          relevant data is prefixed with 'thermo '.
        - The function starts processing from the second occurrence of lines starting
          with 'thermo '.
        - The function assumes that the step number is at index 1, energy at index 4,
          and force at index 6 of each relevant line.

        Examples
        --------
        >>> for workchain in self.ctx.md_seed_workchains:
        >>>     workchain_results = workchain.outputs.retrieved
        >>>     step_E_F_arr = gather_energies_from_workchain(workchain_results)
        >>>     print(step_E_F_arr)
        [[1, energy1, force1],
        [2, energy2, force2],
        ...]

        """
        output = workchain_results.get_object_content("lammps.out")
        steps = [line for line in output.splitlines() if line.startswith("thermo ")]

        energy_array = []
        force_array = []
        step_array = []

        for step in steps:
            split = step.split()
            step_array.append(int(split[1]))
            energy_array.append(float(split[4]))
            force_array.append(float(split[7]))

        step_E_F_arr = np.stack((step_array, energy_array, force_array), axis=1)
        return step_E_F_arr

    def gather_traj_from_workchain(self, workchain_results: FolderData) -> Trajectory:
        """
        Extracts trajectory data from a `LammpsRawCalculation` as pymatgen Trajectory.

        This function parses `structure.lammpstrj` from the given workchain
        results to extract atomic coordinates and lattice information.
        It then constructs a sequence of pymatgen Structure objects
        representing each frame of the trajectory which are combined
        into a pymatgen Trajectory object.

        Parameters
        ----------
        workchain_results : FolderData
            A FolderData containing the results of a workchain, expected to have
            a method `get_object_content` to retrieve the contents of `structure.lammpstrj`.

        Returns
        -------
        Trajectory
            A pymatgen Trajectory object representing the structure over time.
        np.array
            A (n_frames x n x 3) numpy array containing the
            forces of every atom.

        Notes
        -----
        The user can change the `gather_traj_cnt_lattice` input to True/False to
        select if the lattice changes during the simulation.
        The function extracts the num of atoms and frames from the traj file.
        The step duration is `self.inputs.md_timestep_duration_ps.value`
        """
        # Get trajectory file from aiida repo node
        traj_data = workchain_results.get_object_content("structure.lammpstrj")

        # Separate the file the file into lines
        lines = traj_data.splitlines()

        # Typical LAMMPS structure header size
        offset = 9

        # Get the number of atoms
        num_atoms_list = re.findall(
            r"ITEM: NUMBER OF ATOMS\s\d*",
            string=traj_data,
        )
        num_atoms = int(num_atoms_list[0].split("\n")[1])

        # Get the total number of frames
        num_frames_list = re.findall(r"ITEM: TIMESTEP\s\d*", string=traj_data)
        num_frames = int(num_frames_list[-1].split("\n")[1])

        # assembling a pymatgen structure
        struct_list = []
        forces_list = []
        line_posc = 0
        for curr_struct in range(0, num_frames + 1):
            line_posc_ini = line_posc
            line_posc += num_atoms + offset
            curr_struct_list = lines[line_posc_ini:line_posc]
            curr_struct_info = np.array([line.split() for line in curr_struct_list[9:]])

            curr_struct_coords = curr_struct_info[:, :5]

            # This will skip the current structure if the number of atoms is
            # different than expected.
            # This situation may arise when the potential is not good
            # enough and results in very high forces applied to the structure,
            # sending some atoms outside of the cell.
            if curr_struct_coords.shape[0] != num_atoms:
                continue

            curr_struct_forces = curr_struct_info[:, 5:].astype(np.float32)

            # Lammps box bounds snapshot format
            # xlo_bound xhi_bound xy
            # ylo_bound yhi_bound xz
            # zlo_bound zhi_bound yz
            lattice = np.zeros([3, 3])
            lattice_vals = np.array(
                [[vec.split()[1]] for vec in lines[5:8]], dtype=float
            )
            cnt = 0
            for posc, row in enumerate(lattice):
                row[cnt] = lattice_vals[posc]
                cnt += 1

            species = curr_struct_coords[:, 1]
            coord_array = curr_struct_coords[:, 2:5].astype(np.float32)

            curr_struct = Structure(
                lattice=lattice,
                species=species,
                coords=coord_array,
                coords_are_cartesian=True,
            )
            struct_list.append(curr_struct)
            forces_list.append(curr_struct_forces)

        # This flag selects if a constant lattice volume is assumed
        # for all frames.
        cnt_lat_setting = self.inputs.get(
            "gather_traj_cnt_lattice",
            True,
        )
        if isinstance(cnt_lat_setting, Bool):
            cnt_lat_setting = cnt_lat_setting.value

        traj = Trajectory.from_structures(
            struct_list,
            constant_lattice=cnt_lat_setting,
            time_step=self.inputs.md_timestep_duration_ps.value,
        )

        return traj, np.array(forces_list)

    # TODO: Implement a CalcJob for this?
    def check_commitee_results(self):
        self.report("Evaluating trajectories with models...")

        for row in self.ctx.md_seed_results_df.iterrows():
            # self.report(f"Checking struct {row[0]} results with all models...")
            curr_traj = row[1]["trajectory"]

            # Working with all models
            for model_name, model in self.ctx.commitee_models_tupl_name_uuid:
                with load_node(model).as_path() as model_path:
                    # Getting device type from inputs. If not set, CPU will be used as a
                    # fallback.
                    device_type = self.inputs.committee_eval.get_dict().get(
                        "device", "cpu"
                    )
                    device = mace_tools.torch_tools.init_device(device_type)

                    # Setting dtype. float32 will be set as default.
                    dtype = self.inputs.committee_eval.get_dict().get(
                        "dtype", "float32"
                    )
                    mace_tools.torch_tools.set_default_dtype(dtype)

                    # Load MACE model
                    model = torch.load(f=model_path, map_location=device_type)
                    model = model.to(device_type)

                    for param in model.parameters():
                        param.requires_grad = False

                    # Load data and prepare input
                    atoms_list = [
                        AseAtomsAdaptor().get_atoms(pym_struct)
                        for pym_struct in curr_traj
                    ]

                    configs = [
                        mace_data.config_from_atoms(atoms) for atoms in atoms_list
                    ]

                    z_table = mace_tools.utils.AtomicNumberTable(
                        [int(z) for z in model.atomic_numbers]
                    )

                    batch_size = self.inputs.committee_eval.get_dict().get(
                        "batch_size", 64
                    )
                    data_loader = mace_tools.torch_geometric.dataloader.DataLoader(
                        dataset=[
                            mace_data.AtomicData.from_config(
                                config, z_table=z_table, cutoff=float(model.r_max)
                            )
                            for config in configs
                        ],
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                    )

                    # Collect data
                    energies_list = []
                    forces_collection = []

                    compute_stress = self.inputs.committee_eval.get_dict().get(
                        "compute_stress", True
                    )
                    for batch in data_loader:
                        batch = batch.to(device)
                        output = model(batch.to_dict(), compute_stress=compute_stress)
                        energies_list.append(
                            mace_tools.torch_tools.to_numpy(output["energy"])
                        )

                        forces = np.split(
                            mace_tools.torch_tools.to_numpy(output["forces"]),
                            indices_or_sections=batch.ptr[1:],
                            axis=0,
                        )
                        forces_collection.append(forces[:-1])  # drop last as its empty

                    energies = np.concatenate(energies_list, axis=0)
                    forces_list = [
                        forces
                        for forces_list in forces_collection
                        for forces in forces_list
                    ]
                    assert len(atoms_list) == len(energies) == len(forces_list)

                    updated_ene_dict = row[1]["energy"]
                    updated_ene_dict.update({model_name: energies})
                    self.ctx.md_seed_results_df.at[row[0], "energy"] = updated_ene_dict
                    # TODO: Check if this is the proper way of gathering the forces
                    # forces_list = np.array(forces_list)
                    # forces_norm = np.linalg.norm(forces_list, axis=2)
                    # total_force_norm_per_frame = forces_norm.sum(axis=1)

                    self.ctx.md_seed_results_df.at[row[0], "forces"][model_name] = (
                        forces_list  # total_force_norm_per_frame
                    )

    def get_descriptors_from_md_results(self):
        # Getting descriptors for generated structures
        self.report("Getting descriptors for MD generated structures...")

        # Store all frames from the trajectory into a list
        for _, row in self.ctx.md_seed_results_df.iterrows():
            # all_frames_list = []
            curr_traj = row["trajectory"]
            traj_frames = []

            for frame in curr_traj:
                curr_frame: Atoms = AseAtomsAdaptor.get_atoms(frame)
                curr_frame.info["aiida_uuid"] = row["unique_id"]
                curr_frame.info["md_temperature"] = row["md_temperature"]
                traj_frames.append(curr_frame)

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename="-",
                    format="extxyz",
                    images=traj_frames,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            md_xyz_file = SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename="md_db.xyz",
            )

            # Prepare GetMACEDescriptorsCalculation
            mace_descr_calc = CalculationFactory("mace-get-descriptors")
            mace_builder = mace_descr_calc.get_builder()
            mace_builder.model_file = self.ctx.best_model_file
            mace_builder.mace_train_file_path = md_xyz_file
            descriptor_code_path = Path(f"{ROOT_DIR}/active_learning/mace_code")
            code = PortableCode(
                label="mace_get_descriptors",
                filepath_files=descriptor_code_path,
                filepath_executable="./mace_get_descriptors.py",
                # TODO: Add to TOML
                prepend_text="source /gpuscratch/psanz/mace/mace-venv/bin/activate",
            )
            mace_builder.code = code

            # TODO: Add to TOML
            mace_builder.metadata.options = {
                "resources": {
                    "parallel_env": "c128m1024ib_mpi_32slots",
                    "tot_num_mpiprocs": 4,
                },
                "queue_name": "c128m1024ibgpu4.q",
                "max_memory_kb": 102400000,
                "parser_name": "mace-descriptors-parser",
                "max_wallclock_seconds": 117280000,
                "withmpi": False,
                "custom_scheduler_commands": "#$ -l gpu=1",
            }
            mace_builder.metadata.label = (
                row["unique_id"][:8] + "_md_descriptors_" + f"{row['md_temperature']}_K"
            )
            mace_builder.metadata.computer = load_computer("tekla2-new-test")

            mace_builder.metadata.options.output_filename = (
                f"descriptors_{self.ctx.best_model_name}_iter"
                f"-{self.inputs.al_loop_iteration.value}"
            )

            future = self.submit(mace_builder)
            future.base.extras.set("unique_id", row["unique_id"])
            future.base.extras.set("md_temperature", row["md_temperature"])

            self.to_context(md_descriptor_results=append_(future))

    def send_calc_or_remove_structures(self):
        self.report("Deciding which structures to keep...")

        model_acc_multiplier = self.inputs.model_acc_multiplier.value
        e_rmse = self.ctx.m0_rmse_e.value
        e_error_threshold = model_acc_multiplier * e_rmse

        f_rmse = self.ctx.m0_rmse_f.value
        f_error_threshold = model_acc_multiplier * f_rmse

        delete_indices = []
        dft_structures = []

        # Gathering MD descriptor results and adding them to dataframe
        if self.inputs.check_extrapolation:
            for curr_calc in self.ctx.md_descriptor_results:
                # Loading calculation node
                # curr_calc = load_node(calc.uuid)
                curr_unique_id = curr_calc.extras["unique_id"]
                curr_md_temperature = curr_calc.extras["md_temperature"]

                # Creating context manager to load descriptor result files
                # descr_file
                with curr_calc.outputs.descriptors_file.as_path() as md_descr_file_path, open(
                    md_descr_file_path, "rb"
                ) as descr_file:
                    md_descr_dict: list[list[list]] = pickle.load(descr_file)
                    # md_descr_array: np.ndarray = np.load(file=md_descr_file_path)

                # Find row matching the calculation using curr_unique_id and
                # current_temperature
                row_index = self.ctx.md_seed_results_df[
                    self.ctx.md_seed_results_df.unique_id == curr_unique_id
                ][
                    self.ctx.md_seed_results_df.md_temperature == curr_md_temperature
                ].index[0]

                # Assign matching descriptors to the extrapolation column
                desc_f_curr_row: list = md_descr_dict[curr_unique_id]

                # Overwrite row in dataframe
                self.ctx.md_seed_results_df.loc[[row_index], "extrapolation"] = (
                    pd.Series(
                        [desc_f_curr_row],
                        index=self.ctx.md_seed_results_df.index[[row_index]],
                    )
                )

        # Every row contains the results of MD for a single structure, which are:
        # trajectory, energies, forces, al_step, index_in_db, mdb_struct_type,
        # cluster, material_name, unique_id
        submitted_dft_cnt = 0
        for _, row in self.ctx.md_seed_results_df.iterrows():
            # Make len(traj) sized array filled with 'False'.
            extrapolating_frames = np.zeros(shape=len(row["trajectory"]))

            if self.inputs.check_extrapolation:
                curr_struct_descr = row["extrapolation"]

                # Checking if the frames for the current structure are extrapolating
                for frame_idx, frame_descriptors in enumerate(curr_struct_descr):
                    below_min = frame_descriptors < self.ctx.descriptors_min_array
                    above_max = frame_descriptors > self.ctx.descriptors_max_array
                    is_frame_extrapolating = np.any(np.logical_or(below_min, above_max))

                    # Change to True the ones that are extrapolating.
                    if is_frame_extrapolating:
                        extrapolating_frames[frame_idx] = 1

            # Getting all energy predictions
            # TODO - For E and F: Do variance
            model_energies_dict = row["energy"]
            energies_std = mdb_al_ut.get_model_energies_std(model_energies_dict)

            # Any True value in this array is over the energy error threshold
            # and must be sent to calculate with DFT.
            error_e_structures = np.ma.make_mask(energies_std >= e_error_threshold)

            model_forces_dict = row["forces"]
            forces_std = mdb_al_ut.get_model_forces_std(model_forces_dict)
            forces_std_norm = np.linalg.norm(forces_std, axis=2)
            forces_std_norm_max = np.amax(forces_std_norm, axis=1)

            # Any True value in this array is over the force error threshold
            # and must be sent to calculate with DFT.
            error_f_structures = np.ma.make_mask(
                forces_std_norm_max >= f_error_threshold
            )

            # Joining both error masks to get a single True/False array marking
            # structures to be computed with True
            error_all_structures = np.logical_or(
                np.logical_or(error_e_structures, error_f_structures),
                extrapolating_frames,
            )

            # If all values in error_all_structures are false, delete the main
            # structure from D0.
            flag_no_error_structs = np.all(error_all_structures == 0)

            # The index of the structure to delete will
            # be added to a list, which will be used as a mask to select
            # which structures to remove outside of the loop.
            if flag_no_error_structs:
                delete_indices.append(row["unique_id"])
            elif not flag_no_error_structs:
                # If there are some structures to submit, select some of them and
                # mark them for DFT.
                struct_arr = error_all_structures

                if isinstance(error_all_structures, np.bool_):
                    struct_arr = np.ones_like(energies_std)

                selected_high_error = np.nonzero(struct_arr)[0]

                dft_structures = [
                    row["trajectory"][int(struct)] for struct in selected_high_error
                ]

                # REMOVE: For testing purposes.
                # TESTING
                # dft_structures = [row["trajectory"][0]]
                # print('dft_structures: ', dft_structures)

                for calc_idx, dft_struct in enumerate(dft_structures):
                    builder = mdb_al_ut.get_dft_calc_builder(
                        dft_struct,
                        row,
                        calc_idx,
                        self.inputs.train_seed_group.value,
                        dft_settings=self.inputs.dft_settings.get_dict(),
                    )

                    # Submitting current calculation
                    future = self.submit(builder)
                    future.base.extras.set("mdb_calc_uuid", row["unique_id"])
                    future.base.extras.set("mdb_struct_type", row["mdb_struct_type"])
                    future.base.extras.set("struct_name", row["material_name"])
                    self.to_context(dft_struct_seed_calcs=append_(future))
                    submitted_dft_cnt += 1

                    if self.inputs.train_seed_group.value:
                        group = load_group(self.inputs.train_seed_group.value)
                        group.add_nodes(future)

        self.report(
            f"Committee decision: {submitted_dft_cnt} DFT / "
            f"{len(delete_indices)} delete."
        )

        # Deleting well represented structures from seed_gen_db (Ds), if any.
        if len(delete_indices) > 0:
            self.report(
                f"Deleting {len(delete_indices)} structures from seed"
                " generating DB (Ds)"
            )

            mdb_al_ut.remove_structs_from_seed_gen_db(
                self.inputs.seed_db_path, delete_indices
            )

        # If no structure is well represented, nothing will be deleted.
        else:
            self.report("Nothing removed from DB.")

    def return_seed_dft(self):
        """
        Gather and output DFT calculations for the current seed structures.

        This function collects DFT calculations for the structures in the current seed,
        which are then returned as outputs in the workchain using the namespace
        `dft_calcs`. A check is performed to determine if the results agree using
        MACE models, and this check also outputted to the workchain using the
        namespace `stop_md_seed_no_disagreement`.
        """
        try:
            dft_calcs = len(self.ctx.dft_struct_seed_calcs)
            self.report(f"Gathered {dft_calcs} DFT calculations.")

            return_list = mdb_al_ut.gather_dft_calcs(
                [node.uuid for node in self.ctx.dft_struct_seed_calcs]
            )

        except AttributeError:
            # self.ctx.dft_struct_seed_calcs = []
            return_list = List([])

        self.out("dft_calcs", return_list)  # list[dict]
        self.out(
            "stop_md_seed_no_disagreement",
            mdb_al_ut.check_md_seed_agreement(return_list),
        )


class ActiveLearningBaseWorkChain(BaseRestartWorkChain):
    _process_class = ActiveLearningWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.expose_inputs(
            ActiveLearningWorkChain,
            namespace="active_learning",
            exclude=[
                "current_md_seed_structs",
                "current_md_seed_structs_idx",
                "al_loop_iteration",
                "train_seed_group",
                "seed_gen_db",
                "seed_db_path",
                "training_db_path",
                "database_training",
            ],
        )

        spec.expose_outputs(
            ActiveLearningWorkChain,
            exclude=[
                "final_training_db",
                "final_model_file",
            ],
        )

        spec.outline(
            # Load the initial database (D_ini), that will be used as the
            # training database (Dt) without changing the original database.
            # Additionally, create a copy of the database (seed_gen_db, Ds),
            # this will be used to generate the MD seeds.
            cls.get_database,
            # Create inputs for workchains and initialize iterative counter
            cls.setup,
            # This part will loop to complete the process
            # It will loop `self.ctx.inputs.max_al_iterations` times.
            while_(cls.check_al_loop_conditions)(
                # Get random structures from Ds to generate the MD seed.
                cls.get_training_seed,
                # Run training seed
                cls.run_process,
                # Check for correct results
                # cls.inspect_process,
                # Get results from workchain
                cls.get_results_loop,
                # Update Ds and Di to include results from DFT.
                # Update the inputs for the next workchain.
                cls.add_dft_results_to_db,
                cls.get_al_loop_break_conditions,
            ),
            cls.results_final,
        )
        spec.output(
            "final_training_db",
            valid_type=SinglefileData,
        )
        spec.output("final_model_file", valid_type=SinglefileData)

    def get_database(self):
        """Loading initial database."""
        self.report("Reading database file...")

        # The training database (Dt) from which copies are made
        # for further processing. New structures will be added here.
        database_training = ase_read(
            filename=self.inputs.active_learning.init_db_path.value,
            format="extxyz",
            index=":",
        )

        # Create files for database_training and seed_gen_db
        results_dir_path = Path(self.inputs.active_learning.results_dir.value)
        if not results_dir_path.exists():
            results_dir_path.mkdir()

        final_db_path, curr_run_results_dir = mdb_al_ut.get_final_db_path(
            result_dir_path=results_dir_path,
            final_db_name=self.inputs.active_learning.final_db_name.value,
            node=self.node,
        )

        # A copy of the initial database, (Ds)
        # used specifically for generating MD seeds and running the MDs.
        # New structures will be added and well represented configs removed from here.
        self.ctx.seed_db_path = curr_run_results_dir / "seed_db.xyz"
        shutil.copy(
            self.inputs.active_learning.init_db_path.value, self.ctx.seed_db_path
        )

        self.ctx.training_db_path = final_db_path
        shutil.copy(
            self.inputs.active_learning.init_db_path.value, self.ctx.training_db_path
        )

        self.report(
            f"Loaded initial database containing {len(database_training)} structures."
        )

    def get_results_loop(self):
        """Attach the outputs specified in the spec from the last completed process."""
        node = self.ctx.children[self.ctx.iteration - 1]

        # TODO: Gather outputs manually, instead of using __attach_outputs
        self._attach_outputs(node)
        self.ctx.last_workchain_completed = node
        return None

    def add_dft_results_to_db(self):
        """
        Incorporate DFT calculation results into the training/seed generation databases.

        This method updates the training and seed generation databases with DFT
        calculation results. If any DFT calculations have been performed,
        their results are appended to both the training database and the seed
        generation database.
        """
        # Updating current training seed
        seed_gen_db = mdb_al_ut.load_database(self.ctx.seed_db_path)
        training_db = mdb_al_ut.load_database(self.ctx.training_db_path)
        # self.ctx.seed_gen_db = self.outputs["upd_seed_gen_db"]
        # self.ctx.inputs.seed_gen_db = self.outputs["upd_seed_gen_db"]

        last_wc = self.ctx.last_workchain_completed

        try:
            cnt_dft_calcs = len(last_wc.outputs["dft_calcs"])

        except KeyError:
            cnt_dft_calcs = 0

        if cnt_dft_calcs > 0:
            self.report(f"Adding {cnt_dft_calcs} DFT calculations to DB.")

            # Adding calculations to training database and seed_generation database
            for dft_calc in last_wc.outputs["dft_calcs"]:
                # Converting serialized structures to Atoms object.
                if isinstance(dft_calc, dict):
                    dft_calc = mdb_al_ut.aiida_serialized_ase_dict_to_atoms(dft_calc)

                seed_gen_db.append(dft_calc)
                training_db.append(dft_calc)

            # Updating final and seed database.
            self.report("Updating database files...")

            ase_write(
                filename=self.ctx.training_db_path,
                images=training_db,
                format="extxyz",
            )
            ase_write(
                filename=self.ctx.seed_db_path,
                images=seed_gen_db,
                format="extxyz",
            )

            self.report("Database files updated.")

        self.report(
            f"Iteration {self.ctx.iteration}: "
            f"seed_gen_db {len(seed_gen_db)}, "
            f"training_db: {len(training_db)} entries"
        )

    def get_al_loop_break_conditions(self):
        """
        Evaluate and set conditions to potentially break the active learning loop.

        This function checks for specific conditions that might warrant terminating the
        active learning (AL) loop early:
        - Gathers `stop_md_seed_no_disagreement` from the outputs of the inner workchain
          and stores it in the workchain's context. If this is True, the workchain will stop.
        - Checks whether all structures have been removed from the seed generation database
          (indicating no further candidates for evaluation). If this is True,
          the workchain will stop.

        The results of these checks are stored in the workflow's context.
        """
        # Sending seed disagreement flag to context
        self.ctx.stop_md_seed_no_disagreement = self.outputs[
            "stop_md_seed_no_disagreement"
        ]

        # Sending empty seed_gen_db flag to context
        seed_gen_db = mdb_al_ut.load_database(self.ctx.inputs.seed_db_path)
        if len(seed_gen_db) == 0:
            self.ctx.seed_gen_db_all_structs_removed = Bool(True)
        else:
            self.ctx.seed_gen_db_all_structs_removed = Bool(False)

    def setup(self):
        """Call BaseRestartWorkChain setup and create inputs dict in self.ctx.inputs.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain`
        to submit the process in the internal loop.
        """
        self.report("Starting Workchain setup.")
        super().setup()

        self.ctx.inputs = self.exposed_inputs(
            ActiveLearningWorkChain, "active_learning"
        )

        # Creating aiida group to store all calculations
        ctime = time.strftime("%Y%m%dT%H%M%S")

        # TESTING: Change this back to the string below
        seed_group = Group(label=f"remove_test_{ctime}")
        # seed_group = Group(label=f"train_md_seed_{ctime}")

        seed_group.store()
        self.ctx.inputs.train_seed_group = seed_group.uuid
        self.report(f"Created group: '{self.ctx.inputs.train_seed_group}'.")

        # Providing current iteration to children workchain.
        self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        # Setting conditionals to always run the first iteration of the
        # active learning loop.
        self.ctx.stop_md_seed_no_disagreement = Bool(False)
        self.ctx.seed_gen_db_all_structs_removed = Bool(False)

        # Adding database paths to inputs
        self.ctx.inputs.seed_db_path = str(self.ctx.seed_db_path)
        self.ctx.inputs.training_db_path = str(self.ctx.training_db_path)

        self.report("Workchain setup finished.")

    def check_al_loop_conditions(self) -> bool:
        """
        Evaluate conditions to determine whether to continue the active learning loop.

        This method assesses multiple conditions to decide if the active learning (AL)
        loop should continue. It considers whether the maximum number of iterations
        has been reached, and the following two criteria:
         - No disagreement among predictions for an entire MD seed. (if True: stop)
         - Depletion of the seed generation database. (if True: stop)

        The function updates the workchain context with the iteration status and
        generates reports based on the evaluation of these conditions.

        Returns
        -------
        bool
            A boolean value indicating whether the AL loop should continue. Returns
            `True` if conditions are met for another iteration; otherwise,
            returns `False`.

        Notes
        -----
        - The method uses `self.ctx.is_finished`, `self.ctx.iteration`,
        `self.inputs.active_learning.max_iterations.value`,
        `self.ctx.stop_md_seed_no_disagreement.value`,
        and `self.ctx.seed_gen_db_all_structs_removed.value`
        """
        max_iterations = self.inputs.active_learning.max_iterations.value

        # This will be True if the workchain still needs to be running due
        # to the number of iterations.
        iterations_status_ok = (
            not self.ctx.is_finished and self.ctx.iteration < max_iterations
        )
        # If either stop_md_seed_no_disagreement or seed_gen_db_all_structs_removed are
        # True, the loop must be stopped
        # This will be True while the AL loop needs to be repeated
        continue_loop_conditions = (
            not self.ctx.stop_md_seed_no_disagreement.value
            and not self.ctx.seed_gen_db_all_structs_removed.value
        )

        # This will be True if the workchain can be repeated.
        continue_cond = continue_loop_conditions and iterations_status_ok

        # In order to trigger the safeguard check, no disagreement must happen
        # in a single seed.
        if self.ctx.stop_md_seed_no_disagreement.value:
            # If the MD safeguard is enabled, run it and check its outputs.
            if self.inputs.active_learning.safeguard_enable:
                # Run the safeguard md
                self.run_md_safeguard()

                # Gather the md results and store them into the current
                # context, in `self.ctx.safeguard_results_df`
                self.gather_safeguard_results()

                # TESTING:
                print(self.ctx.safeguard_results_df)

                # If extrapolation check is enabled, check it for the safeguard MD
                # frames
                if self.inputs.active_learning.check_extrapolation:
                    self.get_descriptors_from_safeguard()

                # Check the safeguard MD results for disagreement.
                self.check_disagreement_safeguard()

            self.report("Stopping AL Loop as all predictions agree for a MD seed.")
        elif self.ctx.seed_gen_db_all_structs_removed.value:
            self.report(
                "Stopping AL Loop as seed generating database has been depleted."
            )
        else:
            if self.ctx.iteration != 0:
                self.report(
                    f"Proceeding with iteration-{self.ctx.iteration+1} of AL Loop "
                    "as stopping conditions not met."
                )
            self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        return continue_cond

    def run_md_safeguard(self):
        """
        Run MD safeguard for a structure using M0 to test the model readiness.

        This function performs a molecular dynamics simulation for a single structure
        either specified by the user or chooses a random structure within the current
        training seed, utilizing a predefined main model (M0).
        It generates and configures a MACE-LAMMPS calculation job, sets up the necessary
        input files and parameters, and submits these jobs for execution.
        Calculation nodes are stored and managed within the workflow's context for
        later retrieval and analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The function does not return a value but updates the workflow context with
            futures of the submitted MD simulation jobs, facilitating tracking and
            subsequent analysis of these simulations.

        Notes
        -----
        - This function assumes the availability of a trained MACE-LAMMPS potential file within
        the workflow's context.
        - Submitted calculation jobs are added to an AiiDA group for organization and
        are tagged with additional information to link them back to their respective
        positions in the database.
        """
        self.report("Running MD safeguard before ending AL loop...")

        # This string with the label used in the code setup.
        code_str = self.inputs.lammps_mace.get("code")
        builder = CalculationFactory("lammps.raw").get_builder()
        builder.code = load_code(code_str)

        # Getting the lammps potential file in a temporary folder
        with self.ctx.lammps_potential_file.as_path() as lmp_pot_path:
            lmp_pot_filename = Path(lmp_pot_path).name
            lmp_pot_path = str(lmp_pot_path)

            # Setting the trajectory to be retrieved and the
            # potential file to be copied into the calculation folder
            builder_settings = {
                "additional_retrieve_list": ["structure.lammpstrj"],
                "local_copy_list": [
                    (
                        self.ctx.lammps_potential_file.uuid,
                        lmp_pot_path,
                        lmp_pot_filename,
                    )
                ],
            }

            builder.settings = Dict(builder_settings)

            # Select structure (user-defined/random)
            sel_method = self.inputs.active.learning.structure_selection_method
            if sel_method == "user-defined":
                # Load the user selected structure
                curr_structure = ase_read(
                    self.inputs.active.learning.safeguard_user_defined_struct_path,
                    format="extxyz",
                )
            elif sel_method == "random":
                # Select a random structure from the database to use as the safeguard
                curr_structure = ...
                raise NotImplementedError(
                    "Safeguard random structure selection not implemented yet."
                )

            # for idx, curr_structure in enumerate(self.inputs.current_md_seed_structs):
            # Structures are stored as a dict in order to be json-serializable
            for key in ["pbc", "cell", "numbers", "positions", "forces"]:
                curr_structure[key] = np.array(curr_structure[key])

            curr_structure = Atoms.fromdict(curr_structure)

            # Converting structure to pymatgen
            curr_structure = pmg_ase.AseAtomsAdaptor.get_structure(curr_structure)
            struct_properties = curr_structure.properties

            # TODO: Add to TOML.
            # Preparing the MD calculation input using the parameters specified
            temp_val = self.inputs.active_learning.safeguard_temperature_K
            curr_input = self.gen_md_input(
                structure=curr_structure,
                potential_path=lmp_pot_filename,
                current_temp=temp_val,
                temp_coeff=self.inputs.active_learning.safeguard_max_temp_multiplier.value,
                timestep_val=self.inputs.active_learning.safeguard_timestep_duration_ps.value,
                num_timestep=self.inputs.active_learning.safeguard_num_steps.value,
            )

            script = SinglefileData(io.StringIO(curr_input))
            builder.script = script

            lammps_struct_str = LammpsData.from_structure(
                curr_structure, atom_style="atomic"
            ).get_str()

            data = SinglefileData(io.StringIO(lammps_struct_str))
            builder.files = {
                "data": data,
                "mace_potential": self.ctx.lammps_potential_file,
            }
            builder.filenames = {
                "data": "structure.lammps",
                "mace_potential": lmp_pot_filename,
            }

            # Loading metadata settings from workchain inputs
            builder.metadata = self.inputs.lammps_mace.get("metadata")
            builder.metadata.label = f"struct_safeguard_mace_lammps_md_{temp_val}_K"

            # Submitting current calculation
            future = self.submit(builder)

            # Writing extra information that helps associating the calculation
            # with its position on the database.
            for key, val in struct_properties.items():
                future.base.extras.set(key, val)

            if sel_method == "user-defined":
                future.base.extras.set("aiida_uuid", "safeguard")

            future.base.extras.set("index_in_db", "safeguard")
            future.base.extras.set("md_temperature", temp_val)

            # Telling the work chain to wait for the md to finish
            # before continuing the workflow.
            # We append the future to a list of workflows.
            self.to_context(md_safeguard=append_(future))

    def gather_safeguard_results(self):
        """
        Gather safeguard MD simulation results for the specified structure.

        This function collects the results from the MD simulation performed for
        the safeguard before stopping an active learning loop.
        It extracts trajectories, energies, and forces from the MD output and
        aggregates these into a structured format. The collected data is then
        organized into a pandas DataFrame, which is stored in the workflow's context.
        """
        self.report("Gathering safeguard MD results for the given structure...")
        new_rows = []

        # Gathering results for the single structure
        for workchain in self.ctx.md_safeguard:
            workchain_results = workchain.outputs.retrieved
            steps_E_F_arr = self.gather_energies_from_workchain(workchain_results)
            traj, forces = self.gather_traj_from_workchain(workchain_results)

            new_rows.append(
                {
                    "trajectory": traj,
                    "energy": {"m0": steps_E_F_arr[:, 1]},
                    "forces": {"m0": forces},
                    "al_step": self.inputs.al_loop_iteration.value,
                    "index_in_db": workchain.base.extras.all["index_in_db"],
                    "mdb_struct_type": workchain.base.extras.all["mdb_struct_type"],
                    "material_name": workchain.base.extras.all["struct_name"],
                    "unique_id": workchain.base.extras.all["aiida_uuid"],
                    "md_temperature": workchain.base.extras.all["md_temperature"],
                    "extrapolation": np.nan,
                }
            )

        # Creating a DataFrame with all the results.
        # This dataframe should only contain one row.
        self.ctx.safeguard_results_df = pd.DataFrame(new_rows)

    def get_descriptors_from_safeguard(self):
        # Getting descriptors for generated structures
        self.report("Getting descriptors for MD safeguard...")

        # Store all frames from the trajectory into a list
        for _, row in self.ctx.safeguard_results_df.iterrows():
            # all_frames_list = []
            curr_traj = row["trajectory"]
            traj_frames = []

            for frame in curr_traj:
                curr_frame: Atoms = AseAtomsAdaptor.get_atoms(frame)
                curr_frame.info["aiida_uuid"] = row["unique_id"]
                curr_frame.info["md_temperature"] = row["md_temperature"]
                traj_frames.append(curr_frame)

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename="-",
                    format="extxyz",
                    images=traj_frames,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            md_xyz_file = SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename="md_db.xyz",
            )

            # Prepare GetMACEDescriptorsCalculation
            mace_descr_calc = CalculationFactory("mace-get-descriptors")
            mace_builder = mace_descr_calc.get_builder()
            mace_builder.model_file = self.ctx.best_model_file
            mace_builder.mace_train_file_path = md_xyz_file
            descriptor_code_path = Path(f"{ROOT_DIR}/active_learning/mace_code")
            code = PortableCode(
                label="mace_get_descriptors",
                filepath_files=descriptor_code_path,
                filepath_executable="./mace_get_descriptors.py",
                # TODO: Add to TOML
                prepend_text="source /gpuscratch/psanz/mace/mace-venv/bin/activate",
            )
            mace_builder.code = code

            # TODO: Add to TOML
            mace_builder.metadata.options = {
                "resources": {
                    "parallel_env": "c128m1024ib_mpi_32slots",
                    "tot_num_mpiprocs": 4,
                },
                "queue_name": "c128m1024ibgpu4.q",
                "max_memory_kb": 102400000,
                "parser_name": "mace-descriptors-parser",
                "max_wallclock_seconds": 117280000,
                "withmpi": False,
                "custom_scheduler_commands": "#$ -l gpu=1",
            }
            mace_builder.metadata.label = (
                row["unique_id"][:8] + "_md_descriptors_" + f"{row['md_temperature']}_K"
            )
            mace_builder.metadata.computer = load_computer("tekla2-new-test")

            mace_builder.metadata.options.output_filename = (
                f"descriptors_{self.ctx.best_model_name}_iter"
                f"-{self.inputs.al_loop_iteration.value}"
            )

            future = self.submit(mace_builder)
            future.base.extras.set("unique_id", row["unique_id"])
            future.base.extras.set("md_temperature", row["md_temperature"])

            self.to_context(safeguard_descriptor_results=append_(future))

    def check_disagreement_safeguard(self):
        # Load safeguard MD results
        row = self.ctx.safeguard_results_df.iloc[0]

        # Load descriptor results
        if self.inputs.active_learning.check_extrapolation:
            for curr_calc in self.ctx.safeguard_descriptor_results:
                # [ ] Check if any of the frames results in extrapolation
                # return a n_frames long array of true/false values

                # [ ] Check for E/F disagreement with the models
                ...

    def get_training_seed(self):
        """
        Selects a random subset of structures from the seed generation database to
        create a MD seed for the active learning loop.

        This function calculates the number of structures to be included in the MD
        seed based on the specified fraction of the seed generation database's length.
        It then randomly selects and populates the training seed with these structures.

        Returns
        -------
            None. The function updates self.ctx.current_md_seed_structs with the
            selected structures.
        """
        self.report(
            f"Starting AL Loop iteration {self.ctx.iteration+1}/"
            f"{self.inputs.active_learning.max_iterations.value}..."
        )
        self.report("Getting MD seed...")
        self.ctx.inputs.metadata.description = (
            "Perform MD simulations, evaluate and refine ML models. "
            f"Step: {self.ctx.iteration+1}"
        )
        self.ctx.inputs.metadata.label = f"Step - {self.ctx.iteration+1}"

        seed_gen_db = mdb_al_ut.load_database(self.ctx.seed_db_path)
        # Getting length of the seed generating database
        db_length = len(seed_gen_db)

        # Defining the current seed size as a function of the intial seed size
        seed_size = int(self.ctx.inputs.seed_size_frac.value * db_length)

        # This should avoid tring to select more structures than available
        if seed_size > db_length:
            seed_size = db_length

        # For small databases or percentages, the number of structures might be 0
        # if this happens, make it 1.
        if seed_size == 0:
            seed_size = 1

        # Choosing structures at random to create the training seed
        selected_structs = np.random.choice(
            range(db_length),
            size=seed_size,
            replace=False,
        )

        self.ctx.inputs.current_md_seed_structs_idx = list(selected_structs)

        # The set of random structures selected from the seed generation
        # database to be used in training.
        self.ctx.current_md_seed_structs = []

        # Populating training seed with the selected random structures
        for idx in selected_structs:
            self.ctx.current_md_seed_structs.append(seed_gen_db[idx])

        self.report(
            f"Created MD seed with {seed_size}"
            f" structures ({self.ctx.inputs.seed_size_frac.value*100}% of init. size)."
        )
        # Adding current train seed to the context
        current_MD_seed_serialized = []
        for curr_s in self.ctx.current_md_seed_structs:
            curr_s = mdb_al_ut.serialize_ase(curr_s)
            current_MD_seed_serialized.append(curr_s)

        self.ctx.inputs.current_md_seed_structs = current_MD_seed_serialized

    def results_final(self):
        """
        Finalize the results at the end of the workchain.

        This method is responsible for preparing and returning the final training
        database at the conclusion of the workchain. It serializes the structures
        within the training database to a format compatible with AiiDA storage and
        subsequent processing. The serialized structures are then used to prepare
        the final training database, which is outputted from the workchain. This
        signifies the completion of the workchain and the availability of the
        processed training data for further use.
        """
        self.report("Returning final results...")

        # Storing final database as a SingleFileData object
        train_db = mdb_al_ut.prepare_output_final_training_db(
            training_db_path=self.ctx.inputs.training_db_path
        )

        self.out("final_training_db", train_db)
        self.report("Workchain completed!")
