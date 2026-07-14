"""AiiDA workchain for NNP active learning loops using MD."""

import io
import logging
import os
import pickle
import shutil
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import requests
from aiida import orm
from aiida.engine import (
    BaseRestartWorkChain,
    WorkChain,
    append_,
    if_,
    while_,
)
from aiida.plugins import CalculationFactory
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from atlas import ATL_ROOT_DIR
from atlas.active_learning import active_learning_utils as atl_al_ut
from atlas.active_learning.backends import get_backend
from atlas.core.code_utils import (
    ATL_THEME,
    ATLHighlighter,
    ATLRichHandler,
    LevelNameFilter,
    get_atl_version_info,
)
from atlas.workflows.aiida_utils import (
    get_or_create_containerized_code,
    get_or_create_portable_code,
    validate_kspacing,
)


def _build_compile_model_builder(
    workchain, model_node, code_settings, current_settings, device, backend_name
):
    """Build a ``CompileModelCalculation`` builder for one inference target.

    Shared by the child workchain (eval/seed-MD/committee) and the base
    workchain (safeguard). The compile runs on ``code_settings``' computer so
    the artifact matches the node where inference will happen.
    """
    builder = CalculationFactory('atl-compile-model').get_builder()
    builder.model = model_node
    builder.compile_settings = orm.Dict(
        dict={
            'backend': backend_name,
            'device': device,
            'mode': 'aotinductor',
            'target': 'ase',
        }
    )
    builder.metadata.computer = orm.load_computer(code_settings['metadata']['computer'])

    resources = code_settings['metadata']['options'].get('resources', {})
    num_threads = resources.get('num_cores_per_mpiproc', 2)

    code, prepend_text = atl_al_ut.return_code_from_settings(
        current_settings=current_settings,
        code_settings=code_settings,
        workchain=workchain,
        num_threads=num_threads,
        executable_name='atl_compile_model.py',
        code_path=Path(f'{ATL_ROOT_DIR}/active_learning/scripts'),
        portable_code_label='atl-compile-model',
        builder=builder,
    )
    builder.code = code
    builder.metadata.options = code_settings['metadata']['options']
    builder.metadata.options.parser_name = 'atl-compile-model-parser'
    builder.metadata.options.prepend_text = prepend_text
    return builder


class SimpleActiveLearningWorkChain(WorkChain):
    """
    WorkChain to run an active learning loop for a MLIP using MD simulations
    to generate training data.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.input('al_start_mode', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'atl_working_directory', valid_type=orm.Str, serializer=orm.to_aiida_type
        )
        spec.input('init_db_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('toml_file', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('log_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('final_db_name', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('debug_mode', valid_type=orm.Bool, serializer=orm.to_aiida_type)
        spec.input('enable_ntfysh', valid_type=orm.Bool, serializer=orm.to_aiida_type)
        spec.input(
            'ntfysh_topic',
            valid_type=(orm.Str, type(None)),
            serializer=orm.to_aiida_type,
            required=False,
        )
        spec.input('run_name', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'load_init_models',
            valid_type=(orm.List, type(None)),
            serializer=orm.to_aiida_type,
            default=None,
        )
        spec.input(
            'load_descriptor_calc',
            valid_type=(orm.Str, orm.Int, type(None)),
            serializer=orm.to_aiida_type,
            default=None,
        )
        spec.input(
            'load_md_calcs',
            valid_type=(orm.List, type(None)),
            serializer=orm.to_aiida_type,
            default=None,
        )
        spec.input('results_dir', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'current_seed_size', valid_type=orm.Int, serializer=orm.to_aiida_type
        )

        spec.input(
            'al_loop_iteration', valid_type=orm.Int, serializer=orm.to_aiida_type
        )
        spec.input('container_settings', valid_type=orm.Dict)
        spec.input(
            'seed_min_num_structs',
            valid_type=orm.Int,
            serializer=orm.to_aiida_type,
            required=False,
        )
        spec.input('seed_size_frac', valid_type=orm.Float, serializer=orm.to_aiida_type)
        spec.input(
            'seed_max_num_structs', valid_type=orm.Int, serializer=orm.to_aiida_type
        )
        spec.input(
            'seed_select_settings', valid_type=orm.Dict, serializer=orm.to_aiida_type
        )
        spec.input(
            'delete_seed_structs', valid_type=orm.Bool, serializer=orm.to_aiida_type
        )
        spec.input(
            'committee_num_models', valid_type=orm.Int, serializer=orm.to_aiida_type
        )
        spec.input(
            'model_acc_multiplier', valid_type=orm.Float, serializer=orm.to_aiida_type
        )
        spec.input(
            'descriptor_settings', valid_type=orm.Dict, serializer=orm.to_aiida_type
        )
        spec.input(
            'md_temperature_list_K', valid_type=orm.List, serializer=orm.to_aiida_type
        )
        spec.input('md_num_steps', valid_type=orm.Int, serializer=orm.to_aiida_type)
        spec.input(
            'md_max_temp_multiplier', valid_type=orm.Float, serializer=orm.to_aiida_type
        )
        spec.input(
            'md_filters',
            valid_type=(orm.Dict, type(None)),
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input(
            'md_timestep_duration_ps',
            valid_type=orm.Float,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'al_keep_struct_every_n_ps',
            valid_type=orm.Float,
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'current_md_seed_structs_path',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
        )
        spec.input('seed_db_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('training_db_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'test_db_file',
            valid_type=orm.SinglefileData,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input(
            'eval_test_db_settings', valid_type=orm.Dict, serializer=orm.to_aiida_type
        )
        spec.input(
            'test_db_eval_results',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
        )
        spec.input(
            'use_test_db',
            valid_type=orm.Bool,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'current_md_seed_structs_idx',
            valid_type=orm.List,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'train_seed_group',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'mace_train',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input(
            'mlip_train',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input('md_parameters', valid_type=orm.Dict)
        spec.input('dft_method', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'dft_calc_limit',
            valid_type=orm.Int,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input('dft_settings', valid_type=orm.Dict)
        spec.input('committee_eval', valid_type=orm.Dict, serializer=orm.to_aiida_type)
        spec.input(
            'check_extrapolation_type',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input(
            'gather_traj_cnt_lattice', valid_type=orm.Bool, serializer=orm.to_aiida_type
        )
        spec.input('use_kokkos', valid_type=orm.Bool, serializer=orm.to_aiida_type)

        # Data reduction specific inputs
        spec.input(
            'al_mode',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
            required=False,
            default=lambda: orm.Str('md'),
        )
        spec.input(
            'data_reduction_settings',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )

        spec.input(
            'stopping_criteria_settings',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )

        spec.outline(
            cls.step_setup,
            cls.train_mlip_model,
            cls.get_mlip_train_output,
            # Optionally pre-compile the model(s) once per inference target
            # (no-op for backends that need no compilation, e.g. MACE).
            cls.compile_models_for_inference,
            cls.gather_compiled_models,
            # If enabled, run evaluation of test database using M0 model.
            if_(cls.is_test_db_evaluation_enabled)(
                cls.perform_test_db_evaluation,
                cls.get_test_db_results,
            ),
            # TODO: Add here a way to filters structures based on error threshold
            # Evaluate training database error and mark structures for DFT or removal
            # For data reduction mode: select additional structures for training
            if_(cls.should_select_data_reduction_structures)(
                cls.select_data_reduction_structures,
            ),
            # This part of the workflow is only executed if the extrapolation
            # check is enabled.
            # It will get the descriptors for the entire database and use
            # the concave hull as the extrapolation mechanism.
            if_(cls.check_extrapolation_enabled)(
                # Generate descriptors for the current dataset
                # and apply dimensionality reduction if specified.
                # If advanced extrapolation is enabled, the latent space
                # will be used to generate the concave hull.
                cls.gen_descriptors_and_concave_hull,
                cls.get_descriptor_results,
            ),
            # All of the structures in the seed will be run using the MD
            # code selected, using the main model (M0).
            # Then, the trajectories produced with the M0 model will be evaluated
            # using the committee models.
            # Structures and energy predictions will be gathered and prepared
            # According to the difference in error between the models either:
            # The original structure will be removed from D0, or
            # The problematic structure will be calcualated using DFT
            cls.run_md_seed,
            # This uses extrapolation (if enabled), and an energy/force check.
            cls.send_calc_or_remove_structures,
            # Return the generated DFT calculations to the workchain as an output
            cls.return_seed_dft_and_model,
            # TODO: If high error (define this) is found on a training seed,
            # do not change seed until the error is decreased
            # cls.choose_next_seed,
        )
        spec.output('dft_calcs_path', valid_type=orm.Str, required=False)
        spec.output('m0_model_file', valid_type=orm.SinglefileData)
        spec.output('m0_rmse_e', valid_type=orm.Float)
        spec.output('m0_rmse_f', valid_type=orm.Float)
        spec.output('stop_md_seed_no_disagreement', valid_type=orm.Bool)
        spec.output('test_db_eval_results', valid_type=orm.Dict)
        spec.output('test_db_eval_plot', valid_type=orm.SinglefileData)
        spec.output('stop_al_loop_error', valid_type=orm.Bool)

        spec.exit_code(
            420, 'ERROR_SCHEDULER_MLIP', 'error when submitting a MLIP calculation.'
        )
        spec.exit_code(
            421,
            'ERROR_DESCRIPTOR_CALCULATION',
            'Descriptor calculation ({node_id}) could not finish successfully.',
        )

    def set_step_logger(self):
        self.report('Performing secondary logger setup...')

        # Get log path
        log_path = self.inputs.log_path.value

        # Getting aiida logger
        aiida_logger = logging.getLogger('aiida')

        # Remove aiida's own cli handler to avoid duplicate logs
        # Iterate over a copy of the list to modify it safely
        for handler in aiida_logger.handlers[:]:
            if handler.name == 'cli':
                aiida_logger.removeHandler(handler)

        # Set the PARENT logger to INFO to silence framework debug messages
        aiida_logger.setLevel(15)

        # We only want to see our custom debug messages and our reports.
        log_filter = LevelNameFilter(
            levels_to_keep=[
                'ATL_DEBUG',
                'REPORT',
                'SUCCESS',
                'INFO',
                'WARNING',
                'ERROR',
                'DEBUG',
            ]
        )

        curr_logger_handlers = aiida_logger.handlers

        if not any(
            [handl.name == 'atl_rich_handler' for handl in curr_logger_handlers]
        ):
            console = Console(color_system='truecolor', theme=ATL_THEME)
            # Create a file handler
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(1)
            file_handler.set_name('atl_file_handler')
            file_handler.addFilter(log_filter)

            # Create a formatter and set it for the file handler
            file_formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] - %(message)s',
                datefmt='%m/%d/%y %H:%M:%S',
            )
            file_handler.setFormatter(file_formatter)
            aiida_logger.addHandler(file_handler)

        if not any(
            [handl.name == 'atl_rich_handler' for handl in curr_logger_handlers]
        ):
            # Adding console logger
            console = Console(
                color_system='truecolor', theme=ATL_THEME, highlighter=ATLHighlighter()
            )
            ch = ATLRichHandler(
                markup=True,
                show_path=False,
                log_time_format='[%d/%m/%y %H:%M:%S]',
                omit_repeated_times=False,
                console=console,
            )
            ch.set_name('atl_rich_handler')
            ch.setLevel(23)
            formatter_con = logging.Formatter('%(message)s')
            ch.setFormatter(formatter_con)
            aiida_logger.addHandler(ch)

    def step_setup(self):
        # Initialize error tracking variable to False, since the loop hasn't had
        # any chances to get into an error state
        self.ctx.stop_al_loop_error = orm.Bool(False)

        self.node.base.extras.set(
            'atl_working_directory', self.inputs.atl_working_directory.value
        )
        self.node.base.extras.set('atl_version', str(get_atl_version_info()[0]))

        # Setting up secondary logger for the current step.
        # This is only necessary for when runs are submited to the daemon,
        # instead of being run directly. In that case, the logger is not
        # initialized again, or messages appear doubled.
        if not self.inputs.debug_mode:
            self.set_step_logger()
            self.logger.debug('Secondary logger setup complete.')

        # If notifications are enabled, send start message
        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'Starting Active Learning Iteration '
                    f'{self.inputs.al_loop_iteration.value + 1}'
                    f' - {self.node.pk}:'
                    f' {self.inputs.run_name.value}'
                ),
            )

        try:
            import torch

            self.ctx.cuda_available = torch.cuda.is_available()
            if self.ctx.cuda_available:
                self.report(f'CUDA available: {self.ctx.cuda_available}')
                self.logger.debug(
                    f'Default device properties: '
                    f'{torch.cuda.get_device_properties(torch.cuda.current_device())}'
                )
            else:
                self.report('CUDA not available, running on CPU only.')
        except ImportError:
            self.logger.warning('Unable to import torch package. Assuming CPU only.')
            self.ctx.cuda_available = False

        # Initialize MLIP backend from TOML settings (defaults to 'mace')
        toml_path = Path(self.inputs.toml_file.value)
        if toml_path.exists():
            toml_settings = atl_al_ut.read_toml_settings(self.inputs.toml_file.value)
            backend_name = toml_settings.get('mlip', {}).get('training_backend', 'mace')
        else:
            backend_name = 'mace'

        self.ctx.mlip_backend_name = backend_name
        self.ctx.mlip_backend = get_backend(backend_name)
        self.ctx.mlip_train_settings = (
            self.inputs.mlip_train
            if self.inputs.mlip_train is not None
            else self.inputs.mace_train
        )
        self.report(f'Using MLIP backend: {backend_name}')

    def should_select_data_reduction_structures(self):
        """Check if we should select additional structures for data reduction mode."""
        return self.inputs.al_mode.value == 'data_reduction'

    def is_test_db_evaluation_enabled(self):
        """Check if test database evaluation is enabled."""
        return self.inputs.use_test_db

    def select_data_reduction_structures(self):
        """
        Select additional structures from the large database for training.

        This method is only called in data reduction mode and after the first iteration.
        It selects structures based on the iterative_selection_method and adds them
        to the training database while removing them from the seed database.
        """
        self.report('Selecting additional structures for data reduction...')

        # Get settings
        data_reduction_settings = self.inputs.data_reduction_settings.get_dict()
        structures_per_iteration = data_reduction_settings.get(
            'structures_per_iteration', 50
        )
        iterative_selection_method = data_reduction_settings.get(
            'iterative_selection_method', 'uncertainty'
        )

        # Load current databases
        seed_database = atl_al_ut.load_database(self.ctx.seed_db_path)
        training_database = atl_al_ut.load_database(self.ctx.training_db_path)

        if len(seed_database) == 0:
            msg = (
                'No more structures available in the large database. '
                'Stopping structure selection.'
            )
            self.report(msg)
            return

        # Prepare for structure selection
        descriptor_settings = None
        model_files = None

        if iterative_selection_method in ['fps', 'uncertainty']:
            descriptor_settings = self.inputs.descriptor_settings.get_dict()

        if iterative_selection_method == 'uncertainty':
            # Get committee model files from context
            if hasattr(self.ctx, 'commitee_models_tupl_name_uuid'):
                model_files = []
                for _, uuid_str in self.ctx.commitee_models_tupl_name_uuid:
                    # Load the model file from the calculation node
                    calc_node = orm.load_node(uuid_str)
                    if hasattr(calc_node.outputs, 'model_file'):
                        model_files.append(calc_node.outputs.model_file.filepath)

                if not model_files:
                    msg = (
                        'Warning: No model files found for uncertainty calculation. '
                        'Falling back to random selection.'
                    )
                    self.report(msg)
                    iterative_selection_method = 'random'
            else:
                msg = (
                    'Warning: No committee models found. '
                    'Falling back to random selection.'
                )
                self.report(msg)
                iterative_selection_method = 'random'

        # Limit selection to available structures
        n_to_select = min(structures_per_iteration, len(seed_database))

        # Select structures
        selected_structures = atl_al_ut.select_structures_data_reduction(
            database=seed_database,
            n_structures=n_to_select,
            selection_method=iterative_selection_method,
            descriptor_settings=descriptor_settings,
            model_files=model_files,
        )

        # Remove selected structures from seed database
        selected_ids = {s.info['atl_id'] for s in selected_structures}
        remaining_seed_database = [
            s for s in seed_database if s.info['atl_id'] not in selected_ids
        ]

        # Add selected structures to training database
        training_database.extend(selected_structures)

        # Update databases on disk
        ase_write(
            filename=self.ctx.seed_db_path,
            format='extxyz',
            images=remaining_seed_database,
        )
        ase_write(
            filename=self.ctx.training_db_path,
            format='extxyz',
            images=training_database,
        )

        msg = (
            f'Selected {len(selected_structures)} additional structures using '
            f'{iterative_selection_method} method. Remaining in large database: '
            f'{len(remaining_seed_database)}, Total in training database: '
            f'{len(training_database)}'
        )
        self.report(msg)

    def train_mlip_model(self):
        """Submit MLIP training calculations using the active backend.

        Generates a training database, then submits N training calculations
        (one per committee model). Uses PortableCode by default; falls back
        to a registered AiiDA Code if the ``code`` key is present in the
        training settings.
        """
        self.report('Generating new training database file.')

        updated_path, _ = atl_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=atl_al_ut.process_call_root(orm.load_node(self.uuid)),
        )

        database_training = atl_al_ut.load_database(self.inputs.training_db_path.value)

        self.ctx.mlip_backend.prepare_training_data(
            path=updated_path,
            structure_list=database_training,
        )

        self.ctx.resume_mode = self.inputs.al_start_mode.value == 'resume'

        if (
            self.inputs.load_init_models and self.inputs.al_loop_iteration.value == 0
        ) or (
            self.inputs.load_init_models
            and self.ctx.resume_mode
            and not hasattr(self.ctx, 'loaded_init_models')
        ):
            self.report(
                'Loading models from nodes: '
                f"'{self.inputs.load_init_models.get_list()}'."
            )
            return
        else:
            self.report(
                f'Training {self.inputs.committee_num_models.value} models using '
                'current training dataset.'
            )

        calc_count = 0
        train_settings_dict = self.ctx.mlip_train_settings.get_dict()

        containerized = False
        container_dict = self.inputs.container_settings.get_dict()
        if container_dict.get('use_container'):
            containerized = container_dict.get('use_container', False)
        if self.ctx.mlip_train_settings.get('ignore_container') is True:
            containerized = False

        use_portable_code = 'code' not in train_settings_dict
        sched_options = train_settings_dict['metadata'].get('options')

        train_file_path, _ = atl_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=self.node,
        )

        for _ in range(self.inputs.committee_num_models.value):
            model_name = atl_al_ut.generate_model_name()

            if use_portable_code:
                train_calcjob = CalculationFactory('atl-train-mlip')
                builder = train_calcjob.get_builder()

                self.ctx.mlip_backend.prepare_builder(
                    builder=builder,
                    settings_dict=train_settings_dict.get('train_settings', {}),
                    train_data_path=str(train_file_path),
                    model_name=model_name,
                    iteration=self.inputs.al_loop_iteration.value,
                    db_size=len(database_training),
                    containerized=containerized,
                )

                train_cfg = train_settings_dict.get('train_settings', {})
                if self.ctx.mlip_backend_name != 'mace':
                    mace_only = []
                    if train_cfg.get('foundation_model'):
                        mace_only.append('foundation_model')
                    if train_cfg.get('multihead_finetuning'):
                        mace_only.append('multihead_finetuning')
                    if mace_only:
                        self.report(
                            f'WARNING: {", ".join(mace_only)} '
                            f'configured but backend is '
                            f'"{self.ctx.mlip_backend_name}" '
                            f'(MACE-only features, will be ignored)'
                        )

                if containerized:
                    image_name = container_dict.get('image_name', '')
                    engine_command = container_dict.get('engine_command', '')
                    computer = orm.load_computer(
                        train_settings_dict.get('computer', None)
                    )
                    code = get_or_create_containerized_code(
                        label='atl-train-mlip',
                        computer=computer,
                        image_name=image_name,
                        engine_command=engine_command,
                        filepath_executable='atl_train_mlip',
                    )
                    container_prepend = container_dict.get('prepend_text', '')
                else:
                    computer = orm.load_computer(
                        train_settings_dict.get('computer', None)
                    )
                    code = get_or_create_portable_code(
                        label='atl-train-mlip',
                        filepath_files=Path(f'{ATL_ROOT_DIR}/active_learning/scripts'),
                        filepath_executable='atl_train_mlip.py',
                    )
                    container_prepend = ''

                num_threads = sched_options.get('resources', {}).get(
                    'num_cores_per_mpiproc', os.cpu_count()
                )

                builder.code = code
                builder.metadata.computer = computer
                builder.metadata.options = sched_options
                builder.metadata.options.parser_name = 'atl-train-mlip-parser'

                # Pass environment setup dynamically so TOML changes take
                # effect without recreating the code node.
                builder.metadata.options.prepend_text = (
                    train_settings_dict.get('metadata', {})
                    .get('options', {})
                    .get('prepend_text', '')
                    + '\n'
                    + container_prepend
                    + f'\nexport OMP_NUM_THREADS={num_threads}'
                )
            else:
                # Legacy path: backend-specific CalcJob + registered Code
                mace_train_settings = atl_al_ut.update_mace_train_settings_dict(
                    settings_dict=train_settings_dict.get('train_settings'),
                    train_data_path=str(updated_path),
                    curr_model=model_name,
                    curr_iter=self.inputs.al_loop_iteration.value,
                    db_size=len(database_training),
                    containerized=containerized,
                )

                train_calcjob = CalculationFactory(
                    self.ctx.mlip_backend.calcjob_entry_point
                )
                builder = train_calcjob.get_builder()
                builder.multihead_finetuning = self.ctx.mlip_train_settings.get(
                    'multihead_finetuning', False
                )
                builder.model_name = model_name
                builder.mace_settings_dict = orm.Dict(mace_train_settings)
                builder.use_container = orm.Bool(containerized)
                builder.mace_train_file_path = str(train_file_path)

                code_str = train_settings_dict['code']
                code = orm.load_code(code_str)
                computer = code.computer

                builder.code = code
                builder.metadata.options = sched_options
                builder.metadata.options.parser_name = (
                    self.ctx.mlip_backend.parser_entry_point
                )

            builder.metadata.options.output_filename = (
                f'train_{model_name}_iter-{self.inputs.al_loop_iteration.value}'
            )
            builder.metadata.label = model_name

            if not hasattr(builder.metadata.options, 'prepend_text'):
                builder.metadata.options.prepend_text = (
                    train_settings_dict.get('metadata', {})
                    .get('options', {})
                    .get('prepend_text', '')
                )

            calc_limit = computer.metadata.get('atl_calc_limit', 0)
            if calc_limit != 0:
                atl_al_ut.aiida_wait_submit(
                    builder=builder,
                    computer=computer,
                    calc_count=calc_count,
                )

            future = self.submit(builder)
            self.report(f'Submitted training calculation {future.pk}.')
            self.to_context(mlip_training_results=append_(future))

    def get_mlip_train_output(self):
        """Retrieve and process MLIP training output for model selection.

        Evaluates training results to select the best model based on a
        weighted sum of RMSE for energy and forces. Creates a
        LAMMPS-compatible potential and updates context with model info.
        """
        # Get the current iteration and resume mode
        curr_iter = self.inputs.al_loop_iteration.value

        # Ensure that models are loaded only for the first resumed step
        if self.inputs.load_init_models:
            if curr_iter == 0 or (
                self.ctx.resume_mode and not hasattr(self.ctx, 'loaded_init_models')
            ):
                mlip_training_results = [
                    orm.load_node(node) for node in self.inputs.load_init_models
                ]
                # Mark that models have been loaded
                self.ctx.loaded_init_models = True
            else:
                mlip_training_results = self.ctx.mlip_training_results
        else:
            mlip_training_results = self.ctx.mlip_training_results

        model_name_list = []
        weighted_E_F_sum_list = []

        # Iterate over all models and get weighted sum of E and F
        for calc in mlip_training_results:
            # Loading calculation node
            curr_calc: orm.CalcJobNode = orm.load_node(calc.uuid)

            # Skipping model if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                continue

            # Skipping model if RMSE values are not present
            if not hasattr(curr_calc.outputs, 'm_rmse_e') or not hasattr(
                curr_calc.outputs, 'm_rmse_f'
            ):
                continue

            # Getting model name
            model_name_list.append(curr_calc.inputs.model_name.value)

            # Adding E + F multiplied by a weight value, in order to consider
            # forces when deciding which model to keep
            force_weight = self.ctx.mlip_train_settings.get(
                'result_force_weight',
                0.1,
            )
            weighted_E_F_sum = curr_calc.outputs.m_rmse_e.value + (
                force_weight * curr_calc.outputs.m_rmse_f
            )
            weighted_E_F_sum_list.append(weighted_E_F_sum)

        # Get the most accurate model name during the validation by checking
        # the lowest E+F*weight.
        # In case no valid models were found, raise an error, since we need
        # at least one valid model to act as sampler model for the active learning
        # to work.
        try:
            self.ctx.best_model_name = model_name_list[np.argmin(weighted_E_F_sum_list)]
        except ValueError as e:
            self.ctx.stop_al_loop_error = orm.Bool(True)
            raise ChildProcessError(
                'No valid MLIP models were found after training! '
                f'Check for issues in the training step. '
                f'Calculation PKs: \n{mlip_training_results}'
            ) from e

        commitee_models_tupl_name_uuid = []
        for calc in mlip_training_results:
            # Loading calculation node
            curr_calc: orm.CalcJobNode = orm.load_node(calc.uuid)

            # Skipping model if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                self.report(
                    f'Skipping MLIP model that finished with errors (pk: {calc.pk}).'
                )
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

                # Storing best model file in the context
                self.ctx.best_model_file = model_file

                # Adding best model name as an extra to the model file
                self.ctx.best_model_file.base.extras.set(
                    'model_name', self.ctx.best_model_name
                )

                # Convert model to LAMMPS compatible format
                # and return it to workchain context
                if hasattr(self.ctx, 'cuda_available') and self.ctx.cuda_available:
                    lammps_potential = self.ctx.mlip_backend.create_lammps_potential(
                        model_file
                    )
                    if lammps_potential is not None:
                        self.ctx.lammps_potential_file = lammps_potential
                else:
                    self.logger.warning(
                        'CUDA not available, skipping LAMMPS potential file creation.'
                    )

                self.report(
                    f"Best model of current step '{model_name}' ({calc.pk}) as M0 - "
                    f'RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV/Å'
                )
                self.out('m0_model_file', model_file)
                self.out('m0_rmse_e', self.ctx.m0_rmse_e)
                self.out('m0_rmse_f', self.ctx.m0_rmse_f)
            else:
                self.report(
                    f"Trained committee model '{model_name}' ({calc.pk}) - "
                    f'RMSE E: {curr_calc.outputs.m_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {curr_calc.outputs.m_rmse_f.value:.3f} meV/Å'
                )
                commitee_models_tupl_name_uuid.append((model_name, model_file.uuid))

        # Sending committee model paths to current context
        self.ctx.commitee_models_tupl_name_uuid = commitee_models_tupl_name_uuid

        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'Committee model training complete for step '
                    f'{self.inputs.al_loop_iteration.value + 1}'
                    f' - {self.node.pk}.\n\n'
                    f"Best model of current step: '{model_name}' ({calc.pk}) as M0 - "
                    f'RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV/Å'
                ),
            )

    def compile_models_for_inference(self):
        """Optionally pre-compile trained model(s) for each inference target.

        No-op unless the MLIP backend implements ``MLIPModelCompiler`` (e.g.
        Allegro). Otherwise, for each distinct ``(computer, device, model)`` an
        enabled inference consumer will use, submit one compile CalcJob on that
        computer so the artifact matches the node's device/toolchain. Results
        are collected in :meth:`gather_compiled_models`. Purely an optimization:
        consumers fall back to on-demand compilation if an artifact is missing.
        """
        from atlas.active_learning.backends._base import MLIPModelCompiler

        self.ctx.compiled_models = {}
        self.ctx.compile_target_map = {}

        backend = self.ctx.mlip_backend
        if not isinstance(backend, MLIPModelCompiler):
            self.report(
                f"Backend '{self.ctx.mlip_backend_name}' needs no model "
                'compilation; skipping precompile step.'
            )
            return

        current_settings = atl_al_ut.read_toml_settings(
            settings_file=self.inputs.toml_file.value
        )
        if not current_settings.get('mlip', {}).get('precompile_for_inference', True):
            self.report('precompile_for_inference disabled; skipping.')
            return

        targets = self._enumerate_inference_targets(current_settings)

        submitted_keys = set()
        for model_node, code_settings, device in targets:
            computer_label = code_settings['metadata']['computer']
            key = (computer_label, device, model_node.uuid)
            if key in submitted_keys:
                continue
            submitted_keys.add(key)
            future = self._submit_compile_calc(
                model_node=model_node,
                code_settings=code_settings,
                current_settings=current_settings,
                device=device,
            )
            self.ctx.compile_target_map[future.uuid] = key
            self.to_context(compile_calcs=append_(future))
            self.report(
                f'Submitted model-compile calc ({future.pk}) for '
                f'{computer_label} / device={device}.'
            )

    def gather_compiled_models(self):
        """Collect compiled artifacts from the compile calcs into the cache."""
        for calc in getattr(self.ctx, 'compile_calcs', []):
            key = self.ctx.compile_target_map.get(calc.uuid)
            if key is None:
                continue
            if calc.is_finished_ok and 'compiled_model' in calc.outputs:
                self.ctx.compiled_models[key] = calc.outputs.compiled_model
                self.report(
                    f'Compiled model ready for {key[0]} / device={key[1]} '
                    f'(calc {calc.pk}).'
                )
            else:
                self.report(
                    f'WARNING: model compile for {key[0]} / device={key[1]} did '
                    f'not succeed (calc {calc.pk}); inference will compile '
                    'on demand as a fallback.'
                )

    def _enumerate_inference_targets(self, current_settings):
        """Return ``(model_node, code_settings, device)`` tuples to precompile.

        One entry per (enabled consumer, model) pair; deduplicated by
        ``(computer, device, model_uuid)`` in the caller.
        """
        targets = []

        # Test-DB eval — uses the best model (M0) on the [test_db] computer.
        if self.is_test_db_evaluation_enabled():
            if current_settings.get('test_db'):
                test_db_settings = current_settings['test_db']
            else:
                test_db_settings = self.inputs.eval_test_db_settings.get_dict()
            device = test_db_settings.get('model_settings', {}).get('device', 'cpu')
            targets.append((self.ctx.best_model_file, test_db_settings, device))

        # Seed MD — the md node runs MD with M0 and the committee E/F check with
        # every committee model, so all committee members are used there.
        md_settings = current_settings.get('md', {})
        if md_settings.get('metadata', {}).get('computer'):
            md_device = md_settings.get('parameters', {}).get('device', 'cpu')
            md_models = [self.ctx.best_model_file]
            for _name, uuid_str in getattr(
                self.ctx, 'commitee_models_tupl_name_uuid', []
            ):
                md_models.append(orm.load_node(uuid_str))
            for model_node in md_models:
                targets.append((model_node, md_settings, md_device))

        return targets

    def _submit_compile_calc(self, model_node, code_settings, current_settings, device):
        """Build and submit a CompileModelCalculation on the target computer."""
        builder = _build_compile_model_builder(
            workchain=self,
            model_node=model_node,
            code_settings=code_settings,
            current_settings=current_settings,
            device=device,
            backend_name=self.ctx.mlip_backend_name,
        )
        return self.submit(builder)

    def _inference_model_for(self, model_node, computer_label, device):
        """Return the compiled artifact for a target if cached, else the raw model."""
        key = (computer_label, device, model_node.uuid)
        return getattr(self.ctx, 'compiled_models', {}).get(key, model_node)

    def perform_test_db_evaluation(self):
        self.report('Preparing test database evaluation...')

        # Get builder for evaluation calculation
        eval_calc = CalculationFactory('atl-eval-test')
        eval_calc_builder = eval_calc.get_builder()

        # Passing inputs and metadata
        eval_calc_builder.test_database = self.inputs.test_db_file
        eval_calc_builder.settings_file_path = self.inputs.toml_file
        eval_calc_builder.test_db_eval_results = self.inputs.test_db_eval_results
        eval_calc_builder.current_iteration = orm.Int(self.inputs.al_loop_iteration)

        # Loading the settings file again to get updated settings
        settings_path = Path(self.inputs.toml_file.value)
        if settings_path.exists:
            current_settings = atl_al_ut.read_toml_settings(
                settings_file=self.inputs.toml_file.value
            )
            self.logger.debug(
                f'Reloaded settings from file: {self.inputs.toml_file.value}'
            )
        else:
            current_settings = {}

        if current_settings.get('test_db'):
            eval_test_db_settings = current_settings.get('test_db', {})
        else:
            eval_test_db_settings = self.inputs.eval_test_db_settings.get_dict()

        # Ship a pre-compiled artifact for this target if the compile step
        # produced one, else the raw best model (unchanged for MACE).
        eval_device = eval_test_db_settings.get('model_settings', {}).get(
            'device', 'cpu'
        )
        eval_calc_builder.sampler_model = self._inference_model_for(
            self.ctx.best_model_file,
            eval_test_db_settings['metadata']['computer'],
            eval_device,
        )

        # Loading metadata settings
        computer = orm.load_computer(eval_test_db_settings['metadata']['computer'])
        eval_calc_builder.metadata.computer = computer

        # Getting resources dict
        resources_dict = eval_test_db_settings['metadata']['options']['resources']
        num_threads = resources_dict.get('num_cores_per_mpiproc', 2)

        # Prepare and return correct code
        test_db_eval_code_path = Path(f'{ATL_ROOT_DIR}/active_learning/eval_test_db')
        code, prepend_text = atl_al_ut.return_code_from_settings(
            current_settings=current_settings,
            code_settings=eval_test_db_settings,
            workchain=self,
            num_threads=num_threads,
            executable_name='eval_test_db.py',
            code_path=test_db_eval_code_path,
            portable_code_label='atl-eval-test',
            builder=eval_calc_builder,
        )
        eval_calc_builder.code = code

        # Loading AiiDA settings
        mlip_eval_settings_dict = eval_test_db_settings['metadata']['options']

        # Load scheduler and resources options
        eval_calc_builder.metadata.options = mlip_eval_settings_dict
        eval_calc_builder.metadata.options.parser_name = 'atl-eval-test-parser'
        eval_calc_builder.metadata.options.prepend_text = prepend_text

        # Submit evaluation calculation
        future = self.submit(eval_calc_builder)
        self.report(f'Submitted test database evaluation calculation: {future.pk}.')
        self.to_context(test_db_results=append_(future))

    def get_test_db_results(self):
        """Retrieve and process test database evaluation results."""
        # Get test db results
        test_db_result = self.ctx.test_db_results[0]

        # Parse errors from the calculation and update test result dict
        rmse_e: float = test_db_result.outputs.rmse_e.value
        rmse_f: float = test_db_result.outputs.rmse_f.value
        mae_e: float = test_db_result.outputs.mae_e.value
        mae_f: float = test_db_result.outputs.mae_f.value

        test_db_dict = {}
        curr_iter = self.inputs.al_loop_iteration.value
        test_db_dict[f'step_{curr_iter}'] = {
            'rmse_e': rmse_e,
            'rmse_f': rmse_f,
            'mae_e': mae_e,
            'mae_f': mae_f,
        }

        # Get plot SinglefileData
        test_db_plot: orm.SinglefileData = test_db_result.outputs.eval_plot

        # Updating results dictionary
        self.ctx.test_db_eval_results = orm.Dict(test_db_dict)

        self.report(
            f'Test database evaluation complete for iteration '
            f'{self.inputs.al_loop_iteration.value} - RMSE E: {rmse_e:.5e} meV/at, '
            f' RMSE F: {rmse_f:.5e} meV/Å'
            f' MAE E: {mae_e:.5e} meV/at, '
            f' MAE F: {mae_f:.5e} meV/Å'
        )

        # Returning results to workchain
        self.out('test_db_eval_plot', test_db_plot)
        self.out('test_db_eval_results', self.ctx.test_db_eval_results)

    def check_extrapolation_enabled(self):
        """Check if the extrapolation check is enabled."""
        if self.inputs.check_extrapolation_type.value in [None, 'none', 'disabled']:
            extrapolation_status = False

            # Loading descriptor min and max into context as empty arrays
            self.ctx.descriptors_max_array = orm.ArrayData(np.array([]))
            self.ctx.descriptors_min_array = orm.ArrayData(np.array([]))
        else:
            extrapolation_status = True

        self.report(
            'Extrapolation check type is '
            f"'{self.inputs.check_extrapolation_type.value}', "
            'therefore extrapolation is set as: '
            f'{extrapolation_status}'
        )
        return extrapolation_status

    def gen_descriptors_and_concave_hull(self):
        self.report('Preparing descriptors calculation...')

        # Stop the calculation if descriptor calc must be loaded
        if (
            self.inputs.load_descriptor_calc
            and self.inputs.al_loop_iteration.value == 0
            and self.inputs.load_descriptor_calc not in ['', "''", '""']
        ) or (
            self.inputs.load_descriptor_calc
            and self.ctx.resume_mode
            and not hasattr(self.ctx, 'loaded_descriptor_calc')
            and self.inputs.load_descriptor_calc not in ['', "''", '""']
        ):
            self.report(
                'Loading descriptor calculation from node: '
                f"'{self.inputs.load_descriptor_calc.value}'."
            )
            return

        # Get builder for training calculation
        desc_calc = CalculationFactory('atl-descriptors-combined')
        desc_builder = desc_calc.get_builder()

        # Add current iteration to the process label
        desc_builder.metadata.label = (
            f'descriptors-combined_iter-{self.inputs.al_loop_iteration.value}'
        )

        # Getting the best model file
        desc_builder.best_model = self.ctx.best_model_file

        # Getting settings file path
        settings_file_pth = self.inputs.toml_file

        # Loading the settings file again to get updated settings
        settings_path = Path(self.inputs.toml_file.value)
        if settings_path.exists:
            current_settings = atl_al_ut.read_toml_settings(
                settings_file=self.inputs.toml_file.value
            )
            self.logger.debug(
                f'Reloaded settings from file: {self.inputs.toml_file.value}'
            )
        else:
            current_settings = {}

        if current_settings.get('active_learning', {}).get('results_dir'):
            results_dir = current_settings['active_learning'].get('results_dir', '')
            final_db_name = current_settings['active_learning'].get('final_db_name', '')
        else:
            results_dir = self.inputs.results_dir.value
            final_db_name = self.inputs.final_db_name.value

        desc_builder.settings_file_path = settings_file_pth

        if current_settings.get('descriptors'):
            descriptor_settings = current_settings.get('descriptors', {})
        else:
            descriptor_settings = self.inputs.descriptor_settings

        if current_settings.get('check_extrapolation_type'):
            check_extrapolation_type = current_settings.get('check_extrapolation_type')
        else:
            check_extrapolation_type = self.inputs.check_extrapolation_type.value

        # Getting the current seed structures
        mace_train_file_path, _ = atl_al_ut.get_final_db_path(
            result_dir_path=results_dir,
            final_db_name=final_db_name,
            node=self.node,
        )
        desc_builder.training_database_path = str(mace_train_file_path)

        if not desc_builder.training_database_path.is_stored:
            desc_builder.training_database_path.store()

        # Get the autoencoder model file. If not found, the calculation will be
        # submitted without providing the file and a new one will be trained
        # at runtime.
        if hasattr(self.ctx, 'autoencoder_model_file'):
            desc_builder.autoencoder_model = self.ctx.autoencoder_model_file

        # Get portable code
        descriptor_code_path = Path(
            f'{ATL_ROOT_DIR}/active_learning/backends/mace/scripts/combined'
        )

        # Loading metadata settings
        computer = orm.load_computer(descriptor_settings['metadata']['computer'])
        desc_builder.metadata.computer = computer

        # Getting resources dict
        resources_dict = descriptor_settings['metadata']['options']['resources']
        num_threads = resources_dict.get('num_cores_per_mpiproc', 2)

        # Prepare and return correct code
        code, prepend_text = atl_al_ut.return_code_from_settings(
            current_settings=current_settings,
            code_settings=descriptor_settings,
            workchain=self,
            num_threads=num_threads,
            executable_name='atl_check_descr_combined.py',
            code_path=descriptor_code_path,
            portable_code_label='atl-descriptors-combined',
            builder=desc_builder,
        )
        desc_builder.code = code

        # Loading AiiDA settings
        mlip_eval_settings_dict = descriptor_settings['metadata']['options']

        # Load scheduler and resources options
        desc_builder.metadata.options = mlip_eval_settings_dict
        desc_builder.metadata.options.parser_name = 'atl-descriptors-combined-parser'
        desc_builder.metadata.options.prepend_text = prepend_text

        # Get the calculation limit, from the computer metadata set to 0
        # if not present.
        # `atl_calc_limit` is a custom property set with:
        # computer.set_property(name='atl_calc_limit', value=366)
        calc_limit = desc_builder.metadata.computer.metadata.get('atl_calc_limit', 0)

        if calc_limit != 0:
            atl_al_ut.aiida_wait_submit(
                builder=desc_builder,
                computer=computer,
                calc_count=0,
            )

        future = self.submit(desc_builder)
        if check_extrapolation_type in ['advanced', 'alpha-shape']:
            boundary_method = (
                current_settings.get('extrapolation', {})
                .get('concave_hull', {})
                .get('boundary_method', 'concave_hull')
            )
            boundary_label = boundary_method.replace('_', ' ')
            self.report(
                f'Submitted calculation ({future.pk}) for descriptors + '
                f'{boundary_label} boundary.'
            )
        else:
            self.report(f'Submitted calculation ({future.pk}) for descriptors.')

        self.to_context(descrptor_results=append_(future))

    def get_descriptor_results(self):
        """Get descriptor results from the calculation and store them in the context."""
        # Get the current iteration and resume mode
        curr_iter = self.inputs.al_loop_iteration.value

        # Ensure that models are loaded only for the first resumed step
        if (
            self.inputs.load_descriptor_calc
            and self.inputs.load_descriptor_calc not in ['', "''", '""']
        ):
            if curr_iter == 0 or (
                self.ctx.resume_mode and not hasattr(self.ctx, 'loaded_descriptor_calc')
            ):
                curr_calc = orm.load_node(self.inputs.load_descriptor_calc.value)

                # Mark that models have been loaded
                self.ctx.loaded_descriptor_calc = True

                self.report(f"Loaded descriptors calculation from '{curr_calc.pk}'.")
            else:
                curr_calc: orm.CalcJobNode = self.ctx.descrptor_results[0]
        else:
            # Get combined descriptor calculation node
            curr_calc: orm.CalcJobNode = self.ctx.descrptor_results[0]

        # Throw exception if calculation didn't complete successfully
        # Descriptors are essential for the workflow to proceed if
        # enabled.
        # If descriptors are disabled, this part of the code is not
        # reached.
        if curr_calc.exit_status != 0:
            return self.exit_codes.ERROR_DESCRIPTOR_CALCULATION.format(
                node_id=curr_calc.pk,
            )

        # Surface the extrapolation outcome in the run log (fills the gap
        # between the descriptor submission and the next MD step).
        if hasattr(curr_calc.outputs, 'extrapolation_metrics'):
            metrics = curr_calc.outputs.extrapolation_metrics.get_dict()
            boundary_method = metrics.get('boundary_method', 'unknown')
            boundary_label = boundary_method.replace('_', ' ')
            self.report(
                f'Extrapolation check ({boundary_label}): '
                f'{metrics["frac_outside"] * 100:.2f}% of '
                f'{metrics["n_structures"]} structures outside boundary '
                f'({metrics["n_boundaries"]} region(s), '
                f'area {metrics["total_boundary_area"]:.3e}).'
            )

        # Loading descriptor min and max into context as numpy arrays
        self.ctx.descriptors_max_array = curr_calc.outputs.descriptor_max
        self.ctx.descriptors_min_array = curr_calc.outputs.descriptor_min

        # Checking if outputs contains the concave hull and if so,
        # storing it in the context
        if hasattr(curr_calc.outputs, 'concave_hull'):
            self.logger.debug('Concave hull output found, storing in context.')
            self.ctx.concave_hull = curr_calc.outputs.concave_hull

        # Get the autoencoder model file
        if hasattr(curr_calc.outputs, 'autoencoder_model'):
            self.logger.debug('Autoencoder output found, storing in context.')
            self.ctx.autoencoder_model_file = curr_calc.outputs.autoencoder_model

        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'Descriptors calculation complete for step '
                    f'{self.inputs.al_loop_iteration.value + 1}'
                    f' - {self.node.pk}.\n'
                ),
            )

    def run_md_seed(self):
        # Don't proceed with submission if MD calculations must be loaded
        if (self.inputs.load_md_calcs and self.inputs.al_loop_iteration.value == 0) or (
            self.inputs.load_md_calcs
            and self.ctx.resume_mode
            and not hasattr(self.ctx, 'loaded_md_calcs')
        ):
            self.report(
                'Loading MD calculations from node(s): '
                f"'{self.inputs.load_md_calcs.get_list()}'."
            )
            return

        with open(self.inputs.current_md_seed_structs_path.value, 'rb') as f:
            current_md_seed_structs = pickle.load(f)

        self.report(
            f'Starting submission of {len(current_md_seed_structs)} '
            'structures to process...'
        )
        calc_count = 0
        for _, curr_structure in enumerate(current_md_seed_structs):
            # Run training and save new model file
            proc_seed = CalculationFactory('atl-process-md-seed-struct')
            proc_seed_builder = proc_seed.get_builder()

            # Input committee models to `commitee_models` namespace as a dict like:
            # {"model_1": "/path/to/model_1/", "model_2": ...}
            # mace_builder.commitee_models = commitee_dict

            # Converting to ase.Atoms()
            # TODO: Find a way of not having to hardcode the keys
            for key in [
                'pbc',
                'cell',
                'numbers',
                'positions',
                'forces',
                'REF_forces',
                'MLIP_forces',
                'momenta',
                'initial_magmoms',
                'bulk_equivalent',
                'bulk_wyckoff',
                'spacegroup_kinds',
                'atl_mlip_eval_forces',
                'curr_model_forces',
            ]:
                if curr_structure.get(key):
                    curr_structure[key] = np.array(curr_structure[key])

            if isinstance(curr_structure, dict):
                curr_structure = Atoms.fromdict(curr_structure)

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename='-',
                    format='extxyz',
                    images=curr_structure,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            md_xyz_file = orm.SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename='md_db.xyz',
            )
            md_xyz_file.store()

            # Add inputs to builder
            proc_seed_builder.md_structure = md_xyz_file
            proc_seed_builder.best_model_name = self.ctx.best_model_name
            proc_seed_builder.m_rmse_e = self.ctx.m0_rmse_e
            proc_seed_builder.m_rmse_f = self.ctx.m0_rmse_f
            proc_seed_builder.curr_active_learning_step = orm.Int(
                self.inputs.al_loop_iteration.value
            )

            # Optional input, advanced extrapolation might not be enabled.
            if hasattr(self.ctx, 'concave_hull'):
                self.logger.debug('Adding concave hull to md seed processing inputs...')
                self.logger.debug(f'{self.ctx.concave_hull}')
                proc_seed_builder.concave_hull = self.ctx.concave_hull
                self.logger.debug(f'{proc_seed_builder.concave_hull}')
                self.logger.debug('Concave hull added.')
            # if hasattr(self.ctx, 'concave_hull_array'):
            #     proc_seed_builder.concave_hull_array = self.ctx.concave_hull_array
            if hasattr(self.ctx, 'autoencoder_model_file'):
                self.logger.debug(
                    'Adding autoencoder model to md seed processing inputs...'
                )
                self.logger.debug(f'{self.ctx.autoencoder_model_file}')
                proc_seed_builder.autoencoder_model = self.ctx.autoencoder_model_file
                self.logger.debug(f'{proc_seed_builder.autoencoder_model}')
                self.logger.debug('Autoencoder model added.')

            proc_seed_builder.desc_max_arr = self.ctx.descriptors_max_array
            proc_seed_builder.desc_min_arr = self.ctx.descriptors_min_array
            proc_seed_builder.settings_file_pth = self.inputs.toml_file

            # The MD part loads the settings dynamically during runtime.
            # Loading the settings file again to get updated settings
            settings_path = Path(self.inputs.toml_file.value)
            if settings_path.exists:
                current_settings = atl_al_ut.read_toml_settings(
                    settings_file=self.inputs.toml_file.value
                )
            else:
                current_settings = {}

            # Preparing comitee dict
            commitee_dict = {}
            for model_name, model in self.ctx.commitee_models_tupl_name_uuid:
                # Only alphanumeric and underscores are allowed as links
                model_name = model_name.replace('-', '_')

                model = orm.load_node(model)
                if not isinstance(model, orm.SinglefileData):
                    commitee_dict[model_name] = orm.SinglefileData(model)
                else:
                    commitee_dict[model_name] = model

            if not commitee_dict:
                self.report(
                    'Committee dict is empty: no committee models trained. '
                    'Using main model only. Check model training step for errors.'
                )

            # Adding the best model to the committee models
            # Only alphanumeric and underscores are allowed as links
            best_model_name_clean = self.ctx.best_model_name.replace('-', '_')
            commitee_dict[best_model_name_clean] = self.ctx.best_model_file

            # Swap in pre-compiled artifacts for the md computer/device when the
            # compile-once step produced them (no-op for MACE / when absent).
            md_meta = current_settings.get('md', {}).get('metadata', {})
            md_computer_label = md_meta.get('computer')
            if md_computer_label:
                md_device = (
                    current_settings.get('md', {})
                    .get('parameters', {})
                    .get('device', 'cpu')
                )
                commitee_dict = {
                    name: self._inference_model_for(model, md_computer_label, md_device)
                    for name, model in commitee_dict.items()
                }

            proc_seed_builder.commitee_models = commitee_dict

            # Loading computer and removing it from the input dictionary
            if current_settings:
                ignore_container = current_settings.get('md', {}).get(
                    'ignore_container', False
                )
                metadata_dict = current_settings.get('md', {}).get('metadata', {})
                resc_dict = metadata_dict.get('options', {}).get('resources', {})
                num_threads = resc_dict.get('num_cores_per_mpiproc')

                # In case the number of threads is not set in the first try
                # try to get an additional key. If it does not work, set it to 1.
                if num_threads is None:
                    num_threads = resc_dict.get('tot_num_mpiprocs', 1)

                prepend_text_conf = metadata_dict.get('prepend_text', '')
            else:
                # TODO: The self input.descriptor_settings should be replaced with
                # the MD metadata section in the toml file, which should be read when
                # preparing the workchain inputs.

                metadata_dict = self.inputs.descriptor_settings['metadata']
                num_threads = self.inputs.descriptor_settings.get('num_cpus', 1)
                prepend_text_conf = self.inputs.descriptor_settings['metadata'].get(
                    'prepend_text', ''
                )

            options_dict = metadata_dict['options']
            computer = orm.load_computer(metadata_dict['computer'])
            proc_seed_builder.metadata.computer = computer
            proc_seed_builder.metadata.label = (
                f'process_{curr_structure.info.get("struct_name", "unknown")}'
            )
            proc_seed_builder.metadata.description = 'Processing structure using MD.'

            # Getting container settings
            containerized = False
            container_dict = self.inputs.container_settings.get_dict()
            if container_dict.get('use_container'):
                containerized = container_dict.get('use_container', False)
            if ignore_container is True:
                containerized = False

            if not hasattr(self.ctx, '_md_seed_code'):
                if containerized:
                    image_name = container_dict.get('image_name', '')
                    engine_command = container_dict.get('engine_command', '')
                    self.ctx._md_seed_code = get_or_create_containerized_code(
                        label='atl_process_md_seed_struct',
                        computer=computer,
                        image_name=image_name,
                        engine_command=engine_command,
                        filepath_executable='atl_process_structure.py',
                    )
                else:
                    code_path = Path(f'{ATL_ROOT_DIR}/active_learning/md')
                    self.ctx._md_seed_code = get_or_create_portable_code(
                        label='atl_process_md_seed_struct',
                        filepath_files=code_path,
                        filepath_executable='atl_process_structure.py',
                    )
            proc_seed_builder.code = self.ctx._md_seed_code

            # options_dict.pop('computer', None)

            # Load scheduler and resources options
            proc_seed_builder.metadata.options = options_dict
            proc_seed_builder.metadata.options.parser_name = (
                'atl-process-md-seed-struct-parser'
            )

            # Pass environment setup dynamically so TOML changes take effect
            # without recreating the code node.
            container_prepend = (
                container_dict.get('prepend_text', '') if containerized else ''
            )
            proc_seed_builder.metadata.options.prepend_text = (
                prepend_text_conf
                + '\n'
                + container_prepend
                + '\nexport PATH=$PATH:.'
                + f'\nexport OMP_NUM_THREADS={num_threads}'
            )

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `atl_calc_limit` is a custom property set with:
            # computer.set_property(name='atl_calc_limit', value=366)
            calc_limit = proc_seed_builder.metadata.computer.metadata.get(
                'atl_calc_limit', 0
            )

            if calc_limit != 0:
                atl_al_ut.aiida_wait_submit(
                    builder=proc_seed_builder,
                    computer=computer,
                    calc_count=calc_count,
                )
            future = self.submit(proc_seed_builder)

            if curr_structure.info.get('aiida_uuid'):
                future.base.extras.set(
                    'unique_id_old', curr_structure.info['aiida_uuid']
                )
            if curr_structure.info.get('atl_id'):
                future.base.extras.set('unique_id', curr_structure.info['atl_id'])

            future.base.extras.set('atl_db_index', curr_structure.info['atl_db_index'])

            self.to_context(process_committee_results=append_(future))

            self.report(f'Submission done for MD calculation: {future.pk}')

        self.report(
            'All calculation submissions done. '
            f'Running MD for {len(self.ctx.process_committee_results)}'
            ' structures...'
        )

        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'All calculation submissions done. '
                    f'Running MD for {len(self.ctx.process_committee_results)}'
                    ' structures...'
                ),
            )

    def send_calc_or_remove_structures(self):
        """Decide which structures to keep and send to DFT or remove from DB."""
        self.report(
            'Selecting which structures need performing DFT or removing from DB...'
        )

        # Get the current iteration and resume mode
        curr_iter = self.inputs.al_loop_iteration.value

        # Ensure that models are loaded only for the first resumed step
        if self.inputs.load_md_calcs:
            if curr_iter == 0 or (
                self.ctx.resume_mode and not hasattr(self.ctx, 'loaded_md_calcs')
            ):
                processed_structures = [
                    orm.load_node(node) for node in self.inputs.load_md_calcs
                ]
                # Mark that models have been loaded
                self.ctx.loaded_md_calcs = True
            else:
                processed_structures = self.ctx.process_committee_results
        else:
            # Get all of the processed structures
            processed_structures = self.ctx.process_committee_results

        self.ctx.curr_md_all_structures_in_domain = False

        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'{len(processed_structures)} MD calculations complete for step '
                    f'{self.inputs.al_loop_iteration.value + 1}'
                    f' - {self.node.pk}.\n'
                ),
            )

        mlip_calcs_struct_list = []
        mlip_calcs_idx_list = []
        calcs_to_submit = []
        delete_indices = []

        # Selecting which structures to send to DFT and
        # which to remove from the database.
        proc_calcjob: orm.CalcJobNode
        for proc_calcjob in processed_structures:
            # Skip calculation if it didn't finish correctly
            if proc_calcjob.exit_status != 0:
                self.report(
                    'Removing struct. processing that finished with errors '
                    f'(pk: {proc_calcjob.pk}).'
                )
                delete_indices.append(proc_calcjob.base.extras.all.get('unique_id'))
                continue

            # Getting unique_id from extras
            orig_str_uuid = proc_calcjob.base.extras.all.get('unique_id')

            # Loading extrapolating structures
            extrap_strc_file: orm.SinglefileData = (
                proc_calcjob.outputs.extrapolating_structures
            )
            with extrap_strc_file.as_path() as file_path:
                extrap_structs = ase_read(
                    filename=file_path, format='extxyz', index=':'
                )

            for struct in extrap_structs:
                struct.info['atl_md_node'] = proc_calcjob.uuid
                struct.info['atl_db_index'] = proc_calcjob.base.extras.get(
                    'atl_db_index'
                )

            # If no extrapolating structures, mark for removal from database
            # using orig_str_uuid
            if len(extrap_structs) == 0:
                delete_indices.append(orig_str_uuid)
            # If extrapolating structures, send to DFT using the selected
            # method.
            else:
                calcs_to_submit.extend(extrap_structs)

        # Limit number of DFT calculations to be submitted using the
        # `dft_calc_limit` key in the dft settings section. Ommit if not present.
        if self.inputs.dft_calc_limit:
            dft_calc_limit = self.inputs.dft_calc_limit.value

            if len(calcs_to_submit) > dft_calc_limit:
                step = len(calcs_to_submit) / dft_calc_limit
                sampled_indices = [int(i * step) for i in range(dft_calc_limit)]
                calcs_to_submit = [calcs_to_submit[idx] for idx in sampled_indices]

        if len(calcs_to_submit) == 0 and proc_calcjob.is_finished_ok:
            self.ctx.curr_md_all_structures_in_domain = True
            self.report(
                'No structures to send to DFT. All explored structures are in domain.'
            )
        elif len(calcs_to_submit) == 0:
            self.report(
                'No structures to send to DFT. '
                'There were problems during the processing step (MD).'
            )

            # Setting error to stop the active learning loop
            self.ctx.stop_al_loop_error = orm.Bool(True)

        # Validate kspacing coverage before submitting any DFT calculations.
        if self.inputs.dft_method == 'vasp' and len(calcs_to_submit) > 0:
            dft_settings = self.inputs.dft_settings.get_dict()
            kspacing_dict = dft_settings.get('kspacing', {})
            phases = sorted(
                set(s.info.get('phase', 'unknown') for s in calcs_to_submit)
            )
            validate_kspacing(kspacing_dict, phases)

        # Submitting calcs
        calc_count = 0
        for struct in calcs_to_submit:
            # Get row and index
            struct_uuid = struct.info.get('atl_id')
            if not struct_uuid:
                struct_uuid = struct.info.get('aiida_uuid')
            calc_idx = struct.info.get('atl_db_index')

            if self.inputs.dft_method == 'vasp':
                row = struct.info

                # TODO: Dynamically load settings
                # Loading the settings file again to get updated settings
                # settings_path = Path(self.inputs.toml_file.value)
                # if settings_path.exists:
                #     current_settings = atl_al_ut.read_toml_settings(
                #         settings_file=self.inputs.toml_file.value
                #     )
                #     self.logger.debug(
                #         f'Reloaded settings from file: {self.inputs.toml_file.value}'
                #     )
                # else:
                #     current_settings = {}

                builder = atl_al_ut.get_dft_calc_builder_vasp(
                    struct=struct,
                    row=row,
                    calc_idx=calc_idx,
                    group=None,
                    dft_settings=self.inputs.dft_settings.get_dict(),
                )

                # Get the code and computer from the builder, updated
                # to the current version of aiida-vasp.
                try:
                    curr_code = builder.code
                    curr_computer = curr_code.computer
                except AttributeError:
                    curr_code = builder.vasp.code
                    curr_computer = curr_code.computer

                # Get the calculation limit, from the computer metadata set to 0
                # if not present.
                # `atl_calc_limit` is a custom property set with:
                # computer.set_property(name='atl_calc_limit', value=366)
                calc_limit = 0
                try:
                    calc_limit = builder.metadata.computer.metadata.get(
                        'atl_calc_limit', 0
                    )
                except AttributeError:
                    # The code namespace is in a different place in aiida-vasp 4.1.0
                    # This except block tries to get this new namespace and use
                    # it instead.
                    try:
                        calc_limit = builder.vasp.code.computer.metadata.get(
                            'atl_calc_limit', 0
                        )
                    except Exception:
                        # Any exception in the alternative path
                        # will result in calc_limit being 0.
                        calc_limit = 0
                except Exception:
                    # If everything else fails, screw it it, 0 it is.
                    calc_limit = 0

                if calc_limit != 0:
                    atl_al_ut.aiida_wait_submit(
                        builder=builder,
                        computer=curr_computer,
                        calc_count=calc_count,
                        code=curr_code,
                    )

                # Submitting current calculation
                future = self.submit(builder)

                unique_id = row.get('unique_id')
                if not unique_id:
                    unique_id = row.get('aiida_uuid')
                struct_name = row.get('material_name')
                if not struct_name:
                    struct_name = row.get('struct_name')

                future.base.extras.set('atl_calc_uuid', unique_id)
                future.base.extras.set('atl_struct_type', row['atl_struct_type'])
                future.base.extras.set('struct_name', struct_name)
                self.to_context(dft_struct_seed_calcs=append_(future))

                if self.inputs.train_seed_group.value:
                    group = orm.load_group(self.inputs.train_seed_group.value)
                    group.add_nodes(future)

            elif self.inputs.dft_method in ('mace', 'mlip'):
                mlip_calcs_struct_list.append(struct)
                mlip_calcs_idx_list.append(calc_idx)

        is_mlip = self.inputs.dft_method in ('mace', 'mlip')
        if is_mlip and len(mlip_calcs_struct_list) > 0:
            builder = atl_al_ut.get_dft_calc_builder_mlip_list(
                struct_list=mlip_calcs_struct_list,
                dft_settings=self.inputs.dft_settings.get_dict(),
                container_settings=self.inputs.container_settings.get_dict(),
            )

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `atl_calc_limit` is a custom property set with:
            # computer.set_property(name='atl_calc_limit', value=366)
            calc_limit = builder.code.computer.metadata.get('atl_calc_limit', 0)

            # Check if the calculation can be submitted
            if calc_limit != 0:
                atl_al_ut.aiida_wait_submit(
                    builder=builder,
                    computer=builder.code.computer,
                    calc_count=calc_count,
                )

            # Submitting current calculation
            future = self.submit(builder)
            future.base.extras.set('atl_calc_uuid', struct.info.get('aiida_uuid'))
            future.base.extras.set(
                'atl_struct_type', struct.info.get('atl_struct_type')
            )
            future.base.extras.set('struct_name', struct.info.get('struct_name'))
            future.base.extras.set('atl_md_node', struct.info.get('atl_md_node'))

            self.to_context(dft_struct_seed_calcs=append_(future))

            if self.inputs.train_seed_group.value:
                group = orm.load_group(self.inputs.train_seed_group.value)
                group.add_nodes(future)

        self.report(
            f'Committee decision: {len(calcs_to_submit)} get info / '
            f'{len(delete_indices)} delete.'
        )

        self.logger.log(15, f"Structures to delete: '{delete_indices}'")

        if set(delete_indices) == {'unknown'}:
            self.report("All structures to delete marked as 'unknown'. ")
            raise ChildProcessError

        # Deleting well represented structures from seed_gen_db (Ds), if
        # there are any and the seed deletion is enabled in the configuration
        # file.
        delete_seed_structs: bool = self.inputs.delete_seed_structs.value
        if len(delete_indices) > 0 and delete_seed_structs:
            self.report(
                f'Deleting {len(delete_indices)} structures from seed'
                ' generating DB (Ds)'
            )

            atl_al_ut.remove_structs_from_seed_gen_db(
                self.inputs.seed_db_path, delete_indices
            )
            self.report('Done deleting structures. Moving to data acquisition step...')

        # If no structure is well represented, nothing will be deleted.
        else:
            self.report('Nothing removed from DB.')

        # Notify about DFT and deletions if enabled
        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'Committee decision: {len(calcs_to_submit)} get DFT info / '
                    f'{len(delete_indices)} delete.'
                ),
            )

    def return_seed_dft_and_model(self):
        """
        Gather and output the last NNP model and DFT calculations for the current seed.

        This function collects DFT calculations for the structures in the current seed,
        which are then returned as outputs in the workchain using the namespace
        `dft_calcs`. A check is performed to determine if the results agree using
        MACE models, and this check also outputted to the workchain using the
        namespace `stop_md_seed_no_disagreement`.
        """
        self.report('Gathering DFT calculations...')
        # Getting the results directory if not in the context
        if not hasattr(self.ctx, 'results_dir'):
            self.ctx.results_dir = atl_al_ut.get_results_dir_path(
                result_dir_path=self.inputs.results_dir.value,
                node=self.node,
            )

        # Running VASP calulations if its the selected method
        if self.inputs.dft_method == 'vasp':
            try:
                dft_calcs = len(self.ctx.dft_struct_seed_calcs)

                # After setting
                dft_calcs_ok = [
                    node.uuid
                    for node in self.ctx.dft_struct_seed_calcs
                    if node.is_finished_ok
                    # if node.is_finished and node.exit_status in [0, 504, 503]
                ]

                if len(dft_calcs_ok) == 0 and dft_calcs > 0:
                    self.report('No DFT calculations finished correctly.')
                    dft_calc_list = ''
                    self.ctx.stop_al_loop_error = orm.Bool(True)
                elif len(dft_calcs_ok) == 0 and dft_calcs == 0:
                    self.report('No VASP DFT calculation jobs performed.')
                    dft_calc_list = ''
                else:
                    self.report(
                        f'Gathering {len(dft_calcs_ok)} VASP DFT calculations '
                        'that finished correctly...'
                    )
                    dft_calc_list = atl_al_ut.gather_dft_calcs_vasp(dft_calcs_ok)
                    self.report('Done gathering VASP DFT calculations!')

            except AttributeError:
                dft_calc_list = ''

        elif self.inputs.dft_method in ('mace', 'mlip'):
            if hasattr(self.ctx, 'dft_struct_seed_calcs'):
                dft_calcs = self.ctx.dft_struct_seed_calcs
            else:
                dft_calcs = []
            try:
                # Gather correct MACE calculations
                dft_calcs_len = len(dft_calcs)
                dft_calcs_ok = [node.uuid for node in dft_calcs if node.is_finished_ok]

                if len(dft_calcs_ok) == 0 and dft_calcs_len > 0:
                    self.report('No MLIP evaluations finished correctly.')
                    dft_calc_list = ''
                    self.ctx.stop_al_loop_error = orm.Bool(True)
                elif len(dft_calcs_ok) == 0 and dft_calcs_len == 0:
                    self.report('No MLIP evaluation jobs performed.')
                    dft_calc_list = ''
                else:
                    # Gather all MACE evaluations, storing results into a file,
                    # stored in `result_list_path`.
                    # Results are filtered to remove outliers. Outliers are
                    # stored in a separate file in the same folder.
                    dft_calc_list: orm.List = atl_al_ut.gather_dft_calcs_mlip(
                        dft_calc_list=dft_calcs_ok,
                        results_dir=str(self.ctx.results_dir),
                        workchain=self.node.uuid,
                    )
                    self.report(
                        f'Gathered {len(dft_calcs)} MLIP evaluation calculation jobs.'
                    )
            except AttributeError:
                dft_calc_list = ''
                self.report(
                    'Exception raised when gathering MLIP evaluation calculation jobs.'
                )

        if len(dft_calc_list) > 0:
            # Run filtering step based on NN vs DFT difference threshold
            # for both E and F
            filter_settings = self.inputs.dft_settings.get('filter', {})
            if filter_settings.get('filter_dft_calcs', False):
                threshold_E_meV = filter_settings.get('threshold_E_meV', 1e3)
                threshold_F_meV = filter_settings.get('threshold_F_meV', 1e4)

                # For non MLIP cases (DFT), the model predictions are not readily
                # accessible, so they must be generated on the fly.
                if self.inputs.dft_method not in ('mace', 'mlip'):
                    dft_calc_list = atl_al_ut.sampler_populate_E_and_F_list(
                        structure_list=dft_calc_list,
                        model_file=self.ctx.best_model_file,
                        backend_name=self.ctx.mlip_backend_name,
                    )

                self.report(
                    'Removing DFT structures with differences higher than: '
                    f'E - {threshold_E_meV} meV, F - {threshold_F_meV} meV'
                )
                dft_count = len(dft_calc_list)
                dft_calc_list = atl_al_ut.filter_dft_calcs_threshold(
                    dft_calc_list=dft_calc_list,
                    threshold_E_meV=threshold_E_meV,
                    threshold_F_meV=threshold_F_meV,
                    workchain=self,
                )
                self.report(
                    f'Removed {abs(len(dft_calc_list) - dft_count)}'
                    ' DFT above thresholds.'
                )

            return_list_path, _ = atl_al_ut.write_gathered_dft_calcs_to_file(
                dft_calc_list=dft_calc_list,
                results_dir=str(self.ctx.results_dir),
                workchain=self,
            )
        else:
            # If no DFT calculations were performed, return empty string
            return_list_path = ''

        # File containing structures
        if return_list_path:
            if isinstance(return_list_path, Path):
                return_list_path = orm.Str(return_list_path)
            self.out('dft_calcs_path', return_list_path)

        # orm.SinglefileData for the MACE model with the best performance
        self.out('m0_model_file', self.ctx.best_model_file)
        self.logger.log(
            15,
            f'Saved best model '
            f"'{self.ctx.best_model_file.base.extras.all.get('model_name')}'"
            f' ({self.ctx.best_model_file.pk}) in workchain.',
        )

        self.out(
            'stop_md_seed_no_disagreement',
            atl_al_ut.check_md_seed_agreement(
                return_list_path,
                self.ctx.curr_md_all_structures_in_domain,
            ),
        )

        # Return error status as an output
        self.out('stop_al_loop_error', self.ctx.stop_al_loop_error)

        if hasattr(self.inputs, 'enable_ntfysh'):
            requests.post(
                f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                data=(
                    f'Data acquisition stage complete for step '
                    f'{self.inputs.al_loop_iteration.value + 1}'
                    f' - {self.node.pk}.\n'
                ),
            )


class SimpleActiveLearningBaseWorkChain(BaseRestartWorkChain):
    """Base workchain for ATL active learning workflows.

    This workchain is used as a base for the `SimpleActiveLearningWorkChain` workchain.
    It handles setup of the workchain and the main loop, where the active learning
    steps are launched. After every step, the results are checked and added to the
    database, and the next step is prepared. The workchain will loop until the
    stopping conditions are met. If the stopping conditions and the safeguard check
    is enabled, a MD-based safeguard mechanism will be employed to ensure the current
    sampler model is robust, and therefore the loop can be stopped.

    It takes all the inputs of the `SimpleActiveLearningWorkChain` workchain, except for
    the inputs that are specific to the active learning loop.
    Additionally, the mandatory `log_path` and optional `resume_dict` can be provided.
    See the `define` method for more information on the inputs and outputs.

    Note
    ----
    This workchain can be restarted from a previous running workchain if its
    files are recovered. The optional `resume_dict` input is used for this.
    `resume_dict` must be a dictionary that contains the following keys:
     - `last_iteration`: The current iteration of the active learning loop.
     - `train_db_path`: The path to the last training database.
     - `seed_db_path`: The path to the last seed generation database.

    The loop will only be restarted from the beginning of the last step.

    Check `SimpleActiveLearningWorkChain` for information on what is done in each step.
    """

    _process_class = SimpleActiveLearningWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Expose all inputs from SimpleActiveLearningWorkChain except
        # those that are specific to the active learning loop steps themselves.
        # Add those in the exclude list so they are not asked for at this level.
        spec.expose_inputs(
            SimpleActiveLearningWorkChain,
            namespace='active_learning',
            exclude=[
                'current_md_seed_structs_path',
                'current_md_seed_structs_idx',
                'al_loop_iteration',
                'train_seed_group',
                'seed_gen_db',
                'seed_size',
                'seed_db_path',
                'training_db_path',
                'database_training',
                'log_path',
                'debug_mode',
                'enable_ntfysh',
                'ntfysh_topic',
                'current_seed_size',
                'use_test_db',
            ],
        )
        spec.input('log_path', valid_type=orm.Str, serializer=orm.to_aiida_type)

        # Optional input to resume the workchain from a previous state.
        spec.input(
            'resume_dict',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )

        spec.outline(
            # Add a filehandler to aiida logger
            cls.setup_textfile_logging,
            # Run version check,
            cls.log_atl_version,
            # Load the initial database (D_ini), that will be used as the
            # training database (Dt) without changing the original database.
            # Additionally, create a copy of the database (seed_gen_db, Ds),
            # this will be used to generate the MD seeds.
            cls.get_database,
            # Copy toml file to results directory
            cls.copy_input_toml_file,
            # Create inputs for workchains and initialize iterative counter
            cls.setup,
            # Get a settings report for the active learning loop.
            cls.get_input_report,
            # Keep the loop running while the safeguard check is not failed
            while_(cls.should_run_according_to_safeguard_check)(
                # This part will loop to complete the process
                # It will loop `self.ctx.inputs.max_al_iterations` times.
                while_(cls.check_al_loop_conditions)(
                    # Check status of resume mode.
                    cls.check_resume_mode,
                    # Get random structures from Ds to generate the MD seed.
                    # For data reduction mode, filter out structures already
                    # in training DB.
                    cls.get_seed_structures,
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
                if_(cls.should_run_safeguard)(
                    # Optionally pre-compile the sampler model on the safeguard
                    # computer (no-op for backends without compilation).
                    cls.compile_safeguard_model,
                    cls.gather_safeguard_compiled,
                    # Run safeguard check
                    cls.run_safeguard_check,
                    cls.parse_safeguard_check_results,
                ),
            ),
            cls.results_final,
        )
        spec.output(
            'final_training_db',
            valid_type=orm.SinglefileData,
        )
        spec.output('test_db_file', valid_type=orm.SinglefileData, required=False)
        spec.output('final_model_file', valid_type=orm.SinglefileData)

    def setup_textfile_logging(self):
        # Get log path
        log_path = self.inputs.log_path.value

        # Getting aiida logger
        aiida_logger = logging.getLogger('aiida')
        cli_handler = aiida_logger.handlers[0]
        aiida_logger.removeHandler(cli_handler)

        # Set the PARENT logger to INFO to silence framework debug messages
        aiida_logger.setLevel(15)

        # We only want to see our custom debug messages and our reports.
        log_filter = LevelNameFilter(
            levels_to_keep=[
                'ATL_DEBUG',
                'REPORT',
                '[ ✔ ]',
                '[ ! ]',
                '[ X ]',
                '[...]',
            ]
        )

        # Create a file handle
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(1)
        file_handler.addFilter(log_filter)

        # Create a formatter and set it for the file handler
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] - %(message)s',
            datefmt='%m/%d/%y %H:%M:%S',
        )
        file_handler.setFormatter(file_formatter)
        aiida_logger.addHandler(file_handler)

        # Adding console logger
        console = Console(
            color_system='truecolor', theme=ATL_THEME, highlighter=ATLHighlighter()
        )

        ch = ATLRichHandler(
            markup=True,
            show_path=False,
            log_time_format='[%d/%m/%y %H:%M:%S]',
            omit_repeated_times=False,
            console=console,
        )
        ch.setLevel(23)
        formatter_con = logging.Formatter('%(message)s')
        ch.setFormatter(formatter_con)
        aiida_logger.addHandler(ch)

        self.report('Started Simple Active Learning!')
        self.report(f"Logging in '{log_path}'.")

    def log_atl_version(self):
        curr_version, _, hash_str = get_atl_version_info()
        self.report(f"Using ATLAS version: '{curr_version}' ({hash_str[:7]}).")

    def get_database(self):
        """Loading initial database."""
        self.report('Reading database file...')

        # Check the AL mode. If no mode is specified, the default mode
        # is 'data_acquisition', where data is generated using a simulation
        # technique.
        if (
            hasattr(self.inputs.active_learning, 'al_mode')
            and self.inputs.active_learning.al_mode is not None
        ):
            self.ctx.al_mode = self.inputs.active_learning.al_mode.value
        else:
            self.ctx.al_mode = 'data_acquisition'

        self.report(f'Loading databases for active learning mode: {self.ctx.al_mode}.')

        if self.ctx.al_mode == 'data_reduction':
            self._setup_data_reduction_databases()
        else:
            self._setup_data_acquisition_databases()

    def _setup_data_reduction_databases(self):
        """Setup databases for data reduction mode."""
        self.report('Setting up databases for data reduction mode...')

        # Get data reduction settings
        if not self.inputs.active_learning.data_reduction_settings:
            raise ValueError(
                "data_reduction_settings must be provided when al_mode='data_reduction'"
            )

        data_reduction_settings = (
            self.inputs.active_learning.data_reduction_settings.get_dict()
        )
        large_db_path = data_reduction_settings['large_database_path']
        initial_selection_size = data_reduction_settings['initial_selection_size']
        initial_selection_method = data_reduction_settings['initial_selection_method']

        # Validate large database path
        if not Path(large_db_path).exists():
            raise FileNotFoundError(f'Large database file not found: {large_db_path}')

        # Load the large database that will serve as the seed database
        if self.inputs.resume_dict:
            # If resuming, load the seed database from the previous run
            large_database = ase_read(
                filename=self.inputs.resume_dict['seed_db_path'],
                format='extxyz',
                index=':',
            )
            # Load training database from previous run
            database_training = ase_read(
                filename=self.inputs.resume_dict['train_db_path'],
                format='extxyz',
                index=':',
            )
        else:
            # Load the large database
            large_database = ase_read(
                filename=large_db_path,
                format='extxyz',
                index=':',
            )
            # Start with empty training database
            database_training = []

        # Process database structures and add metadata
        for idx, struct in enumerate(large_database):
            if not struct.info.get('atl_db_index'):
                struct.info['atl_db_index'] = idx
            if not struct.info.get('atl_al_step'):
                struct.info['atl_al_step'] = 0
            if not struct.info.get('atl_id'):
                struct.info['atl_id'] = str(uuid.uuid4())
            large_database[idx] = struct

        # Process training database structures
        for idx, struct in enumerate(database_training):
            if not struct.info.get('atl_db_index'):
                struct.info['atl_db_index'] = idx
            if not struct.info.get('atl_al_step'):
                struct.info['atl_al_step'] = 0
            if not struct.info.get('atl_id'):
                struct.info['atl_id'] = str(uuid.uuid4())
            database_training[idx] = struct

        # Create result directories
        results_dir_path = Path(self.inputs.active_learning.results_dir.value)
        if not results_dir_path.exists():
            results_dir_path.mkdir()

        final_db_path, curr_run_results_dir = atl_al_ut.get_final_db_path(
            result_dir_path=results_dir_path,
            final_db_name=self.inputs.active_learning.final_db_name.value,
            node=self.node,
        )
        self.ctx.curr_run_results_dir = curr_run_results_dir

        # Set up seed database (the large database)
        self.ctx.seed_db_path = curr_run_results_dir / 'atl_seed_db.xyz'

        # Set up training database path
        self.ctx.training_db_path = final_db_path

        # If this is a new run and we have an empty training database,
        # perform initial selection
        if not self.inputs.resume_dict and len(database_training) == 0:
            msg = (
                f'Performing initial selection of {initial_selection_size} '
                f'structures using {initial_selection_method} method...'
            )
            self.report(msg)

            # Get descriptor settings for FPS if needed
            descriptor_settings = None
            if initial_selection_method == 'fps':
                descriptor_settings = (
                    self.inputs.active_learning.descriptor_settings.get_dict()
                )

            # Select initial structures
            selected_structures = atl_al_ut.select_structures_data_reduction(
                database=large_database,
                n_structures=initial_selection_size,
                selection_method=initial_selection_method,
                descriptor_settings=descriptor_settings,
            )

            # Remove selected structures from large database
            selected_ids = {s.info['atl_id'] for s in selected_structures}
            large_database = [
                s for s in large_database if s.info['atl_id'] not in selected_ids
            ]

            # Add to training database
            database_training.extend(selected_structures)

            msg = (
                f'Selected {len(selected_structures)} structures '
                'for initial training database.'
            )
            self.report(msg)

        # Save databases
        ase_write(
            filename=self.ctx.seed_db_path, format='extxyz', images=large_database
        )
        ase_write(
            filename=self.ctx.training_db_path,
            format='extxyz',
            images=database_training,
        )

        # Set minimum seed size (for MD seed selection, not the large database)
        seed_inputs = self.inputs.active_learning
        if (
            hasattr(seed_inputs, 'seed_min_num_structs')
            and seed_inputs.seed_min_num_structs.value
        ):
            self.ctx.min_seed_size = int(seed_inputs.seed_min_num_structs.value)
        else:
            # For data reduction, we set a reasonable default
            self.ctx.min_seed_size = min(25, len(large_database))

        # For data reduction mode, seed size is the number of structures per iteration
        structures_per_iteration = data_reduction_settings.get(
            'structures_per_iteration', 50
        )
        self.ctx.seed_size = structures_per_iteration

        msg = (
            f'Data reduction setup complete. Initial seed database: '
            f'{len(large_database)} structures, '
            f'Training database: {len(database_training)} structures.'
        )
        self.report(msg)

    def _setup_data_acquisition_databases(self):
        """Setup databases for traditional MD-based active learning mode."""
        # The training database (Dt) from which copies are made
        # for further processing. New structures will be added here.
        if self.inputs.resume_dict:
            database_training = ase_read(
                filename=self.inputs.resume_dict['train_db_path'],
                format='extxyz',
                index=':',
            )
        else:
            database_training = ase_read(
                filename=self.inputs.active_learning.init_db_path.value,
                format='extxyz',
                index=':',
            )

        # Save a copy of the initial database
        init_db_path = Path(self.inputs.active_learning.init_db_path.value)
        init_db_filename = init_db_path.name
        backup_filename = 'original_' + init_db_filename + '.bak'
        backup_path = init_db_path.with_name(backup_filename)

        if not backup_path.exists():
            ase_write(
                filename=backup_path,
                format='extxyz',
                images=database_training,
            )
            self.report(f"Saved backup of original database in: '{backup_path}'")
        else:
            self.report(
                f"Backup of original database already exists in: '{backup_path}'"
            )

        # Adding the database indexes to the info dict of the structures
        # and the current active learning loop step index (0).
        for idx, struct in enumerate(database_training):
            # Storing position in the database
            if not struct.info.get('atl_db_index'):
                struct.info['atl_db_index'] = idx

            # If workchain is resumed from a previous run, we should keep
            # the step index from the previous run.
            if self.inputs.resume_dict:
                # However, structures without step numbers will be set to 0.
                if not struct.info.get('atl_al_step'):
                    struct.info['atl_al_step'] = 0

            # On the other hand, if starting from scratch, we set the step index to 0,
            # as this is the first step of the active learning loop.
            else:
                # Adding step index
                struct.info['atl_al_step'] = 0

            # Adding unique id to structures that don't have it.
            if not struct.info.get('atl_id'):
                struct.info['atl_id'] = str(uuid.uuid4())

            database_training[idx] = struct

        # Check if every structure has an unique uuid
        missing_indices, duplicates = atl_al_ut.check_atl_ids(database_training)

        if len(missing_indices) != 0:
            self.logger.error(
                f"Found {len(missing_indices)} structures missing 'atl_id'."
            )
            self.logger.error(
                f"Indices of structures missing 'atl_id': {missing_indices}"
            )

        elif len(duplicates) != 0:
            self.logger.error(f"Found {len(duplicates)} repeated 'atl_id' values:")
            for atl_id, count in duplicates.items():
                self.logger.error(f"  ID '{atl_id}' appears {count} times.")

        if len(missing_indices) == 0 and len(duplicates) == 0:
            self.report("All structures have a correct 'atl_id' field.")
        else:
            raise ValueError(
                'Some structures in the initial database lack unique IDs. '
                'Please ensure all structures have a unique "atl_id" in their info '
                'or remove all atl_id keys and re-run to auto-generate them. '
                'See above for more information.'
            )

        # If test database is enabled, load it or prepare it depending on settings,
        # storing it into a node, and adding it into the context
        test_settings = self.inputs.active_learning.get('eval_test_db_settings', {})
        use_test_db = test_settings.get('use_test_db', False)
        test_db_path = test_settings.get('test_db_path')
        test_db_frac = test_settings.get('test_db_frac')

        if use_test_db:
            self.report(
                'Test database usage enabled. '
                'Preparing test database to be used during the AL Loop...'
            )

            # Save a copy of the initial database
            if test_db_path is None:
                init_db_filename = init_db_path.name
                test_db_filename = 'atl_test-db_' + init_db_filename
                test_db_path = init_db_path.with_name(test_db_filename)

                # We store the updated path in the node's extras. If this WorkChain
                # crashes, the resume script can gather the test_settings extras.
                test_settings.base.extras.set('test_db_path', str(test_db_path))

            # We must consider if the workchain is being resumed or not, as this
            # will affect how the test database is prepared.
            # If the workchain is being resumed, we check if the test database was
            # already prepared in the previous run, so we just need to load it and
            # add it to the context.
            # If the workchain is not being resumed, we need to prepare the test
            # database from the initial training database, and then add it to
            # the context.
            if self.inputs.resume_dict:
                self.report(
                    'Resume mode enabled: '
                    'Checking for existing test database prepared in previous run...'
                )

                new_test_db_path = test_settings.base.extras.get(
                    'test_db_path',
                )

                if new_test_db_path and Path(new_test_db_path).exists():
                    test_db_path = Path(new_test_db_path)
                    self.report(
                        f'Found existing test database prepared in previous run: '
                        f"'{test_db_path}'. Loading it and adding to context..."
                    )

            # Create test database, remove structures from Dt and create a
            # SinglefileData node for it.
            test_db_file, test_db_structures, database_training = (
                atl_al_ut.prepare_test_set(
                    test_db_path=test_db_path,
                    test_db_frac=test_db_frac,
                    training_db=database_training,
                )
            )

            # Only store the file if it hasn't been stored yet
            if not test_db_file.is_stored:
                test_db_file.store()
            self.ctx.test_db_file = test_db_file
            self.ctx.use_test_db = orm.Bool(True)

            if not test_db_path.exists():
                ase_write(
                    filename=test_db_path,
                    format='extxyz',
                    images=test_db_structures,
                )
                self.report(f"Saved backup of test database in: '{test_db_path}'")

            self.ctx.test_settings = orm.Dict(test_settings)

            self.report(
                f'Prepared test set containing {len(test_db_structures)} structures.'
            )
            self.out('test_db_file', test_db_file)
        else:
            self.report(
                'Test database usage disabled. '
                'No test database will be used during the AL Loop.'
            )
            self.ctx.test_db_file = None
            self.ctx.use_test_db = orm.Bool(False)
            self.ctx.test_settings = orm.Dict({})

        # Update modified training database
        ase_write(
            filename=self.inputs.active_learning.init_db_path.value,
            format='extxyz',
            images=database_training,
        )

        # Create files for database_training and seed_gen_db
        results_dir_path = Path(self.inputs.active_learning.results_dir.value)
        if not results_dir_path.exists():
            results_dir_path.mkdir()

        final_db_path, curr_run_results_dir = atl_al_ut.get_final_db_path(
            result_dir_path=results_dir_path,
            final_db_name=self.inputs.active_learning.final_db_name.value,
            node=self.node,
        )
        self.ctx.curr_run_results_dir = curr_run_results_dir

        # A copy of the initial database, (Ds)
        # used specifically for generating MD seeds and running the MDs.
        # New structures will be added and well represented configs removed from here.
        self.ctx.seed_db_path = curr_run_results_dir / 'atl_seed_db.xyz'

        # Copying the seed database from the initial database if new run,
        # otherwise copy from the previous run.
        if self.inputs.resume_dict:
            shutil.copy(self.inputs.resume_dict['seed_db_path'], self.ctx.seed_db_path)
        else:
            shutil.copy(
                self.inputs.active_learning.init_db_path.value, self.ctx.seed_db_path
            )

        # Setting the minimum seed size for the md seed. Min size defaults to the
        # seed size fraction if not set.
        al_inputs = self.inputs.active_learning
        if hasattr(al_inputs, 'seed_size_frac') and al_inputs.seed_size_frac.value:
            self.ctx.min_seed_size = int(
                self.inputs.active_learning.seed_min_num_structs.value
            )
        else:
            self.ctx.min_seed_size = int(
                self.inputs.active_learning.seed_size_frac.value
                * len(database_training)
            )

        self.ctx.seed_size = int(
            self.inputs.active_learning.seed_size_frac.value * len(database_training)
        )

        # Create a copy of the training database for the current run.
        # If reading from a previous run, copy the database from the previous run.
        self.ctx.training_db_path = final_db_path
        if self.inputs.resume_dict:
            shutil.copy(
                self.inputs.resume_dict['train_db_path'], self.ctx.training_db_path
            )
            self.report(
                'Loaded initial database from previous run containing '
                f"'{len(database_training)}' structures."
            )
        else:
            shutil.copy(
                self.inputs.active_learning.init_db_path.value,
                self.ctx.training_db_path,
            )
            self.report(
                'Loaded initial database containing '
                f"'{len(database_training)}' structures."
            )

    def copy_input_toml_file(self):
        """Copy the input toml file to the results directory."""
        toml_file_path = Path(self.inputs.active_learning.toml_file.value)
        toml_file_name = toml_file_path.name
        toml_file_dest = self.ctx.curr_run_results_dir / toml_file_name

        shutil.copy(toml_file_path, toml_file_dest)

    def get_results_loop(self):
        """Attach the outputs specified in the spec from the last completed process."""
        self.logger.log(15, f'Getting results for iteration {self.ctx.iteration}.')

        # Mark last iteration differently for first resume step.
        first_resume_step = False
        is_resume_run = (
            hasattr(self.inputs, 'resume_dict') and self.inputs.resume_dict is not None
        )
        if is_resume_run:
            # If resuming, we need to check if this is the first step after resuming.
            # If it is, we need to get the last iteration from the resume_dict.
            # Otherwise, we can just use the current iteration.
            first_resume_step = (
                self.inputs.resume_dict['last_iteration'] == self.ctx.iteration - 1
            )

        if first_resume_step is True and is_resume_run is True:
            last_iteration = self.inputs.resume_dict['last_iteration']
            self.logger.log(
                15,
                f'Detected resume mode, first step. Last iteration:  {last_iteration}',
            )
            node = self.ctx.children[self.ctx.iteration - (last_iteration - 1) - 2]

        # If resuming, but not the first step
        elif first_resume_step is False and is_resume_run is True:
            self.logger.log(
                15,
                f'Detected resume mode but NOT first step. '
                f'Last iteration: {self.ctx.iteration}',
            )
            node = self.ctx.children[-1]
        else:
            node = self.ctx.children[self.ctx.iteration - 1]
            self.logger.log(
                15, f'Detected resume mode OFF. Last iteration:  {node.uuid}'
            )

        self.logger.log(
            15, f"SimpleActiveLearningWorkChain to gather results from: '{node.uuid}'"
        )

        # TODO: Gather outputs manually, instead of using __attach_outputs
        # outputs = self._attach_outputs(node)

        # Sending seed disagreement flag to context
        if hasattr(node.outputs, 'stop_md_seed_no_disagreement'):
            self.ctx.stop_md_seed_no_disagreement = node.outputs[
                'stop_md_seed_no_disagreement'
            ]

        self.ctx.last_workchain_completed = node
        self.logger.log(15, f'Done getting results for iteration {self.ctx.iteration}.')

        # Update test db eval results if available
        if hasattr(node.outputs, 'test_db_eval_results'):
            last_eval_dict = node.outputs['test_db_eval_results'].get_dict()

            # If test_db_eval_results already exists in context, update it
            if hasattr(self.ctx, 'test_db_eval_results'):
                self.ctx.test_db_eval_results.update(last_eval_dict)
            else:
                self.ctx.test_db_eval_results = last_eval_dict

            self.ctx.inputs.test_db_eval_results = self.ctx.test_db_eval_results

        # Send previous calculation error status to context
        # stop_al_loop_error is True if there was an error in the last step
        # and False otherwise
        if hasattr(node.outputs, 'stop_al_loop_error'):
            self.ctx.stop_al_loop_error = node.outputs['stop_al_loop_error']
        else:
            self.ctx.stop_al_loop_error = orm.Bool(False)

        # Resetting safeguard attempted flag
        # The reasoning behind this is that the safeguard must only be attempted
        # once per AL step at most. Since here we are finishing an AL step correctly
        # the safeguard flag is set to False. After the safeguard is run, the flag is
        # set to True. In that way, it can only be reset here when a new AL step is
        # completed.
        self.logger.debug('Setting safeguard_attempted flag to False.')
        self.ctx.safeguard_attempted = False

        return None

    def add_dft_results_to_db(self):
        """
        Incorporate DFT calculation results into the training/seed generation databases.

        This method updates the training and seed generation databases with DFT
        calculation results. If any DFT calculations have been performed,
        their results are appended to both the training database and the seed
        generation database.
        """
        # Updating final and seed database.
        self.report('Checking if database files must be updated...')

        # Updating current training seed
        try:
            seed_gen_db = atl_al_ut.load_database(self.ctx.seed_db_path)
            training_db = atl_al_ut.load_database(self.ctx.training_db_path)
            last_wc = self.ctx.last_workchain_completed
            dft_calcs = ase_read(
                last_wc.outputs['dft_calcs_path'].value, format='extxyz', index=':'
            )

            cnt_dft_calcs = len(dft_calcs)

        except KeyError:
            self.report('No new structures available for DB.')
            cnt_dft_calcs = 0

        if cnt_dft_calcs > 0:
            self.report(f'Adding {cnt_dft_calcs} DFT calculations to DB.')

            # Adding calculations to training database and seed_generation database
            for dft_calc_idx, dft_calc in enumerate(dft_calcs):
                # Converting serialized structures to Atoms object.
                if isinstance(dft_calc, dict):
                    dft_calc: Atoms = atl_al_ut.aiida_serialized_ase_dict_to_atoms(
                        dft_calc
                    )

                # Check if structure has `atl_db_index` key. If not, add it using
                # the current length of the database + the current calc idx + 1.
                atl_db_index = dft_calc.info.get(
                    'atl_db_index', len(training_db) + dft_calc_idx + 1
                )
                dft_calc.info['atl_db_index'] = atl_db_index

                # Add information about the current AL step
                dft_calc.info['atl_al_step'] = self.ctx.iteration

                # Add calc_performed tag
                dft_calc.info['calc_performed'] = True

                # Adding the structure to the training database
                seed_gen_db.append(dft_calc)
                training_db.append(dft_calc)

            # Writing the updated databases to file
            ase_write(
                filename=self.ctx.training_db_path,
                images=training_db,
                format='extxyz',
            )
            ase_write(
                filename=self.ctx.seed_db_path,
                images=seed_gen_db,
                format='extxyz',
            )

            self.report('Database files updated.')

            tmp_folder_path: Path = self.ctx.results_dir / 'run_tmp_data'

            # Saving current step database
            curr_it_db_path = (
                tmp_folder_path / f'train_db_it_{self.ctx.iteration}.xyz.xz'
            )
            ase_write(
                filename=curr_it_db_path,
                images=seed_gen_db,
                format='extxyz',
            )

            # Removing teporary files from the AL loop step.
            # TODO: Can't use Path.walk() here as it's not available in Python 3.9
            # this will be added if the library is updated to Python 3.12
            for _, _, files in os.walk(top=tmp_folder_path):
                for file in files:
                    if 'md_seed_structures' not in file:
                        (tmp_folder_path / file).unlink(missing_ok=True)

        self.report(
            f'Iteration {self.ctx.iteration}: '
            f'seed_gen_db {len(seed_gen_db)}, '
            f'training_db: {len(training_db)} entries'
        )

    def get_al_loop_break_conditions(self):
        """
        Evaluate and set conditions to potentially break the active learning loop.

        This function checks for specific conditions that might warrant terminating the
        active learning (AL) loop early:

        - Gathers `stop_md_seed_no_disagreement` from the outputs of the inner workchain
          and stores it in the workchain's context. If this is True, the workchain will
          stop.
        - Checks whether all structures have been removed from the seed generation
          database (indicating no further candidates for evaluation). If this is True,
          the workchain will stop.

        The results of these checks are stored in the workflow's context.
        """
        # Sending empty seed_gen_db flag to context
        seed_gen_db = atl_al_ut.load_database(self.ctx.inputs.seed_db_path)
        if len(seed_gen_db) == 0:
            self.ctx.seed_gen_db_all_structs_removed = orm.Bool(True)
        else:
            self.ctx.seed_gen_db_all_structs_removed = orm.Bool(False)

    def setup(self):
        """Call `BaseRestartWorkChain` setup and create input dict in self.ctx.inputs.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain`
        to submit the process in the internal loop.
        """
        self.report('Starting Workchain setup...')
        super().setup()

        # Update the iteration counter if resuming from a previous run
        if self.inputs.resume_dict:
            self.report(
                'Resuming from previous run, '
                f'{self.inputs.resume_dict.get("prev_workchain_uuid", "unknown id")} '
                'stopped at iteration: '
                f"'{self.inputs.resume_dict['last_iteration']}'."
            )
            self.ctx.iteration = self.inputs.resume_dict['last_iteration']

        # Setting loop continue condition to True initially
        self.ctx.loop_continue_condition = True

        self.ctx.inputs = self.exposed_inputs(
            SimpleActiveLearningWorkChain, 'active_learning'
        )

        # Creating aiida orm.Group to store all calculations
        ctime = time.strftime('%Y%m%dT%H%M%S')
        seed_group = orm.Group(
            label=f'{self.inputs.active_learning.run_name.value}_train_md_seed_{ctime}'
        )
        seed_group.store()
        self.ctx.inputs.train_seed_group = seed_group.uuid
        self.report(f"Created AiiDA group: '{self.ctx.inputs.train_seed_group}'.")

        # Providing current iteration to children workchain.
        self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        # Setting conditionals to always run the first iteration of the
        # active learning loop.
        self.ctx.stop_md_seed_no_disagreement = orm.Bool(False)
        self.ctx.seed_gen_db_all_structs_removed = orm.Bool(False)

        # Adding database paths to inputs
        self.ctx.inputs.seed_db_path = str(self.ctx.seed_db_path)
        self.ctx.inputs.debug_mode = self.inputs.active_learning.debug_mode
        self.ctx.inputs.log_path = str(self.inputs.log_path.value)
        self.ctx.inputs.training_db_path = str(self.ctx.training_db_path)

        # Eval test database inputs
        self.ctx.inputs.test_db_file = self.ctx.test_db_file
        self.ctx.inputs.eval_test_db_settings = self.ctx.test_settings
        self.ctx.inputs.use_test_db = self.ctx.use_test_db
        self.ctx.inputs.test_db_eval_results = orm.Dict({})

        # Set calculation job working directory as an extra to keep track of it
        self.node.base.extras.set(
            'atl_working_directory', self.ctx.inputs.atl_working_directory.value
        )
        self.node.base.extras.set('atl_version', str(get_atl_version_info()[0]))

        # Show url for ntfy.sh if enabled
        if hasattr(self.inputs.active_learning, 'enable_ntfysh') and hasattr(
            self.inputs.active_learning, 'ntfysh_topic'
        ):
            self.ctx.inputs.enable_ntfysh = self.inputs.active_learning.enable_ntfysh
            self.ctx.inputs.ntfysh_topic = self.inputs.active_learning.ntfysh_topic
            requests.post(
                f'https://ntfy.sh/{self.ctx.inputs.ntfysh_topic.value}',
                data=(
                    f'Starting Active Learning Loop - {self.node.pk}: '
                    f'{self.inputs.active_learning.run_name.value}'
                ),
            )
        else:
            self.ctx.inputs.enable_ntfysh = orm.Bool(False)
            self.ctx.inputs.ntfysh_topic = orm.Str('')

        # Initialize safeguard setttings
        self.ctx.safeguard_check_done = False
        self.ctx.safeguard_attempted = False

        self.report('Workchain setup finished.')

    def check_resume_mode(self):
        # If resuming, we need to check if the last iteration is the same as the
        # current iteration. If it is, we keep resume mode for this first resume
        # iteration. Otherwise this means that resume mode has run for one step,
        # and we can disable it.
        self.ctx.resume_mode = self.ctx.inputs.al_start_mode.value == 'resume'
        if self.ctx.resume_mode:
            curr_iteration = self.ctx.iteration
            prev_iteration = self.inputs.resume_dict.get('last_iteration')
            is_first_resume_step = curr_iteration == prev_iteration

            if not is_first_resume_step:
                self.ctx.inputs.al_start_mode = orm.Str('normal')

    def get_input_report(self):
        console = Console(color_system='truecolor')

        input_dict = {}
        for i in self.ctx.inputs:
            if i != 'metadata':
                if self.ctx.inputs[i]:
                    if isinstance(self.ctx.inputs[i], orm.Dict):
                        input_dict[i] = {}
                        curr_dict = self.ctx.inputs[i].get_dict()
                        for key, val in curr_dict.items():
                            if key not in ['metadata', 'options', 'prepend_text']:
                                input_dict[i][key] = val
                    elif isinstance(self.ctx.inputs[i], orm.List):
                        input_dict[i] = self.ctx.inputs[i].get_list()
                    elif isinstance(
                        self.ctx.inputs[i], (orm.Int, orm.Bool, orm.Float, orm.Str)
                    ):
                        input_dict[i] = self.ctx.inputs[i].value
                    else:
                        input_dict[i] = self.ctx.inputs[i]
                else:
                    input_dict[i] = None

        # Hacky way of logging the inputs to both the console and the log file
        # in different formats.
        aiida_logger = logging.getLogger('aiida')
        ini_level = aiida_logger.level
        aiida_logger.level = 20
        aiida_logger.handlers[-1].setLevel(23)
        aiida_logger.log(msg=f'Active Learning Inputs: \n{input_dict}', level=20)
        aiida_logger.level = ini_level

        print()
        inputs = Panel(Pretty(input_dict), title='Active Learning Inputs')
        console.print(inputs)

    def check_al_loop_conditions(self) -> bool:
        """
        Evaluate conditions to determine whether to continue the active learning loop.

        This method assesses multiple conditions to decide if the active learning (AL)
        loop should continue. It considers whether the maximum number of iterations
        has been reached, and the following two criteria:

         - Depletion of the seed generation database. (if True: stop)

         - No disagreement among predictions for an entire MD seed. However,
           this condition can optionally be overriden if the user enables the
           `XXXXXXXXXX` option, which forces the AL Loop to continue if the sampler
           model validation or test performance is above a certain threshold,
           which can be configured. (if True: stop)

        The function updates the workchain context with the iteration status and
        generates reports based on the evaluation of these conditions.

        Returns
        -------
        orm.Bool
            A boolean value indicating whether the AL loop should continue. Returns
            `True` if conditions are met for another iteration; otherwise,
            returns `False`.

        Notes
        -----
        The method uses `self.ctx.is_finished`, `self.ctx.iteration`,
        `self.inputs.active_learning.max_iterations.value`,
        `self.ctx.stop_md_seed_no_disagreement.value`,
        and `self.ctx.seed_gen_db_all_structs_removed.value`
        """
        if hasattr(self.ctx, 'stop_al_loop_error') and self.ctx.stop_al_loop_error:
            self.report(
                f'Last step ({self.ctx.last_workchain_completed.pk}) '
                'did not finish correctly. Stopping AL Loop.'
            )
            return False

        # Checking for the stop criteria dict
        if hasattr(
            self.inputs.active_learning, 'stopping_criteria_settings'
        ) and isinstance(
            self.inputs.active_learning.stopping_criteria_settings, orm.Dict
        ):
            stop_criteria = (
                self.inputs.active_learning.stopping_criteria_settings.get_dict()
            )

        # Getting the specified trheshold type by the user, if any
        stop_type = stop_criteria.get('stopping_threshold_type')
        if stop_type is not None:
            # Getting the specified threshold value by the user from the
            # active learning
            # TODO
            ...

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

        # Check if we need to stop the loop and report accordingly
        if self.ctx.stop_md_seed_no_disagreement.value:
            self.report('Stopping AL Loop as all predictions agree for a MD seed.')
        elif self.ctx.seed_gen_db_all_structs_removed.value:
            self.report(
                'Stopping AL Loop as seed generating database has been depleted.'
            )
        else:
            # This condition avoids printing a confusing message during first
            # the first loop iteration
            if self.ctx.iteration != 0:
                self.report(
                    f'Proceeding with iteration-{self.ctx.iteration + 1} of AL Loop '
                    'as stopping conditions not met.'
                )
            self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        self.ctx.loop_continue_condition = continue_cond

        return continue_cond

    def should_run_safeguard(self) -> bool:
        self.logger.debug('Checking if safeguard is enabled...')

        # Loading the settings file again to get updated settings
        settings_path = Path(self.ctx.inputs.toml_file.value)
        if settings_path.exists:
            current_settings = atl_al_ut.read_toml_settings(
                settings_file=self.ctx.inputs.toml_file.value
            )
        else:
            current_settings = {}

        # # Loading computer and removing it from the input dictionary
        if current_settings:
            safeguard_settings = current_settings.get('safeguard', {})

        # Get 'enable' setting from safeguard settings
        self.ctx.safeguard_enabled = safeguard_settings.get('enable', False)
        self.report(f'Is safeguard enabled: {self.ctx.safeguard_enabled}')

        # Checking if safeguard has not been attempted yet
        self.logger.debug('Checking if safeguard has not been attempted yet...')
        self.report(f'Safeguard attempted: {self.ctx.safeguard_attempted}')
        condition = self.ctx.safeguard_enabled and not self.ctx.safeguard_attempted
        self.report(f'Should run safeguard check: {condition}')

        return condition

    def should_run_according_to_safeguard_check(self) -> bool:
        """
        Decide whether to continue the AL loop based on the safeguard check.

        The loop will continue if the safeguard check has not been completed or has
        passed correctly in the previous attempt, and the maximum number of iterations
        has not been reached.

        Returns
        -------
        bool
            True if the loop should continue, False otherwise.
        """
        # Check maximum iterations
        max_iterations = self.inputs.active_learning.max_iterations.value
        below_max_iterations = self.ctx.iteration < max_iterations

        # Check previous AL loop error status
        if hasattr(self.ctx, 'stop_al_loop_error'):
            al_error_stop = self.ctx.stop_al_loop_error.value
        else:
            al_error_stop = False

        self.report(f'Is safeguard check complete: {self.ctx.safeguard_check_done}.')
        self.report(
            f'Is current step below maximum allowed step? {below_max_iterations}.'
        )
        self.report(f'Did previous AL stop with errors? {al_error_stop}.')

        self.report(f'Safeguard attempted: {self.ctx.safeguard_attempted}')

        # Loading the settings file again to get updated settings
        settings_path = Path(self.ctx.inputs.toml_file.value)
        if settings_path.exists:
            current_settings = atl_al_ut.read_toml_settings(
                settings_file=self.ctx.inputs.toml_file.value
            )
        else:
            current_settings = {}

        # # Loading computer and removing it from the input dictionary
        if current_settings:
            safeguard_settings = current_settings.get('safeguard', {})

        # Get 'enable' setting from safeguard settings
        self.ctx.safeguard_enabled = safeguard_settings.get('enable', False)

        # Decide whether to continue based on safeguard settings
        # If safeguard is disabled, ignore this
        match self.ctx.safeguard_enabled:
            # If safeguard is disabled
            case False:
                if al_error_stop:
                    should_run = False
                    self.report(
                        'Safeguard is not enabled, and errors found: stopping AL loop.'
                    )
                else:
                    should_run = (
                        below_max_iterations
                        and not al_error_stop
                        and self.ctx.loop_continue_condition
                    )
                    self.report(
                        'Safeguard is not enabled, and no errors found, should run:'
                        f' {should_run}'
                    )

            # If safeguard is enabled
            case True:
                # This will be True if the safeguard check allows to continue the loop
                should_run = (
                    not self.ctx.safeguard_check_done
                    and below_max_iterations
                    and not al_error_stop
                    and not self.ctx.safeguard_attempted
                    and self.ctx.loop_continue_condition
                )
                self.report(
                    f'Safeguard check decision on whether to continue '
                    f'the loop: {should_run}.'
                )

        if should_run is False:
            self.report('Stopping AL Loop due to passed safeguard check!')
            self.ctx.safeguard_check_done = True

        return should_run

    def compile_safeguard_model(self):
        """Optionally pre-compile the sampler model for the safeguard node.

        No-op unless the backend implements ``MLIPModelCompiler``. Result is
        collected in :meth:`gather_safeguard_compiled` and used by the safeguard
        MD calc; inference falls back to on-demand compilation otherwise.
        """
        from atlas.active_learning.backends import get_backend
        from atlas.active_learning.backends._base import MLIPModelCompiler

        self.ctx.safeguard_compiled_model = None

        if getattr(self.ctx, 'stop_al_loop_error', False) is True:
            return

        current_settings = atl_al_ut.read_toml_settings(
            settings_file=self.ctx.inputs.toml_file.value
        )
        backend_name = current_settings.get('mlip', {}).get('training_backend', 'mace')
        try:
            backend = get_backend(backend_name)
        except ValueError:
            return
        if not isinstance(backend, MLIPModelCompiler):
            return
        if not current_settings.get('mlip', {}).get('precompile_for_inference', True):
            return

        safeguard_settings = current_settings.get('safeguard', {})
        if not safeguard_settings.get('metadata', {}).get('computer'):
            return

        sampler_model = self.ctx.last_workchain_completed.outputs['m0_model_file']
        device = (
            safeguard_settings.get('md', {}).get('parameters', {}).get('device', 'cpu')
        )
        builder = _build_compile_model_builder(
            workchain=self,
            model_node=sampler_model,
            code_settings=safeguard_settings,
            current_settings=current_settings,
            device=device,
            backend_name=backend_name,
        )
        future = self.submit(builder)
        self.to_context(safeguard_compile_calc=future)
        self.report(f'Submitted safeguard model-compile calc ({future.pk}).')

    def gather_safeguard_compiled(self):
        """Store the compiled safeguard model if its compile calc succeeded."""
        calc = getattr(self.ctx, 'safeguard_compile_calc', None)
        if (
            calc is not None
            and calc.is_finished_ok
            and 'compiled_model' in calc.outputs
        ):
            self.ctx.safeguard_compiled_model = calc.outputs.compiled_model
            self.report(f'Safeguard model compiled (calc {calc.pk}).')
        elif calc is not None:
            self.report(
                f'WARNING: safeguard model compile did not succeed '
                f'(calc {calc.pk}); safeguard will compile on demand.'
            )

    def run_safeguard_check(self):
        self.report('Running safeguard checks...')

        # Skipping safeguard if there was an error in the previous step
        if (
            hasattr(self.ctx, 'stop_al_loop_error')
            and self.ctx.stop_al_loop_error is True
        ):
            self.report(
                f'Last step ({self.ctx.last_workchain_completed.pk}) '
                'did not finish correctly. '
                'Skipping safeguard checks and stopping AL Loop.'
            )
            self.ctx.safeguard_check_done = True
            return False

        # Run safeguard logic
        self.safeguard_check_md()

    def safeguard_check_md(self):
        self.report('Running MD for safeguard check...')

        # Load sampler model (prefer a pre-compiled artifact for this node).
        last_wk = self.ctx.last_workchain_completed
        sampler_model: orm.SinglefileData = last_wk.outputs['m0_model_file']
        if getattr(self.ctx, 'safeguard_compiled_model', None) is not None:
            sampler_model = self.ctx.safeguard_compiled_model

        # Get types of all called calculations
        called_types = [nod.process_label for nod in last_wk.called]

        # Get training errors from last workchain. Since workchains can be resumed
        # there might not be any training calculations in the last workchain,
        # so we read the values from the MD calculations instead
        # md_calc = last_wk.called[called_types.index('ProcessMDSeedStructCalculation')]
        m_rmse_e: orm.Float = last_wk.outputs['m0_rmse_e']
        m_rmse_f: orm.Float = last_wk.outputs['m0_rmse_f']

        # Load extrapolation files if available in last workchain
        try:
            descriptor_calc = last_wk.called[
                called_types.index('GetDescriptorsCombinedCalculation')
            ]
        except ValueError:
            if (
                hasattr(self.inputs.active_learning, 'load_descriptor_calc')
                and self.inputs.active_learning.load_descriptor_calc != ''
            ):
                descriptor_calc = orm.load_node(
                    self.inputs.active_learning.load_descriptor_calc.value
                )
            else:
                # This will happen if 'check_extrapolation_type' is 'none' in the
                # user settings.
                descriptor_calc = None

        # Keep default values as None if not available
        desc_max_arr = None
        concave_hull = None
        autoencoder_model = None
        desc_min_arr = None

        if descriptor_calc:
            try:
                desc_max_arr: orm.ArrayData = descriptor_calc.outputs['descriptor_max']
                desc_min_arr: orm.ArrayData = descriptor_calc.outputs['descriptor_min']
                if hasattr(descriptor_calc.outputs, 'autoencoder_model'):
                    autoencoder_model: orm.SinglefileData = descriptor_calc.outputs[
                        'autoencoder_model'
                    ]
                if hasattr(descriptor_calc.outputs, 'concave_hull'):
                    concave_hull: orm.ArrayData = descriptor_calc.outputs[
                        'concave_hull'
                    ]
            except ValueError:
                pass

        # Loading the settings file again to get updated settings
        settings_path = Path(self.ctx.inputs.toml_file.value)
        if settings_path.exists:
            current_settings = atl_al_ut.read_toml_settings(
                settings_file=self.ctx.inputs.toml_file.value
            )
        else:
            current_settings = {}

        # # Loading computer and removing it from the input dictionary
        if current_settings:
            safeguard_settings = current_settings.get('safeguard', {})
            ignore_container = safeguard_settings.get('ignore_container', False)
            metadata_dict = safeguard_settings.get('metadata', {})
            resc_dict = metadata_dict.get('options', {}).get('resources', {})
            num_threads = resc_dict.get('num_cores_per_mpiproc')

            # In case the number of threads is not set in the first try
            # try to get an additional key. If it does not work, set it to 1.
            if num_threads is None:
                num_threads = resc_dict.get('tot_num_mpiprocs', 1)

            prepend_text_conf = metadata_dict.get('prepend_text', '')
        else:
            # Fallback settings
            metadata_dict = {}
            num_threads = 1
            prepend_text_conf = ''
            ignore_container = False

        options_dict = metadata_dict['options']
        computer = orm.load_computer(metadata_dict.get('computer'))

        # Getting container settings
        containerized = False
        container_dict = self.inputs.active_learning.container_settings.get_dict()
        if container_dict.get('use_container'):
            containerized = container_dict.get('use_container', False)
        if ignore_container is True:
            containerized = False

        if containerized:
            image_name = container_dict.get('image_name', '')
            engine_command = container_dict.get('engine_command', '')
            code = get_or_create_containerized_code(
                label='atl_run_safeguard_md',
                computer=computer,
                image_name=image_name,
                engine_command=engine_command,
                filepath_executable='atl_run_safeguard_md.py',
            )
        else:
            # Get portable code
            code_path = Path(
                f'{ATL_ROOT_DIR}/active_learning/safeguard/safeguard_scripts'
            )

            code = get_or_create_portable_code(
                label='atl_run_safeguard_md',
                filepath_files=code_path,
                filepath_executable='atl_run_safeguard_md.py',
            )

        # Select and generate structures for safeguard
        target_structure_mode = safeguard_settings.get('target_structure_mode', 'base')

        # Load target structures based on the selected mode
        if target_structure_mode == 'base':
            self.report("Using 'base' structures for safeguard processing.")

            database_training = ase_read(
                filename=self.inputs.active_learning.init_db_path.value,
                format='extxyz',
                index=':',
            )

            # Load MD input structures
            target_md_structs: list[Atoms] = [
                struct for struct in database_training if struct.info.get('base')
            ]

            supercell_vec = safeguard_settings.get(
                'base_struct_supercell_size', [3, 3, 3]
            )

            if supercell_vec:
                target_md_structs = [
                    struct.repeat(supercell_vec) for struct in target_md_structs
                ]

        elif target_structure_mode == 'target':
            # Get target structures from the user settings, which can either contain
            # a path to a database file or a list of structure UUIDs.
            self.report("Using 'target' structures for safeguard processing.")
            struct_target_list: list = safeguard_settings.get('struct_target_list', [])

            target_md_structs: list[Atoms] = []

            for target in struct_target_list:
                if isinstance(target, str) and Path(target).exists():
                    # Load structures from database file
                    loaded_structs = ase_read(
                        filename=target,
                        format='extxyz',
                        index=':',
                    )
                    target_md_structs.extend(loaded_structs)
                elif isinstance(target, str):
                    # Load structure by UUID from the training database
                    database_training = ase_read(
                        filename=self.inputs.active_learning.init_db_path.value,
                        format='extxyz',
                        index=':',
                    )
                    for struct in database_training:
                        if struct.info.get('atl_id') == target:
                            target_md_structs.append(struct)

        self.report(
            f'Starting submission of {len(target_md_structs)} '
            'structures for safeguard...'
        )

        calc_count = 0
        for target_struct in target_md_structs:
            # Creating builder for safeguard code
            safeguard_calc = CalculationFactory('atl-safeguard-md')
            sg_builder = safeguard_calc.get_builder()
            sg_builder.code = code

            sg_builder.metadata.computer = computer
            sg_builder.metadata.label = (
                f'process_{target_struct.info.get("struct_name", "unknown")}'
            )
            sg_builder.metadata.description = (
                'Processing structure using MD as a safeguard.'
            )

            sg_builder.metadata.options = options_dict
            sg_builder.metadata.options.parser_name = 'atl-safeguard-md-parser'

            # Pass environment setup dynamically so TOML changes take effect
            # without recreating the code node.
            container_prepend = (
                container_dict.get('prepend_text', '') if containerized else ''
            )
            sg_builder.metadata.options.prepend_text = (
                prepend_text_conf
                + '\n'
                + container_prepend
                + '\nexport PATH=$PATH:.'
                + f'\nexport OMP_NUM_THREADS={num_threads}'
            )

            sg_builder.settings_file_pth = self.ctx.inputs.toml_file

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename='-',
                    format='extxyz',
                    images=target_struct,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            target_structure_file = orm.SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename='md_db.xyz',
            )
            target_structure_file.store()

            sg_builder.md_structure = target_structure_file

            sg_builder.sampler_model = sampler_model

            sg_builder.m_rmse_e = m_rmse_e
            sg_builder.m_rmse_f = m_rmse_f

            if autoencoder_model:
                sg_builder.autoencoder_model = autoencoder_model

            if concave_hull:
                sg_builder.concave_hull = concave_hull

            if desc_max_arr is not None:
                sg_builder.desc_max_arr = desc_max_arr
                sg_builder.desc_min_arr = desc_min_arr

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `atl_calc_limit` is a custom property set with:
            # computer.set_property(name='atl_calc_limit', value=366)
            calc_limit = sg_builder.metadata.computer.metadata.get('atl_calc_limit', 0)

            if calc_limit != 0:
                atl_al_ut.aiida_wait_submit(
                    builder=sg_builder,
                    computer=computer,
                    calc_count=calc_count,
                )
            future = self.submit(sg_builder)

            if target_struct.info.get('aiida_uuid'):
                future.base.extras.set(
                    'unique_id_old', target_struct.info['aiida_uuid']
                )
            if target_struct.info.get('atl_id'):
                future.base.extras.set('unique_id', target_struct.info['atl_id'])

            future.base.extras.set('atl_db_index', target_struct.info['atl_db_index'])
            # future.base.extras.set("md_temperature", temp_val)
            self.to_context(process_safeguard_results=append_(future))

        if len(target_md_structs) > 0:
            self.report(
                f'Submission done. Running MD for {len(target_md_structs)}'
                ' safeguard structures...'
            )
        del target_md_structs

    def parse_safeguard_check_results(self):
        """Check safeguard output, setting flags to stop/continue the AL Loop."""
        self.report('Checking safeguard results...')

        if hasattr(self.ctx, 'process_safeguard_results'):
            # Get all of the processed structures
            processed_structures = self.ctx.process_safeguard_results

            tot_num_safeguard_calcs = len(processed_structures)

            if hasattr(self.inputs, 'enable_ntfysh'):
                requests.post(
                    f'https://ntfy.sh/{self.inputs.ntfysh_topic.value}',
                    data=(
                        f'{len(processed_structures)} MD calculations complete for'
                        f' step {self.inputs.al_loop_iteration.value + 1}'
                        f' - {self.node.pk}.\n'
                    ),
                )

            safeguard_failed_ids = []
            safeguard_passed_ids = []
            safeguard_errored_ids = []

            proc_calcjob: orm.CalcJobNode
            for proc_calcjob in processed_structures:
                # Skip calculation if it didn't finish correctly.
                # Skipped calculations count as failed checks.
                if proc_calcjob.exit_status != 0:
                    self.report(
                        'Removing struct. processing that finished with errors '
                        f'(pk: {proc_calcjob.pk}).'
                    )
                    safeguard_failed_ids.append(
                        proc_calcjob.base.extras.all.get('unique_id')
                    )
                    safeguard_errored_ids.append(
                        proc_calcjob.base.extras.all.get('unique_id')
                    )
                    continue

                # Getting unique_id from extras
                orig_str_uuid = proc_calcjob.base.extras.all.get('unique_id')

                # Loading extrapolating structures
                extrap_strc_file: orm.SinglefileData = (
                    proc_calcjob.outputs.extrapolating_structures
                )
                with extrap_strc_file.as_path() as file_path:
                    extrap_structs = ase_read(
                        filename=file_path, format='extxyz', index=':'
                    )

                for struct in extrap_structs:
                    struct.info['atl_md_node'] = proc_calcjob.uuid
                    struct.info['atl_db_index'] = proc_calcjob.base.extras.get(
                        'atl_db_index'
                    )

                # If no extrapolating structures, the safeguard check is successful
                if len(extrap_structs) == 0:
                    safeguard_passed_ids.append(orig_str_uuid)
                # If there are extrapolating structures, the safeguard check is failed.
                else:
                    safeguard_failed_ids.extend(extrap_structs)

                self.logger.debug(
                    f'Trajectory from {proc_calcjob.pk} has {len(extrap_structs)} '
                    'out of domain frames.'
                )

            self.logger.debug(f'Errored structures count: {len(safeguard_errored_ids)}')
            self.logger.debug(f'Failed structures count: {len(safeguard_failed_ids)}')
            self.logger.debug(f'Passed structures count: {len(safeguard_passed_ids)}')

            self.ctx.safeguard_attempted = True

            # Set flags based on safeguard results
            if len(safeguard_errored_ids) >= tot_num_safeguard_calcs:
                self.report('All safeguard calculations errored. Stopping AL Loop.')
                self.ctx.safeguard_check_done = True
                self.ctx.safeguard_attempted = True
                del self.ctx.process_safeguard_results
            elif len(safeguard_failed_ids) > 0:
                self.report(
                    f'Safeguard check failed for {len(safeguard_failed_ids)}'
                    ' structures. Continuing AL Loop...'
                )

                # Setting flag for continuing the AL loop
                self.ctx.safeguard_check_done = False
                self.ctx.safeguard_attempted = False

                # Adding failed structures to the seed generation database
                seed_gen_db: list = atl_al_ut.load_database(self.ctx.seed_db_path)

                for failed_struct in safeguard_failed_ids:
                    if isinstance(failed_struct, Atoms):
                        seed_gen_db.append(failed_struct)

                ase_write(
                    filename=self.ctx.seed_db_path,
                    images=seed_gen_db,
                    format='extxyz',
                )

                # Cleaning up context
                del self.ctx.process_safeguard_results

            elif len(safeguard_failed_ids) == 0:
                self.report(
                    f'Safeguard check passed for all ({tot_num_safeguard_calcs}) '
                    'structures. Stopping AL Loop.'
                )
                self.ctx.safeguard_check_done = True

        else:
            self.report('No safeguard results found. Proceeding without safeguard.')
            self.ctx.safeguard_check_done = False
            self.ctx.safeguard_attempted = True

    def get_seed_structures(self):
        """
        Dispatcher method to get seed structures based on the active learning mode.

        For MD mode: calls get_md_seed()
        For data_reduction mode: calls get_data_reduction_seed()
        """
        if self.ctx.al_mode == 'data_reduction':
            self.get_data_reduction_seed()
        else:
            self.get_md_seed()

    def get_data_reduction_seed(self):
        """
        Filter structures for data reduction mode.

        In data reduction mode, we need to filter out any structures that are
        already in the training database from the seed database. This ensures
        we don't re-select structures that have already been chosen.

        The actual structure selection for training will happen in the
        SimpleActiveLearningWorkChain.
        """
        self.report(
            f'Starting AL Loop iteration {self.ctx.iteration + 1}/'
            f'{self.inputs.active_learning.max_iterations.value}...'
        )
        self.report('Filtering seed database for data reduction mode...')

        # Load current seed database (large database)
        seed_gen_db: list = atl_al_ut.load_database(self.ctx.seed_db_path)

        # Load current training database
        training_db: list = atl_al_ut.load_database(self.ctx.training_db_path)

        # Get IDs of structures already in training database
        training_ids = {struct.info['atl_id'] for struct in training_db}

        # Filter out structures that are already in training database
        filtered_seed_db = [
            struct
            for struct in seed_gen_db
            if struct.info['atl_id'] not in training_ids
        ]

        # Remove structures with config_type == 'IsolatedAtom' if any
        filtered_seed_db = [
            struct
            for struct in filtered_seed_db
            if struct.info.get('config_type') != 'IsolatedAtom'
        ]

        # Update the seed database file with filtered structures
        ase_write(
            filename=self.ctx.seed_db_path,
            format='extxyz',
            images=filtered_seed_db,
        )

        # Update context metadata for the child workchain
        self.ctx.inputs.metadata.description = (
            'Data reduction: Select and train with most informative structures. '
            f'Step: {self.ctx.iteration + 1}'
        )
        self.ctx.inputs.metadata.label = f'DataReduction_Step_{self.ctx.iteration + 1}'

        self.report(
            f'Filtered seed database: {len(seed_gen_db)} -> {len(filtered_seed_db)} '
            f'structures (removed {len(seed_gen_db) - len(filtered_seed_db)} '
            'already in training database)'
        )

        # For data reduction mode, we don't set up MD seed structures here
        # The structure selection for training will happen in the
        # SimpleActiveLearningWorkChain based on the iterative_selection_method
        self.ctx.inputs.current_md_seed_structs_path = ''
        self.ctx.inputs.current_md_seed_structs_idx = []

    def get_md_seed(self):
        """
        Selects a random subset of structures from the seed generation database to
        create a MD seed for the active learning loop.

        This function calculates the number of structures to be included in the MD
        seed based on the specified fraction of the seed generation database's length.
        It then randomly selects and populates the training seed with these structures.

        Returns
        -------
            None. The function updates current_md_seed_structs with the
            selected structures.
        """
        self.report(
            f'Starting AL Loop iteration {self.ctx.iteration + 1}/'
            f'{self.inputs.active_learning.max_iterations.value}...'
        )
        self.report('Getting MD seed...')
        self.ctx.inputs.metadata.description = (
            'Perform MD simulations, evaluate and refine ML models. '
            f'Step: {self.ctx.iteration + 1}'
        )
        self.ctx.inputs.metadata.label = f'Step - {self.ctx.iteration + 1}'

        seed_gen_db: list = atl_al_ut.load_database(self.ctx.seed_db_path)

        # Removing structures with config_type == 'IsolatedAtom'
        # from the seed generation database.
        seed_gen_db = [
            struct
            for struct in seed_gen_db
            if struct.info.get('config_type') != 'IsolatedAtom'
        ]

        # Getting length of the seed generating database
        db_length = len(seed_gen_db)

        # Defining the current md seed size as a function of the amount of structures
        # in the seed generation database
        seed_size = int(self.ctx.inputs.seed_size_frac.value * db_length)

        # Increasing the seed size if it is below the minimum seed size
        if seed_size < self.ctx.min_seed_size:
            seed_size = self.ctx.min_seed_size
            self.report(
                'Set MD seed size to minimum seed size as the calculated seed size was '
                f'below the minimum seed size: {self.ctx.min_seed_size}.'
            )

        # Limit the maximum number of structures to the seed size limit set on the input
        if seed_size > self.ctx.inputs.seed_max_num_structs.value:
            seed_size = int(self.ctx.inputs.seed_max_num_structs.value)
            self.report(f"MD seed size too large: limited to '{seed_size}'.")

        # This should avoid tring to select more structures than available
        if seed_size > db_length:
            seed_size = db_length
            self.report(
                'MD seed size larger than seed generation database. '
                f"Limited size to '{db_length}'."
            )

        # For small databases or percentages, the number of structures might be 0
        # if this happens, make it 1.
        if seed_size < 1:
            seed_size = 1
            self.report('MD seed size is 0. Set to 1.')

        # Get seed selection type
        seed_select_settings = self.ctx.inputs.seed_select_settings[
            'seed_select_settings'
        ]

        # Apply small-cell training-like seed selection if enabled.
        # In order to use small_first, the current iteration number must be
        # less than small_first_max_iter.
        if (
            seed_select_settings['seed_select_type'] == 'small_first'
            and self.ctx.iteration + 1 <= seed_select_settings['small_first_max_iter']
        ):
            self.report("Using 'small_first' seed selection type.")
            seed_select_type = 'small_first'
        else:
            seed_select_type = 'random'

        # Filter the structure database and only use the small structures.
        if seed_select_type == 'small_first':
            max_size = seed_select_settings['small_first_max_size']
            seed_gen_db_small = [s for s in seed_gen_db if len(s) <= max_size]
            if len(seed_gen_db_small) >= seed_size:
                seed_gen_db = seed_gen_db_small
                self.logger.warning(
                    f'Limited seed generation database to small cells'
                    f' (below {max_size}).'
                )
            else:
                self.logger.warning(
                    f'There are not enough structures with less than '
                    f'{max_size} atoms in the given database. '
                    f'There should be at least {seed_size} structures. '
                    "Please, decrease 'small_first_max_size' or remove "
                    "the 'small_first' seed selection mode. "
                    'and try again.'
                )

        # Load algorithm for seed selection
        # Set score for structures in the seed generation database
        # and use it to select the structures for the MD seed.
        seed_ranking_algo_settings = self.ctx.inputs['seed_select_settings'].get(
            'seed_ranking_settings', {}
        )

        seed_ranking_algorithm = seed_ranking_algo_settings.get(
            'seed_ranking_algorithm', 'random'
        )

        # If enabled, use Farthest Point Sampling (FPS) on the descriptors
        # starting from an initially selected structure to get new points.
        if seed_ranking_algorithm == 'descriptor_fps':
            descriptor_fps_settings = seed_ranking_algo_settings.get(
                'descriptor_fps', {}
            )
            self.report('Initializing farthest point sampling ranking...')
            # Select an initial structure depending on the seed ranking algorithm
            # `initial_structure` parameter.
            init_struct_type = descriptor_fps_settings.get(
                'initial_structure', 'random'
            )
            self.report(
                f"Selecting initial structure using '{init_struct_type}' method..."
            )

            # Sorting the seed generation database by the REF_energy so that the
            # lowest energy structure goes first
            if init_struct_type == 'lowest_energy':
                sorted_seed_db = sorted(
                    seed_gen_db, key=lambda x: float(x.info['REF_energy'])
                )
                init_structure_uuid = sorted_seed_db[0].info['atl_id']
            elif init_struct_type == 'random':
                # If the initial structure is random, we will select a random
                # structure from the seed generation database.
                sorted_seed_db = seed_gen_db
                init_structure_uuid = np.random.choice(
                    [s.info['atl_id'] for s in seed_gen_db]
                )

            # If the seed selection algorithm is descriptor_fps, we need to
            # calculate the fingerprints for the structures in the seed generation
            # database.
            self.report('Calculating descriptors for seed generation database...')
            descriptor_settings_dict = (
                self.inputs.active_learning.descriptor_settings.get_dict()
            )

            # Setting the average type for the descriptor to get global descriptors.
            if descriptor_settings_dict.get('descriptor', {}).get('average') is None:
                descriptor_settings_dict.get('descriptor', {})['average'] = 'inner'

            descr_dict, _, _ = atl_al_ut.generate_descriptors(
                database=sorted_seed_db,
                descriptor_type=descriptor_fps_settings.get('descriptor_type', 'soap'),
                descriptor_settings=descriptor_fps_settings.get('descriptor', {}),
            )
            self.report('Done calculating descriptors!')

            # Compute FPS to get the scoring
            self.report('Ranking using farthest point sampling...')
            # Pass the initial structure as a list to 'selected_uuids'
            # n_to_select is the rest of the database (total length - 1)
            scores = atl_al_ut.calculate_fps_scores_descriptor(
                selected_uuids=[init_structure_uuid],
                descriptor_dict=descr_dict,
                n_to_select=len(seed_gen_db) - 1,
            )

            # The FPS function only scores the new structures.
            # We must manually inject the initial structure into the scores dictionary
            # with the highest possible score (1.0) so it sorts to the very top.
            scores[init_structure_uuid] = {'score': 1.0, 'distance': 0.0}
            self.report('Ranking completed!')

            # Add scores to the structures in the seed generation database
            for struct in seed_gen_db:
                # If the structure is not in the scores, set the score to 0
                if struct.info['atl_id'] not in scores:
                    scores[struct.info['atl_id']] = {'score': 0.0, 'distance': 0.0}

                # Add the score to the structure info
                struct.info['md_seed_ranking_score'] = scores[
                    struct.info['atl_id']
                ].get('score', 0.0)

            sorted_seed_db = sorted(
                seed_gen_db,
                key=lambda x: x.info['md_seed_ranking_score'],
                reverse=True,
            )

            # Freeing up memory
            atl_al_ut.flush_system_memory()

        elif seed_ranking_algorithm == 'random':
            # If the seed selection algorithm is random, we will select structures
            # at random from the seed generation database, and we will use a 1 as
            # the score for all structures.
            self.report("Using 'random' seed ranking algorithm.")
            scores = {s.info['atl_id']: 1.0 for s in seed_gen_db}

            # Add scores to the structures in the seed generation database
            for struct in seed_gen_db:
                # If the structure is not in the scores, set the score to 0
                if struct.info['atl_id'] not in scores:
                    scores[struct.info['atl_id']] = 0.0

                # Add the score to the structure info
                struct.info['md_seed_ranking_score'] = scores[struct.info['atl_id']]

            # Apply random shuffling to the seed generation database
            # TODO: Use numpy new random number generator class, not default one.
            sorted_seed_db = seed_gen_db.copy()
            np.random.shuffle(seed_gen_db)

        elif seed_ranking_algorithm == 'boundary':
            ...

        # Selecting first db_length structures from the seed generation database
        # to be used in the training seed.
        # Seed generation database is already sorted by the
        # seed ranking algorithm, so we can just take the first `seed_size` structures.
        selected_structs_idxs = range(seed_size)

        self.ctx.inputs.current_md_seed_structs_idx = list(selected_structs_idxs)

        # The set of random structures selected from the seed generation
        # database to be used in training.
        current_md_seed_structs = []

        # Populating MD seed with the selected random structures
        for idx in selected_structs_idxs:
            seed_struct = sorted_seed_db[idx]
            current_md_seed_structs.append(seed_struct)

        self.report(
            f'Created MD seed with {seed_size} structures '
            f'({(seed_size / db_length) * 100:.1f}% of current database size).'
        )

        # Adding current train seed to the context
        current_MD_seed_serialized = []
        for curr_s in current_md_seed_structs:
            curr_s = atl_al_ut.serialize_ase(curr_s)
            current_MD_seed_serialized.append(curr_s)

        current_md_seed_structs = current_MD_seed_serialized

        # Saving the current md seed into the result directory as a file.
        self.ctx.results_dir = atl_al_ut.get_results_dir_path(
            result_dir_path=self.inputs.active_learning.results_dir.value,
            node=self.node,
        )
        self.ctx.current_md_seed_structs_path = (
            f'{self.ctx.results_dir}/run_tmp_data/md_seed_structures-'
            f'{self.ctx.iteration}.pkl'
        )

        # Providing current seed size as a variable
        self.ctx.inputs.current_seed_size = orm.Int(seed_size)

        with open(self.ctx.current_md_seed_structs_path, 'wb') as seed_file:
            pickle.dump(current_md_seed_structs, seed_file)
        self.ctx.inputs.current_md_seed_structs_path = (
            self.ctx.current_md_seed_structs_path
        )

    def results_final(self):
        """
        Finalize the results at the end of the workchain.

        This method prepares and returns the final training database at the
        conclusion of the workchain. It serializes the structures within the training
        database to a format compatible with AiiDA storage and subsequent processing.
        The serialized structures are then used to prepare the final training database,
        which is outputted from the workchain. This signifies the completion of the
        workchain and the availability of the processed training data for further use.
        """
        self.report('Returning final results...')

        # Storing final database as a orm.SinglefileData object
        train_db = atl_al_ut.prepare_output_final_training_db(
            training_db_path=self.ctx.inputs.training_db_path
        )

        self.out('final_training_db', train_db)

        if not hasattr(self.ctx, 'stop_al_loop_error'):
            self.logger.error(
                f"'stop_al_loop_error' could not be found in last iteration. "
                f"Most likely, workchain '{self.node.pk}' crashed..."
            )
        elif (
            hasattr(self.ctx, 'stop_al_loop_error')
            and self.ctx.stop_al_loop_error is False
        ):
            # Returning final model as orm.SinglefileData object
            final_model_singlefile = self.ctx.last_workchain_completed.outputs[
                'm0_model_file'
            ]
            self.out(
                'final_model_file',
                self.ctx.last_workchain_completed.outputs['m0_model_file'],
            )

            target_file_name = (
                f'al_loop_{self.inputs.active_learning.run_name.value}.model'
            )
            target_file_path = self.ctx.curr_run_results_dir / target_file_name

            with (
                final_model_singlefile.open(mode='rb') as source,
                open(target_file_path, mode='wb') as target,
            ):
                shutil.copyfileobj(source, target)

            self.report('Workchain completed correctly!')
        else:
            self.logger.error(f"Workchain '{self.node.pk}' exited with errors...")
