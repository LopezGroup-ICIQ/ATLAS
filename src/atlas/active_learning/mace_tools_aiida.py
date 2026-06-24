#!/usr/bin/env python
"""AiiDA plugin for MACE calculations.

MACE-specific CalcJobs and Parsers have been extracted to
``atlas.active_learning.backends.mace.calcjobs`` as part of the MLIP-agnostic
refactoring. They are re-exported here for backward compatibility and to
preserve existing AiiDA entry points.
"""

import json
import pickle
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from tomlkit import dumps as tomlkit_dumps

# Re-export MACE-specific classes for backward compatibility.
# AiiDA entry points in pyproject.toml reference this module.
from atlas.active_learning.backends.mace.calcjobs import (  # noqa: F401
    CheckMACECommiteeResultsCalculationParser,
    CheckMACECommitteeResultsCalculation,
    EvaluateMACEConfigsCalculation,
    EvaluateMACEConfigsCalculationParser,
    GetMACEDescriptorsCalculation,
    GetMACEDescriptorsCalculationParser,
    LAMMPSMACERawParser,
    RunMDCalculationGPULAMMPSMACE,
    TrainMACEModelCalculation,
    TrainMACEModelCalculationParser,
    prepare_cli_args_mace,
)
from atlas.workflows.datatypes import image_types as atl_img


# atl-process-md-seed-struct
class ProcessMDSeedStructCalculation(CalcJob):
    """
    Launch a calculation to process a structure in an AL Loop step.

    This CalcJob will process the output of a structure calculation
    by doing a MD simulation using the user provided settings, followed
    by checking all frames for extrapolation using several possible methods.

    Parameters
    ----------
    md_structure : orm.SinglefileData
        File containing the structure to be used for the MD, in the extxyz format.
    commitee_models : PortNamespace
        A namespace to hold an arbitrary number of committee MACE potentials.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.
    m_rmse_e : orm.Float
        Validation RMSE of the best model for the energy, in meV / atom.
    m_rmse_f : orm.Float
        Validation RMSE of the best model for the forces, in meV / A.
    concave_hull : orm.ArrayData, optional
        Array containing the concave hull to be used for the extrapolation check.
    desc_max_arr : orm.ArrayData
        Array containing the maximum values for the descriptors.
    desc_min_arr : orm.ArrayData
        Array containing the minimum values for the descriptors.
    settings_file_pth : orm.Str
        Path to the ATL settings file in the .toml format.

    Outputs
    -------
    extrapolating_structures : orm.SinglefileData
        File containing all structures that were found to be extrapolating.
        Uses the extxyz format.
    extrapolation_plot: atl_img.ImagePNGData
        File containing a visualization of the extrapolation check and latent
        space boundaries

    Exit Codes
    ----------
    420 : ERROR_INVALID_OUTPUT
        Structure could not be processed.
    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications for the CalcJob."""
        super().define(spec)
        # Namespace that will hold an arbitrary number of committee MACE potentials
        spec.input_namespace(
            'commitee_models',
            dynamic=True,
            valid_type=orm.SinglefileData,
            non_db=True,
        )
        spec.input(
            'best_model_name',
            valid_type=orm.Str,
            help='Name of the best model.',
            required=True,
        )
        spec.input(
            'md_structure',
            valid_type=orm.SinglefileData,
            help=(
                'File containing the structure to be used for the MD,'
                'in the extxyz format.'
            ),
            required=True,
            # non_db=True,
        )
        spec.input(
            'autoencoder_model',
            valid_type=orm.SinglefileData,
            help='File containing the autoencoder model.',
            required=False,
            # non_db=True,
            default=None,
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'm_rmse_e',
            valid_type=orm.Float,
            help='Validation RMSE of the best model for the energy, in meV / atom.',
        )

        spec.input(
            'm_rmse_f',
            valid_type=orm.Float,
            help='Validation RMSE of the best model for the forces, in meV / A.',
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'concave_hull',
            valid_type=orm.List,
            help=('List containing several concave hulls as lists of tuples.'),
            required=False,
            # non_db=True,
            default=None,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'desc_max_arr',
            valid_type=orm.ArrayData,
            help=('Array containing the maximum values for the descriptors'),
            required=True,
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'desc_min_arr',
            valid_type=orm.ArrayData,
            help=('Array containing the minimum values for the descriptors.'),
            required=True,
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'settings_file_pth',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the .toml format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'curr_active_learning_step',
            valid_type=orm.Int,
            help='Current active learning iteration step.',
        )
        spec.output(
            'extrapolating_structures',
            valid_type=orm.SinglefileData,
            help=(
                'File containing all structures that were found to be extrapolating. '
                'Uses the the extxyz format.'
            ),
        )
        spec.output(
            'extrapolation_plot',
            valid_type=(atl_img.ImagePNGData, None),
            help=('File containing a figure showing the extrapolation results.'),
            required=False,
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            "Structure '{node_id}' could not be processed.",
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Create E and F RMSE array
        rmse_arr = np.array([self.inputs.m_rmse_e.value, self.inputs.m_rmse_f.value])
        # Create a named temporary file using tmpfile
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.npy', prefix='atl_process_md-'
        ) as f:
            np.save(f.name, rmse_arr)
            folder.insert_path(
                src=f.name,
                dest_name='rmse_arr.npy',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        # Copying structure to use for the MD
        md_structure: orm.SinglefileData = self.inputs.md_structure
        with md_structure.as_path() as md_struct_path:
            folder.insert_path(
                src=md_struct_path,
                dest_name='curr_structure.xyz',
            )

        # Loading settings to include current active learning step
        toml_settings = self.inputs.settings_file_pth.value
        with open(toml_settings, 'rb') as f:
            loaded_toml_settings = tomllib.load(f)

        # Adding current active learning step to monitor stage
        loaded_toml_settings['active_learning']['current_iteration'] = (
            self.inputs.curr_active_learning_step.value
        )

        # Saving modified settings to a temporary file and copying the
        # modified toml into the calculation folder
        with tempfile.NamedTemporaryFile(
            mode='wb', delete=True, suffix='.toml', prefix='atl_process_md-'
        ) as f_toml:
            f_toml.write(tomlkit_dumps(loaded_toml_settings).encode())
            toml_settings = f_toml.name

            # Copying settings file
            folder.insert_path(
                src=toml_settings,
                dest_name='settings.toml',
            )

            # Remove the file after insertion
            f_toml.close()
            Path(f_toml.name).unlink(missing_ok=True)

        # Copying concave hull for extrapolation
        if hasattr(self.inputs, 'concave_hull') and isinstance(
            self.inputs.concave_hull, (np.ndarray, orm.ArrayData)
        ):
            concave_hull = self.inputs.concave_hull.get_array()
            with tempfile.NamedTemporaryFile(
                mode='w',
                delete=True,
                suffix='.npy',
                prefix='atl_process_md-',
            ) as f:
                np.save(f.name, concave_hull)
                folder.insert_path(
                    src=f.name,
                    dest_name='concave_hull.npy',
                )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        elif hasattr(self.inputs, 'concave_hull') and isinstance(
            self.inputs.concave_hull, (list, orm.List)
        ):
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=True,
                suffix='.pkl',
                prefix='atl_process_md-',
            ) as f:
                pickle.dump(self.inputs.concave_hull.get_list(), f)

                # Ensure data is written to disk
                f.flush()
                folder.insert_path(
                    src=f.name,
                    dest_name='concave_hulls.pkl',
                )

        # Copying concave hull for extrapolation
        if hasattr(self.inputs, 'autoencoder_model') and isinstance(
            self.inputs.autoencoder_model,
            orm.SinglefileData,
        ):
            with self.inputs.autoencoder_model.as_path() as autoencoder_path:
                folder.insert_path(
                    src=autoencoder_path,
                    dest_name='autoencoder_model.pth',
                )

        # Copying descriptors max and min
        desc_max_arr: orm.ArrayData = self.inputs.desc_max_arr.get_array()
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.npy', prefix='atl_process_md-'
        ) as f:
            np.save(f.name, desc_max_arr)
            folder.insert_path(
                src=f.name,
                dest_name='curr_it_db_max.npy',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        desc_min_arr = self.inputs.desc_min_arr.get_array()
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.npy', prefix='atl_process_md-'
        ) as f:
            np.save(f.name, desc_min_arr)
            folder.insert_path(
                src=f.name,
                dest_name='curr_it_db_min.npy',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        # Copying configuration to temporary folder
        best_model_name = self.inputs.best_model_name.value.replace('-', '_')
        for model_str, model_singlefile in self.inputs.commitee_models.items():
            # If the best model is in the name, use it as the current model
            if model_str in best_model_name:
                with model_singlefile.as_path() as model_path:
                    folder.insert_path(
                        src=model_path,
                        dest_name='curr_model.model',
                    )
            else:
                with model_singlefile.as_path() as model_path:
                    folder.insert_path(
                        src=model_path,
                        dest_name=f'{model_str}.model',
                    )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        # codeinfo.stdout_name = self.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        # calcinfo.provenance_exclude_list = [
        #     self.inputs.mace_settings_dict["train_file"]
        # ]
        calcinfo.remote_copy_list = []

        # Gathering files. They won't be added to the repository,
        # and instead kept into a temporary folder.
        # They can later be processed during the parse function
        # by accessing the temporary folder.
        calcinfo.retrieve_temporary_list = [
            # self.metadata.options.output_filename,
            './results/*.xyz',
            './results/*.png',
            './logs/*',
        ]

        return calcinfo


class ProcessMDSeedStructCalculationParser(Parser):
    """Parser for the retrieved files from an active learning MD sampling job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        extrapolating_structures = None
        extrapolation_plot = None

        for child_file in retrieved_temporary_folder.rglob('*'):
            if 'extrapolating_frames.xyz' in child_file.name:
                extrapolating_structures = orm.SinglefileData(file=child_file)
            if '.png' in child_file.name:
                extrapolation_plot = atl_img.ImagePNGData(filepath=child_file)

        # Return failed code
        if not extrapolating_structures:
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        # TODO: extrapolating_plot can be None, but the output will result in error,
        # as the required=False is not having an effect? This is a workaround for that
        if not extrapolation_plot:
            with tempfile.NamedTemporaryFile(
                mode='ab+',
                delete=True,
                suffix='.txt',
                prefix='atl_extrapolation_plot_placeholder-',
            ) as f:
                f.write(b'')
                extrapolation_plot = orm.SinglefileData(file=f)

        self.out('extrapolating_structures', extrapolating_structures)
        self.out('extrapolation_plot', extrapolation_plot)


# atl-descriptors-combined-parser
class GetDescriptorsCombinedParser(Parser):
    """
    Parser for a descriptor and extrapolation gathering job.

    Methods
    -------
    parse(**kwargs)
        Parses the temporarily retrieved files.
        Outputs are stored in AiiDA SinglefileData objects.
    """

    def parse(self, **kwargs):
        """
        Parse the retrieved files of the calculation job.

        Returns
        -------
        descriptor_max : aiida.orm.SinglefileData
            File containing the maximum values for the descriptors.
        descriptor_min : aiida.orm.SinglefileData
            File containing the minimum values for the descriptors.
        concave_hull : aiida.orm.SinglefileData, optional
            File containing the concave hull of the latent space as an array.
        latent_space : aiida.orm.SinglefileData, optional
            File containing the latent space represented as an array.
        """
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        descriptor_max = None
        descriptor_min = None
        concave_hull = None
        latent_space = None
        extrapolation_plot = None
        autoencoder_model = None
        concave_hull_data = None

        # Gathering results from the temporary folder
        # for child_file in retrieved_temporary_folder.iterdir():
        for child_file in retrieved_temporary_folder.rglob('*'):
            match child_file.name:
                # case "curr_it_db_descriptors.pkl":
                # descriptor_arr_file = orm.SinglefileData(file=child_file.absolute())
                case 'curr_it_db_max.npy':
                    descriptor_max = orm.ArrayData(
                        arrays=np.load(child_file.absolute())
                    )
                case 'curr_it_db_min.npy':
                    descriptor_min = orm.ArrayData(
                        arrays=np.load(child_file.absolute())
                    )
                case 'concave_hull.npy':
                    concave_hull = orm.ArrayData(arrays=np.load(child_file.absolute()))
                case 'concave_hulls_data.pkl':
                    with open(child_file.absolute(), 'rb') as f:
                        concave_hull_data = orm.SinglefileData(
                            file=child_file.absolute()
                        )
                case 'concave_hulls.pkl':
                    with open(child_file.absolute(), 'rb') as f:
                        concave_hull = orm.List(pickle.load(f))
                case 'latent_space.npy':
                    latent_space = orm.ArrayData(arrays=np.load(child_file.absolute()))
                case 'concave_hull.png':
                    extrapolation_plot = atl_img.ImagePNGData(
                        filepath=child_file.absolute()
                    )
                case 'autoencoder_model.pth':
                    autoencoder_model = orm.SinglefileData(file=child_file.absolute())

        # Return failed code if the mandatory outputs are missing
        if not all((descriptor_max, descriptor_min)):
            return self.exit_codes.ERROR_OUTPUT_NOT_FOUND

        # Return CalcJob outputs
        self.out('descriptor_max', descriptor_max)
        self.out('descriptor_min', descriptor_min)

        if latent_space:
            self.out('latent_space', latent_space)
        if concave_hull:
            self.out('concave_hull', concave_hull)
        if concave_hull_data:
            self.out('detailed_concave_hull', concave_hull_data)
        if extrapolation_plot:
            self.out('extrapolation_plot', extrapolation_plot)
        if autoencoder_model:
            self.out('autoencoder_model', autoencoder_model)


# atl-eval-test-parser
class EvalTestDatabaseCalculationParser(Parser):
    """
    Parser for a descriptor and extrapolation gathering job.

    Methods
    -------
    parse(**kwargs)
        Parses the temporarily retrieved files.
        Outputs are stored in AiiDA SinglefileData objects.
    """

    def parse(self, **kwargs):
        """
        Parse the retrieved files of the calculation job.

        Returns
        -------
        descriptor_max : aiida.orm.SinglefileData
            File containing the maximum values for the descriptors.
        descriptor_min : aiida.orm.SinglefileData
            File containing the minimum values for the descriptors.
        concave_hull : aiida.orm.SinglefileData, optional
            File containing the concave hull of the latent space as an array.
        latent_space : aiida.orm.SinglefileData, optional
            File containing the latent space represented as an array.
        """
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        rmse_e = None
        rmse_f = None
        mae_e = None
        mae_f = None
        eval_plot = None

        # Gathering results from the temporary folder
        # for child_file in retrieved_temporary_folder.iterdir():
        for child_file in retrieved_temporary_folder.rglob('*'):
            match child_file.name:
                # case "curr_it_db_descriptors.pkl":
                # descriptor_arr_file = orm.SinglefileData(file=child_file.absolute())
                case 'test_db_eval_results_updated.json':
                    with open(child_file.absolute()) as f:
                        results_dict = json.load(f)
                        current_iter = results_dict['current_iteration']
                        current_results = results_dict[f'step_{current_iter}']

                        # Get error values
                        rmse_e = orm.Float(current_results['rmse_e'])
                        rmse_f = orm.Float(current_results['rmse_f'])
                        mae_e = orm.Float(current_results['mae_e'])
                        mae_f = orm.Float(current_results['mae_f'])

                        # Save float outputs
                        self.out('rmse_e', rmse_e)
                        self.out('rmse_f', rmse_f)
                        self.out('mae_e', mae_e)
                        self.out('mae_f', mae_f)

                        # Save dict as output
                        self.out(
                            'test_db_eval_results',
                            orm.SinglefileData(file=child_file.absolute()),
                        )

                case 'test_db_eval_plots.png':
                    eval_plot = atl_img.ImagePNGData(filepath=child_file.absolute())
                    self.out('eval_plot', eval_plot)

        # Return failed code if the mandatory outputs are missing
        if not all((rmse_e, rmse_f, mae_e, mae_f)):
            return self.exit_codes.ERROR_OUTPUT_NOT_FOUND


# entry-point: atl-eval-test
class EvalTestDatabaseCalculation(CalcJob):
    """CalcJob to evaluate the test database using the sampler model (M0).

    This calculation predicts all energies and forces for the test database,
    computing the MAE and RMSE values for both properties at each step of the
    active learning iteration, saving the results in the workchain and preparing
    a report figure showing the evolution of the error over time.

    Parameters
    ----------
    spec : aiida.engine.processes.ports.PortNamespace
        The process specification to define the inputs, outputs, and exit codes.

    Inputs
    ------
    sampler_model : aiida.orm.SinglefileData
        File containing the MACE model to use for evaluation.
    current_iteration : aiida.orm.Int
        Current iteration number.
    settings_file_path : aiida.orm.Str
        Path to the ATL settings file in the toml format.
    test_database : aiida.orm.SinglefileData
        File containing the structures for testing the model.
    test_db_eval_results : aiida.orm.Dict
        Dictionary containing the evaluation results up until now.


    Outputs
    -------
    rmse_e : aiida.orm.Float
        Root mean square error for energy predictions.
    rmse_f : aiida.orm.Float
        Root mean square error for force predictions.
    mae_e : aiida.orm.Float
        Mean absolute error for energy predictions.
    mae_f : aiida.orm.Float
        Mean absolute error for force predictions.
    eval_plot : aiida.orm.SinglefileData
        File containing the evaluation plot.
    test_db_eval_results : aiida.orm.SinglefileData
        Dictionary containing the updated evaluation results.


    Exit Codes
    ----------
    420 : ERROR_OUT_OF_VRAM
        CUDA out of GPU memory.
    421 : ERROR_OUTPUT_NOT_FOUND
        Missing output file.
    """

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)

        # Namespace that will hold an arbitrary number of committee MACE potentials
        spec.input(
            'sampler_model',
            valid_type=orm.SinglefileData,
            non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'current_iteration',
            valid_type=orm.Int,
            help='Current iteration number.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'settings_file_path',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the toml format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'test_database',
            valid_type=orm.SinglefileData,
            non_db=True,
            help='File containing the structures for testing the model.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'test_db_eval_results',
            help='Dictionary containing the evaluation results up until now.',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
        )
        spec.output(
            'rmse_e',
            valid_type=orm.Float,
            help='Root mean square error for energy predictions.',
            required=False,
        )
        spec.output(
            'rmse_f',
            valid_type=orm.Float,
            help='Root mean square error for force predictions.',
            required=False,
        )
        spec.output(
            'mae_e',
            valid_type=orm.Float,
            help='Mean absolute error for energy predictions.',
            required=False,
        )
        spec.output(
            'mae_f',
            valid_type=orm.Float,
            help='Mean absolute error for force predictions.',
        )
        spec.output(
            'eval_plot',
            valid_type=atl_img.ImagePNGData,
            help='File containing the evaluation plot.',
            required=False,
        )
        spec.output(
            'test_db_eval_results',
            help='Dictionary containing the updated evaluation results.',
            valid_type=orm.SinglefileData,
        )

        spec.exit_code(420, 'ERROR_OUT_OF_VRAM', 'CUDA out of GPU memory.')
        spec.exit_code(421, 'ERROR_OUTPUT_NOT_FOUND', 'Missing output file.')

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Copying configuration to temporary folder
        with self.inputs.sampler_model.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='curr_iter_best.model',
            )

        # Copying settings file
        toml_settings = self.inputs.settings_file_path.value
        folder.insert_path(
            src=toml_settings,
            dest_name='settings.toml',
        )

        # Copying test database file
        with self.inputs.test_database.as_path() as test_db_path:
            folder.insert_path(
                src=test_db_path,
                dest_name='test_db.xyz',
            )

        test_db_eval_results = self.inputs.test_db_eval_results.get_dict()

        current_iteration = self.inputs.current_iteration.value
        test_db_eval_results['current_iteration'] = current_iteration

        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.json', prefix='atl_mace_eval-test-'
        ) as tmp_file:
            # Write the updated results dict to a temporary json file
            with open(tmp_file.name, 'w') as f:
                json_str = json.dumps(obj=test_db_eval_results)
                f.write(json_str)

            # Insert the temporary file into the folder
            folder.insert_path(
                src=tmp_file.name,
                dest_name='test_db_eval_results.json',
            )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files.
        calcinfo.retrieve_list = [
            # "results.out",
            # "./results/curr_it_db*",
            # "./results/*.png",
            # "./results/*.npy",
            './logs/*',
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            '*.png',
            '*.json',
        ]

        return calcinfo


# entry-point: atl-descriptors-combined
class GetDescriptorsCombinedCalculation(CalcJob):
    """CalcJob to gather the descriptors for the training database of an AL Loop.

    This calculation job computes the descriptors for all the configurations in
    the training database. Additionally, further extrapolation metrics are computed
    depending on the the selected extrapolation type.
    With min-max extrapolation enabled, the the minimum and maximum range for all
    descriptors in the training database is computed.
    With advanced extrapolation, the ranges plus the concave hull of the latent space
    for all configurations in the training database, are provided, along with a plot
    showing the configuration distribution in the latent space and the concave hull.

    Parameters
    ----------
    spec : aiida.engine.processes.ports.PortNamespace
        The process specification to define the inputs, outputs, and exit codes.

    Inputs
    ------
    commitee_models : PortNamespace
        A namespace to hold an arbitrary number of committee MACE potentials.
    settings_file_path : orm.Str
        Path to the ATL settings file in the .toml format.
    training_database_path : orm.Str
        Path to the configurations to evaluate, provided in the extxyz format.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.
        If not provided, a new autoencoder is trained is computed.
    latent_space : orm.SinglefileData, optional
        File containing the latent space represented as an array.
        If not provided, the latent space is computed.

    Outputs
    -------
    descriptor_max : orm.ArrayData
        File containing the maximum values for the descriptors.
    descriptor_min : orm.ArrayData
        File containing the minimum values for the descriptors.
    latent_space : orm.ArrayData, optional
        File containing the latent space as an array.
    concave_hull : orm.ArrayData, optional
        File containing the concave hull of the latent space as an array.
    extrapolation_plot : atl_img.ImagePNGData, optional
        Figure showing the extrapolation for the current database.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.

    Exit Codes
    ----------
    420 : ERROR_OUT_OF_VRAM
        CUDA out of GPU memory.
    421 : ERROR_OUTPUT_NOT_FOUND
        Missing output file.
    """

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)

        # Namespace that will hold an arbitrary number of committee MACE potentials
        spec.input(
            'best_model',
            valid_type=orm.SinglefileData,
            non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'settings_file_path',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the toml format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'training_database_path',
            valid_type=orm.Str,
            help='Path with the configurations to evaluate in extxyz format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'latent_space',
            valid_type=orm.ArrayData,
            help='File containing the latent space as an array.',
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
            non_db=True,
        )
        spec.input(
            'autoencoder_model',
            valid_type=orm.ArrayData,
            help='File containing the autoencoder model.',
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
            non_db=True,
        )
        spec.output(
            'concave_hull',
            valid_type=orm.List,
            help='Array containing the concave hull of the latent space.',
            required=False,
        )
        spec.output(
            'detailed_concave_hull',
            valid_type=orm.SinglefileData,
            help='Pickled list of dicts containing quadtree and latents space info.',
            required=False,
        )
        spec.output(
            'latent_space',
            valid_type=orm.ArrayData,
            help='Array containing the latent space.',
            required=False,
        )
        spec.output(
            'extrapolation_plot',
            valid_type=atl_img.ImagePNGData,
            help='Figure showing the extrapolation for the current database.',
            required=False,
        )
        spec.output(
            'descriptor_max',
            valid_type=orm.ArrayData,
            help='File containing the maximum values for the descriptors.',
        )
        spec.output(
            'descriptor_min',
            valid_type=orm.ArrayData,
            help='File containing the minimum values for the descriptors.',
        )
        spec.output(
            'autoencoder_model',
            valid_type=orm.SinglefileData,
            help='File containing the autoencoder model.',
            required=False,
        )

        spec.exit_code(420, 'ERROR_OUT_OF_VRAM', 'CUDA out of GPU memory.')
        spec.exit_code(421, 'ERROR_OUTPUT_NOT_FOUND', 'Missing output file.')

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Copying configuration to temporary folder
        with self.inputs.best_model.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='curr_iter_best.model',
            )

        # Copying settings file
        toml_settings = self.inputs.settings_file_path.value
        folder.insert_path(
            src=toml_settings,
            dest_name='settings.toml',
        )

        # Copying database file
        train_db_path = self.inputs.training_database_path.value
        train_db_path = str(Path(train_db_path).resolve())
        folder.insert_path(
            src=train_db_path,
            dest_name='training_db.xyz',
        )

        # Copying configuration to temporary folder
        if self.inputs.latent_space:
            with self.inputs.latent_space.as_path() as latent_space_path:
                folder.insert_path(
                    src=latent_space_path,
                    dest_name='latent_space.npy',
                )
        if self.inputs.autoencoder_model:
            with self.inputs.autoencoder_model.as_path() as autoencoder_model_path:
                folder.insert_path(
                    src=autoencoder_model_path,
                    dest_name='autoencoder_model.pth',
                )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files.
        calcinfo.retrieve_list = [
            './logs/*',
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            # '*_output.out',
            # 'results/*.pkl',
            # 'results/curr_it_db*',
            # 'results/*.png',
            # 'results/*.npy',
            # 'results/concave_hull.npy',
            # 'results/latent_space.npy',
            # 'results/*.pth',
            'results/*',
            # '*.pth',
        ]

        return calcinfo
