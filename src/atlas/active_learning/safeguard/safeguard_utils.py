"""AiiDA plugin for MACE calculations."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser

from atlas.workflows.datatypes import image_types as atl_img


# atl-safeguard-md
class RunMDSafeguardCalculation(CalcJob):
    """
    CalcJob to run MD simulations for safeguard checking during active learning loops.

    This CalcJob will run MD simulations to test the stability of the models trained
    using active learning before deciding to stop the active learning loop. The MD
    simulations will be analyzed to identify any extrapolating structures using any
    of the available extrapolation methods.

    Parameters
    ----------
    md_structure : orm.SinglefileData
        File containing the structure to be used for the MD, in the extxyz format.
    sampler_model : orm.SinglefileData
        File containing the MLIP model to test for stability.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.
    m_rmse_e : orm.Float
        Validation RMSE of the best model for the energy, in meV / atom.
    m_rmse_f : orm.Float
        Validation RMSE of the best model for the forces, in meV / Å.
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

    Exit Codes
    ----------
    420 : ERROR_INVALID_OUTPUT
        Structure could not be processed.

    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications for the CalcJob."""
        super().define(spec)
        spec.input(
            'sampler_model',
            valid_type=orm.SinglefileData,
            non_db=True,
            required=True,
        )
        # spec.input(
        #     'best_model_name',
        #     valid_type=orm.Str,
        #     help='Name of the best model.',
        #     required=True,
        # )
        spec.input(
            'md_structure',
            valid_type=orm.SinglefileData,
            help=(
                'File containing the structure to be used for the MD,'
                'in the extxyz format.'
            ),
            required=True,
            serializer=orm.to_aiida_type,
            # non_db=True,
        )
        spec.input(
            'm_rmse_e',
            valid_type=orm.Float,
            help='Validation RMSE of the best model for the energy, in meV / atom.',
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'm_rmse_f',
            valid_type=orm.Float,
            help='Validation RMSE of the best model for the forces, in meV / Å.',
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'autoencoder_model',
            valid_type=(orm.SinglefileData, None),
            help='File containing the autoencoder model.',
            required=False,
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'desc_max_arr',
            valid_type=(orm.ArrayData, None),
            help='Array containing the maximum values for the descriptors.',
            required=False,
            non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'desc_min_arr',
            valid_type=(orm.ArrayData, None),
            help='Array containing the minimum values for the descriptors.',
            required=False,
            non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'concave_hull',
            valid_type=orm.List,
            help=(
                'List of arrays containing the concave hull to be used for the '
                'extrapolation check.'
            ),
            required=False,
            # non_db=True,
            default=None,
        )
        spec.input(
            'settings_file_pth',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the .toml format.',
            serializer=orm.to_aiida_type,
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
            mode='w', delete=True, suffix='.npy', prefix='atl_safeguard_md-'
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

        # Copying settings file
        toml_settings = self.inputs.settings_file_pth.value
        folder.insert_path(
            src=toml_settings,
            dest_name='settings.toml',
        )

        # Copying concave hull for extrapolation
        if hasattr(self.inputs, 'concave_hull') and isinstance(
            self.inputs.concave_hull, (np.ndarray, orm.ArrayData)
        ):
            concave_hull = self.inputs.concave_hull.get_array()
            with tempfile.NamedTemporaryFile(
                mode='w', delete=True, suffix='.npy', prefix='atl_safeguard_md-'
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

                # Remove the file after insertion
                f.close()
                Path(f.name).unlink(missing_ok=True)

        # Copying concave hull for extrapolation
        if hasattr(self.inputs, 'autoencoder_model'):
            with self.inputs.autoencoder_model.as_path() as autoencoder_path:
                folder.insert_path(
                    src=autoencoder_path,
                    dest_name='autoencoder_model.pth',
                )

        # Copying descriptors max and min
        if hasattr(self.inputs, 'desc_max_arr'):
            desc_max_arr: orm.ArrayData = self.inputs.desc_max_arr.get_array()
            with tempfile.NamedTemporaryFile(
                mode='w', delete=True, suffix='.npy', prefix='atl_safeguard_md-'
            ) as f:
                np.save(f.name, desc_max_arr)
                folder.insert_path(
                    src=f.name,
                    dest_name='curr_it_db_max.npy',
                )

                # Remove the file after insertion
                f.close()
                Path(f.name).unlink(missing_ok=True)

        if hasattr(self.inputs, 'desc_min_arr'):
            desc_min_arr = self.inputs.desc_min_arr.get_array()
            with tempfile.NamedTemporaryFile(
                mode='w', delete=True, suffix='.npy', prefix='atl_safeguard_md-'
            ) as f:
                np.save(f.name, desc_min_arr)
                folder.insert_path(
                    src=f.name,
                    dest_name='curr_it_db_min.npy',
                )

                # Remove the file after insertion
                f.close()
                Path(f.name).unlink(missing_ok=True)

        # best_model_name = self.inputs.best_model_name.value.replace('-', '_')
        # for model_str, model_singlefile in self.inputs.commitee_models.items():
        # If the best model is in the name, use it as the current model
        # if model_str in best_model_name:
        #     with model_singlefile.as_path() as model_path:
        #         folder.insert_path(
        #             src=model_path,
        #             dest_name='curr_model.model',
        #         )
        # else:

        # Copying models to temporarty folder. Preserve the model file's real
        # extension (e.g. '.model' for MACE, '.ckpt' for Allegro) so the remote
        # safeguard script, which loads it via ``backend.model_file_extension``,
        # finds it regardless of backend.
        from atlas.active_learning.backends import model_file_ext

        model_ext = model_file_ext(self.inputs.sampler_model.filename)
        with self.inputs.sampler_model.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name=f'sampler_model{model_ext}',
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
            './results/*',
            './logs/*',
        ]

        return calcinfo


# atl-safeguard-md-parser
class RunMDSafeguardCalculationParser(Parser):
    """Parser for the retrieved files from a safeguard calculation job."""

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
                extrapolation_plot = atl_img.ImagePNGData(filepath=f)

        self.out('extrapolating_structures', extrapolating_structures)
        self.out('extrapolation_plot', extrapolation_plot)
