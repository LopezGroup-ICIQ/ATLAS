"""Generic backend-agnostic MLIP training CalcJob and Parser.

Uses PortableCode to bundle ``atl_train_mlip.py`` which dispatches
training to the configured backend. The parser reads the standardized
``training_results.json`` written by that script.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import yaml
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser


class TrainMLIPCalculationParser(Parser):
    """Parser for the generic MLIP training CalcJob.

    Reads ``training_results.json`` written by ``atl_train_mlip.py``
    and retrieves the model file and metrics.
    """

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        results_json = retrieved_temporary_folder / 'training_results.json'
        if not results_json.exists():
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        with open(results_json) as f:
            results = json.load(f)

        model_file_path = results.get('model_file')
        rmse_e = results.get('rmse_e')
        rmse_f = results.get('rmse_f')

        if model_file_path is None or rmse_e is None or rmse_f is None:
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        model_path = retrieved_temporary_folder / model_file_path
        if not model_path.exists():
            for candidate in retrieved_temporary_folder.rglob(
                Path(model_file_path).name
            ):
                model_path = candidate
                break

        if not model_path.exists():
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        if np.isnan(rmse_e) or np.isnan(rmse_f):
            return self.exit_codes.ERROR_NAN_TRAINING_RESULTS.format(
                node_id=self.node.pk,
            )

        self.out('model_file', orm.SinglefileData(file=model_path))
        self.out('m_rmse_e', orm.Float(rmse_e))
        self.out('m_rmse_f', orm.Float(rmse_f))

        train_log_path = results.get('train_log')
        if train_log_path:
            log_path = retrieved_temporary_folder / train_log_path
            if not log_path.exists():
                for candidate in retrieved_temporary_folder.rglob(
                    Path(train_log_path).name
                ):
                    log_path = candidate
                    break
            if log_path.exists():
                self.out('train_file', orm.SinglefileData(file=log_path))


class TrainMLIPCalculation(CalcJob):
    """Generic CalcJob for MLIP training via PortableCode.

    Writes a backend-specific training config and a ``settings.toml``
    that tells the bundled script which backend to dispatch to.
    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications."""
        super().define(spec)
        spec.input(
            'train_config_dict',
            valid_type=orm.Dict,
            help='Backend-specific training config dictionary (written as YAML).',
        )
        spec.input(
            'mlip_settings',
            valid_type=orm.Dict,
            help='MLIP settings dict containing at least training_backend key.',
        )
        spec.input(
            'train_file_path',
            valid_type=orm.Str,
            help='Path to the training data file (extxyz) on the remote machine.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'model_name',
            valid_type=orm.Str,
            help='Name given to this training run.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'pretrained_model_path',
            valid_type=orm.Str,
            help='Path to a pretrained/foundation model to copy to remote.',
            serializer=orm.to_aiida_type,
            required=False,
        )

        spec.output(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Trained MLIP model file.',
        )
        spec.output(
            'train_file',
            valid_type=orm.SinglefileData,
            help='Training log / metrics file.',
            required=False,
        )
        spec.output(
            'm_rmse_e',
            valid_type=orm.Float,
            help='Validation RMSE for energy (meV/atom).',
        )
        spec.output(
            'm_rmse_f',
            valid_type=orm.Float,
            help='Validation RMSE for forces (meV/A).',
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            'Training calculation ({node_id}) could not run.',
        )
        spec.exit_code(
            421,
            'ERROR_NAN_TRAINING_RESULTS',
            'Training produced NaN RMSE values in {node_id}.',
        )

    def prepare_for_submission(self, folder):
        """Write input files for the backend-agnostic training script."""
        config_dict = dict(self.inputs.train_config_dict.get_dict())

        seed = np.random.randint(1, 100000000)
        config_dict['seed'] = seed

        # Write backend-specific training config as YAML
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.yaml', prefix='atl_train_config-'
        ) as f:
            yaml.dump(config_dict, f)
            folder.insert_path(src=f.name, dest_name='train_config.yaml')
            f.close()
            Path(f.name).unlink(missing_ok=True)

        # Write settings.toml for backend dispatch
        mlip_settings = self.inputs.mlip_settings.get_dict()
        settings_toml_content = (
            '[mlip]\n'
            f'training_backend = "{mlip_settings.get("training_backend", "mace")}"\n'
        )
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.toml', prefix='atl_settings-'
        ) as f:
            f.write(settings_toml_content)
            f.flush()
            folder.insert_path(src=f.name, dest_name='settings.toml')
            f.close()
            Path(f.name).unlink(missing_ok=True)

        # Copy pretrained/foundation model if provided
        if 'pretrained_model_path' in self.inputs:
            pt_path = Path(self.inputs.pretrained_model_path.value).resolve()
            if pt_path.exists():
                folder.insert_path(
                    src=str(pt_path), dest_name=pt_path.name
                )

        # Copy training data
        final_db_path = self.inputs.train_file_path.value
        raw_name = config_dict.get(
            'train_file',
            config_dict.get('dataset_file_name', final_db_path),
        )
        train_file_name = Path(raw_name).name
        folder.insert_path(src=final_db_path, dest_name=train_file_name)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = [train_file_name]
        calcinfo.remote_copy_list = []

        calcinfo.retrieve_temporary_list = [
            self.metadata.options.output_filename,
            'training_results.json',
            './*.model',
            './*.pth',
            './results/*',
            './train_*',
        ]

        return calcinfo
