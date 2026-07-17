"""Allegro-specific AiiDA CalcJob and Parser classes."""

from pathlib import Path

import numpy as np
import yaml
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser


class TrainAllegroModelCalculationParser(Parser):
    """Parser for retrieved files from an Allegro training calculation."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        model_file = None
        rmse_e = None
        rmse_f = None
        train_file = None

        for child_file in retrieved_temporary_folder.rglob('*'):
            if child_file.name == 'deployed_model.pth':
                model_file = orm.SinglefileData(file=child_file)
                continue

            if child_file.name == 'best_model.pth' and model_file is None:
                model_file = orm.SinglefileData(file=child_file)
                continue

            if child_file.name == 'metrics.csv' or child_file.name == 'log':
                train_file = orm.SinglefileData(file=child_file)

            if child_file.name == 'metrics.csv':
                lines = child_file.read_text().strip().split('\n')
                if len(lines) > 1:
                    header = lines[0].split(',')
                    last_row = lines[-1].split(',')
                    metrics = dict(zip(header, last_row, strict=False))

                    e_key = next(
                        (
                            k
                            for k in metrics
                            if 'rmse' in k.lower() and 'energy' in k.lower()
                        ),
                        None,
                    )
                    f_key = next(
                        (
                            k
                            for k in metrics
                            if 'rmse' in k.lower() and 'force' in k.lower()
                        ),
                        None,
                    )

                    if e_key:
                        rmse_e = float(metrics[e_key]) * 1000  # eV -> meV/atom
                    if f_key:
                        rmse_f = float(metrics[f_key]) * 1000  # eV/A -> meV/A

        if None in (rmse_e, rmse_f, model_file):
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        if np.isnan(rmse_e) or np.isnan(rmse_f):
            return self.exit_codes.ERROR_NAN_TRAINING_RESULTS.format(
                node_id=self.node.pk,
            )

        self.out('model_file', model_file)
        if train_file:
            self.out('train_file', train_file)
        self.out('m_rmse_e', orm.Float(rmse_e))
        self.out('m_rmse_f', orm.Float(rmse_f))


class TrainAllegroModelCalculation(CalcJob):
    """CalcJob to perform Allegro training via ``nequip-train``.

    Inputs
    ------
    allegro_config_dict : orm.Dict
        Dictionary representing the nequip-train YAML config.
    train_file_path : orm.Str
        Path to the training data (extxyz) on the remote machine.
    model_name : orm.Str
        Name given to this training run.
    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications."""
        super().define(spec)
        spec.input(
            'allegro_config_dict',
            valid_type=orm.Dict,
            help='Dictionary representing the nequip-train YAML config.',
        )
        spec.input(
            'train_file_path',
            valid_type=orm.Str,
            help='Path to training data in extxyz format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'model_name',
            valid_type=orm.Str,
            help='Name given to this training run.',
            serializer=orm.to_aiida_type,
        )

        spec.output(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Trained (deployed) Allegro model.',
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
        """Write input files for nequip-train."""
        config_dict = dict(self.inputs.allegro_config_dict.get_dict())

        seed = np.random.randint(1, 100000000)
        config_dict['seed'] = seed
        config_dict['dataset_file_name'] = self.inputs.train_file_path.value

        config_path = 'train_config.yaml'
        with folder.open(config_path, 'w') as fh:
            yaml.dump(config_dict, fh, default_flow_style=False)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [config_path]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ['results/*']
        calcinfo.retrieve_temporary_list = ['results/**/*']

        return calcinfo
