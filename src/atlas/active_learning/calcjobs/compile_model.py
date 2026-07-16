"""Backend-agnostic "compile model for inference" CalcJob and Parser.

Runs on the computer where the model will be used for inference, so the
compiled artifact matches that node's device and toolchain. Bundles the raw
model + a small ``compile_settings.json`` and runs ``atl_compile_model.py``,
which dispatches to ``backend.compile_model`` (see ``MLIPModelCompiler``).

Only used for backends that implement ``MLIPModelCompiler`` (e.g. Allegro);
compile-free backends (e.g. MACE) never submit this calc.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser


class CompileModelCalculationParser(Parser):
    """Parser for :class:`CompileModelCalculation`.

    Reads ``compile_results.json`` (written by ``atl_compile_model.py``) and
    returns the compiled artifact as the ``compiled_model`` output.
    """

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        retrieved_temporary_folder = Path(kwargs['retrieved_temporary_folder'])

        results_json = retrieved_temporary_folder / 'compile_results.json'
        if not results_json.exists():
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(node_id=self.node.pk)

        with open(results_json) as f:
            results = json.load(f)

        compiled_name = results.get('compiled_model')
        if not compiled_name:
            return self.exit_codes.ERROR_COMPILE_FAILED.format(node_id=self.node.pk)

        compiled_path = retrieved_temporary_folder / compiled_name
        if not compiled_path.exists():
            for candidate in retrieved_temporary_folder.rglob(Path(compiled_name).name):
                compiled_path = candidate
                break

        if not compiled_path.exists():
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(node_id=self.node.pk)

        self.out('compiled_model', orm.SinglefileData(file=compiled_path))


class CompileModelCalculation(CalcJob):
    """Compile a trained model into an inference artifact on the target node."""

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications."""
        super().define(spec)
        spec.input(
            'model',
            valid_type=orm.SinglefileData,
            help='Raw trained model/checkpoint to compile.',
        )
        spec.input(
            'compile_settings',
            valid_type=orm.Dict,
            help=("Compile settings: keys 'backend', 'device', 'mode', 'target'."),
        )
        spec.output(
            'compiled_model',
            valid_type=orm.SinglefileData,
            help='Compiled, inference-ready model artifact.',
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            'Compile calculation ({node_id}) produced no parseable output.',
        )
        spec.exit_code(
            421,
            'ERROR_COMPILE_FAILED',
            'Model compilation failed in {node_id} (see scheduler output).',
        )

    def prepare_for_submission(self, folder):
        """Ship the model + compile settings and set up the run."""
        settings = dict(self.inputs.compile_settings.get_dict())

        # Ship the raw model under its own filename.
        model_filename = self.inputs.model.filename
        with self.inputs.model.as_path() as model_path:
            folder.insert_path(src=str(model_path), dest_name=model_filename)

        # The remote script reads these to dispatch to backend.compile_model.
        compile_cfg = {
            'backend': settings.get('backend', 'mace'),
            'device': settings.get('device', 'cpu'),
            'mode': settings.get('mode', 'aotinductor'),
            'target': settings.get('target', 'ase'),
            'model_filename': model_filename,
        }

        # Exact name of the compiled artifact the script will produce, so the
        # retrieve list uses a literal filename rather than a glob. The AiiDA
        # async transport raises if a glob matches nothing (e.g. an empty
        # ``*.nequip.pth`` when aotinductor produced only a ``.pt2``).
        from atlas.active_learning.backends import model_file_stem

        compiled_ext = (
            '.nequip.pt2' if compile_cfg['mode'] == 'aotinductor' else '.nequip.pth'
        )
        compiled_name = f'{model_file_stem(model_filename)}{compiled_ext}'
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.json', prefix='atl_compile-'
        ) as f:
            json.dump(compile_cfg, f)
            tmp_name = f.name
        folder.insert_path(src=tmp_name, dest_name='compile_settings.json')
        Path(tmp_name).unlink(missing_ok=True)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        # Use a literal stdout name: ``output_filename`` is not a standard
        # CalcJob option and the target computer's option dict does not set it.
        codeinfo.stdout_name = 'compile.out'

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = [model_filename]
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_temporary_list = [
            'compile.out',
            'compile_results.json',
            compiled_name,
        ]

        return calcinfo
