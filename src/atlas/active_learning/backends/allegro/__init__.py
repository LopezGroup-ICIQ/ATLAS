"""Allegro backend for ATLAS active learning.

This module provides the Allegro implementation of the MLIP backend
protocols. Allegro is built on top of NequIP and shares its training
infrastructure (``nequip-train``) and ASE calculator
(``NequIPCalculator``).

Allegro implements ``MLIPTrainer``, ``MLIPCalculatorFactory``, and
``MLIPCommitteeEvaluator`` but **not** ``MLIPDescriptorProvider`` —
Allegro does not expose MACE-style invariant descriptors.
When Allegro is used, the workflow falls back to SOAP descriptors
for extrapolation detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atlas.active_learning.backends import register_backend

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def _patch_torch_load_if_needed():
    """Fix ``torch.load`` for e3nn < 0.5.2 + PyTorch >= 2.6.

    Sets ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` so that all processes
    (including forkserver multiprocessing workers spawned by nequip)
    default to ``weights_only=False``.
    """
    import os
    import warnings

    try:
        import torch
        from packaging.version import Version

        torch_version = Version(torch.__version__.split('+')[0])
        if torch_version < Version('2.6'):
            return

        import e3nn

        e3nn_version = Version(e3nn.__version__)
        if e3nn_version >= Version('0.5.2'):
            return

        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

        warnings.warn(
            f'Detected e3nn {e3nn.__version__} with PyTorch '
            f'{torch.__version__}. Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1. '
            f'Consider upgrading e3nn >= 0.5.2.',
            stacklevel=2,
        )
    except ImportError:
        pass


def _is_isolated_atom(atoms: Atoms) -> bool:
    """Return True if the structure is an isolated atom.

    nequip cannot train on isolated atoms: they have no neighbors within
    ``r_max`` and crash dataset construction. A single-atom cell is always
    isolated; tags are checked as a secondary signal since the extxyz
    conversion may not preserve every ``info`` key.
    """
    if len(atoms) <= 1:
        return True

    info = atoms.info
    return any(
        str(info.get(key, '')).lower() in ('isolatedatom', 'isolated_atom')
        for key in ('config_type', 'mdb_struct_type', 'phase')
    )


def _parse_metrics_csv(csv_path: Path) -> tuple[float | None, float | None]:
    """Read validation energy/force RMSE (in meV) from a nequip metrics CSV.

    nequip logs one CSV row per train step and one per validation epoch;
    validation columns are empty on train rows and vice versa. We take the
    last row that carries a validation total-energy RMSE. Values are in eV
    (energy) and eV/A (forces); ATLAS reports meV, so both are scaled x1000.
    """
    import csv

    e_col = 'val0_epoch/total_energy_rmse'
    f_col = 'val0_epoch/forces_rmse'

    rmse_e = None
    rmse_f = None
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            e_val = row.get(e_col, '')
            f_val = row.get(f_col, '')
            if e_val not in ('', None):
                rmse_e = float(e_val) * 1000
            if f_val not in ('', None):
                rmse_f = float(f_val) * 1000

    return rmse_e, rmse_f


@register_backend('allegro')
class AllegroBackend:
    """Allegro backend implementing Trainer, Calculator, and Committee."""

    # -- MLIPTrainer --------------------------------------------------------

    @property
    def calcjob_entry_point(self) -> str:
        return 'allegro-train'

    @property
    def parser_entry_point(self) -> str:
        return 'allegro-training-parser'

    def prepare_training_data(
        self,
        path: str | Path,
        structure_list: list[Atoms],
        **kwargs,
    ) -> Path:
        from atlas.active_learning.conversion import gen_mace_train_structure_list

        filtered = [s for s in structure_list if not _is_isolated_atom(s)]

        gen_mace_train_structure_list(
            path=path,
            structure_list=filtered,
            **kwargs,
        )

        # ``gen_mace_train_structure_list`` skips writing when the file
        # already exists, so filter the file on disk as well. Allegro
        # (nequip) cannot handle isolated atoms: they have no neighbors
        # within ``r_max`` and crash dataset construction.
        self._drop_isolated_atoms_from_file(path)

        return Path(path)

    @staticmethod
    def _drop_isolated_atoms_from_file(path: str | Path) -> None:
        """Remove isolated-atom structures from an extxyz training file."""
        from ase.io import read, write

        path = Path(path)
        if not path.exists():
            return

        images = read(str(path), index=':')
        kept = [s for s in images if not _is_isolated_atom(s)]
        if len(kept) != len(images):
            write(str(path), kept, format='extxyz')

    def prepare_builder(
        self,
        builder,
        settings_dict: dict,
        train_data_path: str,
        model_name: str,
        iteration: int,
        db_size: int,
        containerized: bool = False,
    ):
        from aiida import orm

        from atlas.active_learning.backends.allegro.training import (
            build_allegro_train_config,
        )

        config = build_allegro_train_config(
            settings_dict=settings_dict,
            train_data_path=train_data_path,
            model_name=model_name,
            iteration=iteration,
            db_size=db_size,
        )

        builder.train_config_dict = orm.Dict(config)
        builder.model_name = model_name
        builder.train_file_path = train_data_path
        builder.mlip_settings = orm.Dict({'training_backend': 'allegro'})

        return builder

    def select_best_model(
        self,
        training_results: list,
        force_weight: float = 0.1,
    ) -> tuple[str, object, float, float, list[tuple[str, str]]]:
        import numpy as np
        from aiida import orm

        model_name_list = []
        weighted_E_F_sum_list = []

        for calc in training_results:
            curr_calc = orm.load_node(calc.uuid)

            if curr_calc.exit_status != 0:
                continue

            if not hasattr(curr_calc.outputs, 'm_rmse_e') or not hasattr(
                curr_calc.outputs, 'm_rmse_f'
            ):
                continue

            model_name_list.append(curr_calc.inputs.model_name.value)

            weighted_E_F_sum = curr_calc.outputs.m_rmse_e.value + (
                force_weight * curr_calc.outputs.m_rmse_f
            )
            weighted_E_F_sum_list.append(weighted_E_F_sum)

        if not model_name_list:
            raise ValueError(
                'No valid Allegro models were found after training. '
                f'Calculation PKs: {training_results}'
            )

        best_idx = int(np.argmin(weighted_E_F_sum_list))
        best_model_name = model_name_list[best_idx]

        best_model_file = None
        best_rmse_e = None
        best_rmse_f = None
        committee_models = []

        for calc in training_results:
            curr_calc = orm.load_node(calc.uuid)

            if curr_calc.exit_status != 0:
                continue

            model_name = curr_calc.inputs.model_name.value
            model_file = curr_calc.outputs.model_file

            if model_name == best_model_name:
                best_model_file = model_file
                best_rmse_e = curr_calc.outputs.m_rmse_e.value
                best_rmse_f = curr_calc.outputs.m_rmse_f.value
            else:
                committee_models.append((model_name, model_file.uuid))

        return (
            best_model_name,
            best_model_file,
            best_rmse_e,
            best_rmse_f,
            committee_models,
        )

    def create_lammps_potential(self, model_file) -> object | None:
        return model_file

    def run_training(self, config_path: str | Path) -> None:
        import subprocess

        _patch_torch_load_if_needed()

        config_path = Path(config_path)
        # New nequip uses a Hydra CLI: point it at the config directory
        # and name (without the ``.yaml`` suffix). Run as a subprocess so
        # Hydra's global state and output-dir handling stay isolated and
        # the ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`` env var is inherited.
        subprocess.check_call(
            [
                'nequip-train',
                '--config-path',
                str(config_path.parent.resolve()),
                '--config-name',
                config_path.stem,
            ]
        )

    def parse_training_results(self, results_dir: Path) -> dict:
        results_dir = Path(results_dir)
        model_file = None
        best_ckpt = None
        last_ckpt = None
        rmse_e = None
        rmse_f = None
        train_log = None

        for child_file in results_dir.rglob('*'):
            if child_file.name == 'best.ckpt':
                best_ckpt = str(child_file)
            elif child_file.name == 'last.ckpt':
                last_ckpt = str(child_file)
            elif child_file.name == 'metrics.csv':
                train_log = str(child_file)
                rmse_e, rmse_f = _parse_metrics_csv(child_file)

        model_file = best_ckpt or last_ckpt

        # Package the checkpoint into a self-contained ``*.nequip.zip`` here,
        # while the training dataset is still available in the working dir.
        # Downstream consumers (eval/MD/safeguard) can then compile/load the
        # package without the training data. If packaging fails we fall back to
        # the raw checkpoint (which still works when its dataset is present).
        if model_file is not None:
            from atlas.active_learning.backends.allegro.training import (
                package_allegro_model,
            )

            package = package_allegro_model(model_file)
            if package is not None:
                model_file = str(package)

        result = {
            'model_file': model_file,
            'rmse_e': rmse_e,
            'rmse_f': rmse_f,
        }
        if train_log:
            result['train_log'] = train_log
        return result

    # -- MLIPCalculatorFactory ----------------------------------------------

    @property
    def model_file_extension(self) -> str:
        # ``parse_training_results`` packages the checkpoint into a
        # self-contained ``*.nequip.zip`` (see ``package_allegro_model``); that
        # package is the artifact that flows through the pipeline and is
        # compiled on demand for inference. See ``create_calculator``.
        return '.nequip.zip'

    @property
    def lammps_pair_style(self) -> str:
        return 'allegro'

    def create_calculator(
        self,
        model_path: str | Path,
        device: str = 'cpu',
        dtype: str = 'float32',
        **kwargs,
    ) -> Calculator:
        from atlas.active_learning.backends.allegro.calculator import (
            create_allegro_calculator,
        )

        return create_allegro_calculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # -- MLIPModelCompiler --------------------------------------------------

    @property
    def compiled_model_extension(self) -> str:
        # aotinductor emits '.nequip.pt2'; torchscript '.nequip.pth'. The
        # default compile mode is aotinductor (see ``deploy_allegro_model``).
        return '.nequip.pt2'

    def compile_model(
        self,
        model_path: str | Path,
        device: str = 'cpu',
        mode: str = 'aotinductor',
        target: str = 'ase',
    ) -> Path | None:
        """Compile a trained Allegro checkpoint for inference on this node.

        Wraps ``deploy_allegro_model`` (``nequip-compile``), which is
        device-specific and reuses an existing compiled artifact if present.
        """
        from atlas.active_learning.backends.allegro.training import (
            deploy_allegro_model,
        )

        return deploy_allegro_model(
            checkpoint_path=model_path,
            device=device,
            mode=mode,
            target=target,
        )

    # -- MLIPCommitteeEvaluator ---------------------------------------------

    @property
    def supports_committee_training(self) -> bool:
        return True

    def evaluate_committee(
        self,
        structures: list[Atoms],
        model_files: list[str | Path],
        device: str = 'cpu',
        dtype: str = 'float32',
        **kwargs,
    ) -> dict[str, dict[str, list]]:
        from atlas.active_learning.backends.allegro.calculator import (
            create_allegro_calculator,
        )

        comm_results = {}

        for model_path in model_files:
            model_path = Path(model_path)
            model_name = model_path.stem

            comm_results[model_name] = {'REF_energy': [], 'REF_forces': []}

            calculator = create_allegro_calculator(
                model_path=model_path,
                device=device,
                dtype=dtype,
                **kwargs,
            )

            for frame in structures:
                frame.calc = calculator
                comm_results[model_name]['REF_energy'].append(
                    frame.get_potential_energy() * 1000 / len(frame)
                )
                comm_results[model_name]['REF_forces'].append(
                    frame.get_forces() * 1000
                )

        return comm_results
