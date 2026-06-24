"""MACE backend for ATLAS active learning.

This module provides the concrete MACE implementation of the MLIP backend
protocols. It wraps existing MACE-specific functions from the codebase,
delegating to them without duplicating logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atlas.active_learning.backends import register_backend

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@register_backend('mace')
class MACEBackend:
    """MACE backend implementing all four MLIP protocols.

    This backend delegates to the existing MACE-specific functions in
    ``active_learning_utils``, ``conversion``, and ``mace_tools_aiida``.
    """

    # -- MLIPTrainer --------------------------------------------------------

    @property
    def calcjob_entry_point(self) -> str:
        return 'mace-train'

    @property
    def parser_entry_point(self) -> str:
        return 'mace-training-parser'

    def prepare_training_data(
        self,
        path: str | Path,
        structure_list: list[Atoms],
        **kwargs,
    ) -> Path:
        from atlas.active_learning.conversion import gen_mace_train_structure_list

        gen_mace_train_structure_list(
            path=path,
            structure_list=structure_list,
            **kwargs,
        )
        return Path(path)

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

        from atlas.active_learning.backends.mace.training import (
            update_mace_train_settings_dict,
        )

        mace_train_settings = update_mace_train_settings_dict(
            settings_dict=settings_dict,
            train_data_path=train_data_path,
            curr_model=model_name,
            curr_iter=iteration,
            db_size=db_size,
            containerized=containerized,
        )

        builder.mace_settings_dict = orm.Dict(mace_train_settings)
        builder.model_name = model_name
        builder.mace_train_file_path = train_data_path

        return builder

    def select_best_model(
        self,
        training_results: list,
        force_weight: float = 0.1,
    ) -> tuple[str, object, float, float, list[tuple[str, str]]]:
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
                'No valid MLIP models were found after training. '
                f'Check for issues in the training step. '
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
        from atlas.active_learning.backends.mace.training import (
            create_mace_lammps_model_impl,
        )

        return create_mace_lammps_model_impl(model_file)

    # -- MLIPCalculatorFactory ----------------------------------------------

    @property
    def model_file_extension(self) -> str:
        return '.model'

    def create_calculator(
        self,
        model_path: str | Path,
        device: str = 'cpu',
        dtype: str = 'float32',
        **kwargs,
    ) -> Calculator:
        from atlas.active_learning.backends.mace.calculator import (
            create_mace_calculator,
        )

        return create_mace_calculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # -- MLIPDescriptorProvider ---------------------------------------------

    def generate_descriptors(
        self,
        database: list[Atoms],
        model_path: str | Path | None,
        settings: dict,
        **kwargs,
    ) -> tuple[dict, np.ndarray, list[str]]:
        from atlas.active_learning.backends.mace.descriptors import (
            generate_descriptors_mace,
        )

        return generate_descriptors_mace(
            model_path=model_path,
            database=database,
            descriptor_settings=settings,
            outer_average=kwargs.get('outer_average', False),
            verbose=kwargs.get('verbose', False),
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
        import torch
        from mace.calculators import MACECalculator

        batch_size = kwargs.get('batch_size', 12)
        comm_results = {}

        for model_path in model_files:
            model_path = Path(model_path)
            model_name = model_path.stem

            comm_results[model_name] = {'REF_energy': [], 'REF_forces': []}

            model_loaded = torch.load(model_path, map_location=torch.device(device))
            calculator = MACECalculator(
                models=[model_loaded],
                device=device,
                default_dtype=dtype,
                batch_size=batch_size,
            )

            for frame in structures:
                frame.calc = calculator
                comm_results[model_name]['REF_energy'].append(
                    frame.get_potential_energy() * 1000 / len(frame)
                )
                comm_results[model_name]['REF_forces'].append(frame.get_forces() * 1000)

        return comm_results
