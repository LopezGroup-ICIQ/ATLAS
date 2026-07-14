"""Protocol definitions for MLIP backend interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


@runtime_checkable
class MLIPTrainer(Protocol):
    """Protocol for MLIP training backends.

    Backends that support model training implement this protocol to provide
    the AiiDA CalcJob entry point, training data preparation, builder
    configuration, and model selection logic.
    """

    @property
    def calcjob_entry_point(self) -> str:
        """AiiDA entry point string for the training CalcJob."""
        ...

    @property
    def parser_entry_point(self) -> str:
        """AiiDA entry point string for the training parser."""
        ...

    def prepare_training_data(
        self,
        path: str | Path,
        structure_list: list[Atoms],
        **kwargs,
    ) -> Path:
        """Convert structure list to model-specific training format.

        Parameters
        ----------
        path : str | Path
            Output file path for the training data.
        structure_list : list[Atoms]
            List of ASE Atoms objects with DFT reference data.

        Returns
        -------
        Path
            Path to the written training data file.
        """
        ...

    def prepare_builder(
        self,
        builder: Any,
        settings_dict: dict,
        train_data_path: str,
        model_name: str,
        iteration: int,
        db_size: int,
        containerized: bool = False,
    ) -> Any:
        """Populate the AiiDA CalcJob builder with model-specific inputs.

        Parameters
        ----------
        builder : ProcessBuilder
            The AiiDA CalcJob builder to populate.
        settings_dict : dict
            Model-specific training settings from the TOML config.
        train_data_path : str
            Path to the training data file.
        model_name : str
            Unique name for this training run.
        iteration : int
            Current AL loop iteration number.
        db_size : int
            Number of structures in the training database.
        containerized : bool
            Whether to use containerized execution.

        Returns
        -------
        ProcessBuilder
            The populated builder, ready for submission.
        """
        ...

    def select_best_model(
        self,
        training_results: list,
        force_weight: float = 0.1,
    ) -> tuple[str, Any, float, float, list[tuple[str, str]]]:
        """Select the best model from N trained models.

        Parameters
        ----------
        training_results : list
            List of completed AiiDA CalcJobNode results.
        force_weight : float
            Weight for forces in the RMSE-based selection criterion.

        Returns
        -------
        tuple
            Tuple of (best_model_name, best_model_file, rmse_e, rmse_f,
            committee_models_tupl_name_uuid). The last element is a list
            of (name, uuid) tuples for the non-best models.
        """
        ...

    def create_lammps_potential(self, model_file: Any) -> Any | None:
        """Convert a trained model to LAMMPS-compatible format.

        Parameters
        ----------
        model_file : SinglefileData
            The trained model file.

        Returns
        -------
        SinglefileData | None
            LAMMPS potential file, or None if not supported.
        """
        ...

    def run_training(self, config_path: str | Path) -> None:
        """Execute model training from a config file.

        Called by the remote training script on the HPC node.
        Each backend dispatches to its own training CLI.

        Parameters
        ----------
        config_path : str | Path
            Path to the backend-specific training config file.
        """
        ...

    def parse_training_results(self, results_dir: Path) -> dict:
        """Extract model file path and metrics from training output.

        Called by the remote training script after training completes.
        Each backend knows its own output format.

        Parameters
        ----------
        results_dir : Path
            Directory containing training outputs.

        Returns
        -------
        dict
            Dictionary with keys 'model_file' (str), 'rmse_e' (float,
            meV/atom), 'rmse_f' (float, meV/A), and optionally
            'train_log' (str).
        """
        ...


@runtime_checkable
class MLIPCalculatorFactory(Protocol):
    """Protocol for creating ASE calculators from trained MLIP models."""

    @property
    def model_file_extension(self) -> str:
        """File extension for model files (e.g. '.model' for MACE)."""
        ...

    @property
    def lammps_pair_style(self) -> str:
        """LAMMPS pair_style string for this backend (e.g. 'mace', 'allegro')."""
        ...

    def create_calculator(
        self,
        model_path: str | Path,
        device: str = 'cpu',
        dtype: str = 'float32',
        **kwargs,
    ) -> Calculator:
        """Create an ASE Calculator from a model file on disk.

        Parameters
        ----------
        model_path : str | Path
            Path to the model file.
        device : str
            Device for inference ('cpu' or 'cuda').
        dtype : str
            Data type for inference.

        Returns
        -------
        Calculator
            An ASE-compatible calculator.
        """
        ...


@runtime_checkable
class MLIPDescriptorProvider(Protocol):
    """Protocol for models that can provide structural descriptors.

    Not all MLIP backends support descriptor extraction. Backends that
    do not can omit this protocol, and the workflow will fall back to
    SOAP descriptors.
    """

    def generate_descriptors(
        self,
        database: list[Atoms],
        model_path: str | Path | None,
        settings: dict,
        **kwargs,
    ) -> tuple[dict, np.ndarray, list[str]]:
        """Generate structural descriptors for a list of structures.

        Parameters
        ----------
        database : list[Atoms]
            List of ASE Atoms objects.
        model_path : str | Path | None
            Path to the model file for descriptor extraction.
        settings : dict
            Descriptor generation settings.

        Returns
        -------
        tuple
            (descriptor_dict, descriptor_array, uuid_list) where
            descriptor_dict is keyed by atl_id, descriptor_array is a
            vertically stacked numpy array, and uuid_list contains the
            UUIDs of newly assigned structures.
        """
        ...


@runtime_checkable
class MLIPCommitteeEvaluator(Protocol):
    """Protocol for committee-based uncertainty quantification.

    Backends that support training multiple models with different seeds
    implement this protocol to evaluate structures with the committee
    and compute disagreement statistics.
    """

    @property
    def supports_committee_training(self) -> bool:
        """Whether this backend supports training multiple committee models."""
        ...

    def evaluate_committee(
        self,
        structures: list[Atoms],
        model_files: list[str | Path],
        device: str = 'cpu',
        dtype: str = 'float32',
        **kwargs,
    ) -> dict[str, dict[str, list]]:
        """Evaluate structures with multiple committee models.

        Parameters
        ----------
        structures : list[Atoms]
            Structures to evaluate.
        model_files : list[str | Path]
            Paths to committee model files.
        device : str
            Device for inference.
        dtype : str
            Data type for inference.

        Returns
        -------
        dict
            Dictionary keyed by model name, each containing 'REF_energy'
            (meV/atom) and 'REF_forces' (meV/A) lists.
        """
        ...


@runtime_checkable
class MLIPModelCompiler(Protocol):
    """Protocol for backends whose models must be compiled before inference.

    Some backends (e.g. Allegro/NequIP) cannot run inference from the raw
    trained checkpoint. It must first be compiled (``nequip-compile``) into a
    device-specific artifact. Compilation is expensive and toolchain-sensitive,
    so it is best done once, on the computer where inference will run, and
    the compiled artifact reused there.

    Backends that need no compilation (e.g. MACE) simply omit this protocol;
    callers gate on ``isinstance(backend, MLIPModelCompiler)`` and, when absent,
    ship the raw model unchanged.
    """

    def compile_model(
        self,
        model_path: str | Path,
        device: str = 'cpu',
        mode: str = 'aotinductor',
        target: str = 'ase',
    ) -> Path | None:
        """Compile a trained model into an inference-ready artifact.

        Runs on the compute node where the artifact will be used, so the
        compiled result matches that node's device/toolchain.

        Parameters
        ----------
        model_path : str | Path
            Path to the trained model/checkpoint.
        device : str
            Target device for the compiled model ('cpu' or 'cuda'). The
            artifact is device-specific.
        mode : str
            Compilation backend (e.g. 'aotinductor').
        target : str
            Inference target the artifact is compiled for (e.g. 'ase').

        Returns
        -------
        Path | None
            Path to the compiled artifact, or None on failure.
        """
        ...

    @property
    def compiled_model_extension(self) -> str:
        """File extension of the compiled artifact (e.g. '.nequip.pt2')."""
        ...
