"""fairchem backend for ATLAS active learning.

fairchem (fairchem-core) provides Meta's pretrained potentials: UMA, eSEN and
AllSCAIP. There is no ATLAS training path for them; they are used through the
fairchem ASE calculator, driven by a published model identifier such as
``'uma-s-1p2'`` (``[md.parameters].md_type = "fairchem:uma-s-1p2"``).

This backend therefore implements ``MLIPCalculatorFactory`` and
``MLIPPretrainedModel`` only, and omits ``MLIPTrainer``, ``MLIPModelCompiler``
and ``MLIPCommitteeEvaluator`` -- mirroring the Orb backend.

Named ``fairchem`` (not ``equiformer``): EquiformerV2 was a fairchem/OCP v1
model and is not part of fairchem-core 2.x, whose materials-capable models are
UMA and eSEN.

fairchem-core is imported lazily inside the calculator, so importing this module
never requires it; a clear "install fairchem-core" error is raised only when a
calculator is actually built. API verified against fairchem-core 2.21.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.active_learning.backends import register_backend

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator

#: Published models known to fairchem-core 2.21. Advisory only: the authoritative
#: set is ``pretrained_mlip.available_models`` in the installed fairchem, which
#: ``create_calculator`` validates against. UMA models are materials-capable
#: (multi-task, use the 'omat' task); eSEN/AllSCAIP are single-task.
_KNOWN_MODELS = (
    'uma-s-1p2',
    'uma-s-1p1',
    'uma-m-1p1',
    'esen-md-direct-all-omol',
    'esen-sm-conserving-all-omol',
    'esen-sm-direct-all-omol',
    'allscaip-md-conserving-all-omol',
    'allscaip-md-direct-all-omol',
    'esen-sm-conserving-all-oc25',
    'esen-md-direct-all-oc25',
    'esen-sm-filtered-odac25',
    'esen-sm-full-odac25',
)


@register_backend('fairchem')
class FairchemBackend:
    """fairchem pretrained potentials -- UMA/eSEN/AllSCAIP (inference only)."""

    # -- MLIPCalculatorFactory ----------------------------------------------

    @property
    def model_file_extension(self) -> str:
        # fairchem models are pretrained identifiers, not files ATLAS produces.
        # Kept only to satisfy the protocol and keep list_inference_models'
        # glob well-defined (it matches nothing).
        return '.pt'

    @property
    def lammps_pair_style(self) -> str:
        return ''

    def create_calculator(
        self,
        model_path: str | Path | None = None,
        device: str = 'cpu',
        dtype: str = 'float32',
        **kwargs,
    ) -> Calculator:
        from atlas.active_learning.backends.fairchem.calculator import (
            create_fairchem_calculator,
        )

        return create_fairchem_calculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # -- MLIPPretrainedModel ------------------------------------------------

    @property
    def default_pretrained_model(self) -> str:
        return 'uma-s-1p2'

    @property
    def available_pretrained_models(self) -> tuple[str, ...]:
        return _KNOWN_MODELS

    def resolve_pretrained_model(self, variant: str | None) -> str:
        """Return the fairchem model identifier for a variant.

        fairchem identifiers are the published names (``'uma-s-1p2'``,
        ``'esen-sm-conserving-all-omol'``), so a variant is passed through
        unchanged; ``create_calculator`` validates it against the installed
        ``available_models``. None selects :attr:`default_pretrained_model`.
        """
        if variant is None:
            return self.default_pretrained_model
        return variant
