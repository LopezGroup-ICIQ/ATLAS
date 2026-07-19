"""EquiformerV3 backend for ATLAS active learning.

EquiformerV3 (atomicarchitects/equiformer_v3) is an SE(3)-equivariant graph
transformer with published, ungated checkpoints trained on materials data
(OMat24 / MPtrj / sAlex). There is no ATLAS training path; it is used for
inference through the OCP-era ``fairchem`` fork's ``OCPCalculator``
(``[md.parameters].md_type = "equiformer:mptrj_gradient"`` or a local
checkpoint path).

Like Orb and the modern ``fairchem`` backend it implements
``MLIPCalculatorFactory`` and ``MLIPPretrainedModel`` only. It is a distinct
backend from ``fairchem`` because it needs the OCP-era fairchem fork
(``OCPCalculator``), not fairchem-core 2.x -- a separate, conflicting
dependency stack that must run from its own container image.

fairchem is imported lazily inside the calculator, so importing this module
never requires it. Verified end-to-end against the ``mptrj_gradient`` checkpoint
(bulk Cu4 fcc -> E = -16.39 eV, forces ~ 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.active_learning.backends import register_backend

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator

#: Default published checkpoint: the MPtrj-fine-tuned gradient model (materials,
#: conservative forces).
DEFAULT_CHECKPOINT = 'mptrj_gradient'

#: Published checkpoints on the Hugging Face repo (all under ``checkpoint/``).
_KNOWN_CHECKPOINTS = (
    'mptrj_gradient',
    'omat24_direct',
    'omat24_gradient',
    'omat24-mptrj-salex_gradient',
)


def published_checkpoint_filename(name: str) -> str:
    """Return the Hugging Face path for a published checkpoint name.

    Accepts a bare name (``'mptrj_gradient'``) or one already carrying the
    ``checkpoint/`` prefix and/or ``.pt`` suffix.
    """
    stem = name.removeprefix('checkpoint/').removesuffix('.pt')
    return f'checkpoint/{stem}.pt'


@register_backend('equiformer')
class EquiformerBackend:
    """EquiformerV3 pretrained potentials (inference only)."""

    # -- MLIPCalculatorFactory ----------------------------------------------

    @property
    def model_file_extension(self) -> str:
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
        from atlas.active_learning.backends.equiformer.calculator import (
            create_equiformer_calculator,
        )

        return create_equiformer_calculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # -- MLIPPretrainedModel ------------------------------------------------

    @property
    def default_pretrained_model(self) -> str:
        return DEFAULT_CHECKPOINT

    @property
    def available_pretrained_models(self) -> tuple[str, ...]:
        return _KNOWN_CHECKPOINTS

    def resolve_pretrained_model(self, variant: str | None) -> str:
        """Return the checkpoint name for a variant.

        A variant is a published checkpoint name (``'mptrj_gradient'``) passed
        through unchanged; ``create_calculator`` downloads it, or accepts a
        local ``.pt`` path. None selects :attr:`default_pretrained_model`.
        """
        if variant is None:
            return self.default_pretrained_model
        return variant
