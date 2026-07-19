"""Orb backend for ATLAS active learning.

Orb (orb-models) provides pretrained universal interatomic potentials. There is
no ATLAS training path for Orb: it is used through its ASE calculator, driven by
a published potential identifier such as ``'orb-v2'`` or ``'orb-d3-xs-v2'`` (see
``[md.parameters].md_type = "orb:orb-v2"``).

Accordingly this backend implements ``MLIPCalculatorFactory`` and
``MLIPPretrainedModel`` only. It deliberately omits ``MLIPTrainer`` (there is no
training), ``MLIPModelCompiler`` (no compile step) and ``MLIPCommitteeEvaluator``
(a single pretrained potential is not a committee). The isinstance gates on those
protocols then treat Orb correctly without any special-casing.

orb-models is imported lazily inside the calculator, so importing this module
never requires it; a clear "install orb-models" error is raised only when a
calculator is actually built. API verified against orb-models 0.5.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.active_learning.backends import register_backend

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator

#: Well-known published potentials, used for error messages and the default.
#: Advisory only: the authoritative set is ``ORB_PRETRAINED_MODELS`` in the
#: installed orb-models, which ``create_calculator`` validates against.
_KNOWN_PRETRAINED = (
    'orb-v2',
    'orb-d3-v2',
    'orb-d3-sm-v2',
    'orb-d3-xs-v2',
    'orb-mptraj-only-v2',
    'orb-v3-conservative-inf-omat',
    'orb-v3-conservative-20-omat',
    'orb-v3-direct-inf-omat',
    'orb-v3-direct-20-omat',
)


@register_backend('orb')
class OrbBackend:
    """Orb pretrained graph-network potentials (inference only)."""

    # -- MLIPCalculatorFactory ----------------------------------------------

    @property
    def model_file_extension(self) -> str:
        # Orb models are pretrained identifiers, not files ATLAS produces. This
        # extension is never written; it only keeps the protocol satisfied and
        # ``list_inference_models``' glob well-defined (it matches nothing).
        return '.orb'

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
        from atlas.active_learning.backends.orb.calculator import (
            create_orb_calculator,
        )

        return create_orb_calculator(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # -- MLIPPretrainedModel ------------------------------------------------

    @property
    def default_pretrained_model(self) -> str:
        return 'orb-v2'

    @property
    def available_pretrained_models(self) -> tuple[str, ...]:
        return _KNOWN_PRETRAINED

    def resolve_pretrained_model(self, variant: str | None) -> str:
        """Return the orb-models identifier for a variant.

        Orb identifiers are already the dashed names orb-models expects
        (``'orb-v2'``, ``'orb-d3-xs-v2'``), so a given variant is passed through
        unchanged; ``create_calculator`` validates it against the installed
        registry. None selects :attr:`default_pretrained_model`.
        """
        if variant is None:
            return self.default_pretrained_model
        return variant
