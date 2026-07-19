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

The OCP-era fork also bundles the **EquiformerV2** model code, so this backend
loads EquiformerV2 checkpoints too, by published name
(``md_type = "equiformer:eqV2_31M_omat"``, downloaded from the gated
``fairchem/OMAT24`` repo) or a local path
(``md_type = "equiformer:/path/to/eqV2.pt"``). V2 registers automatically from
the fork's core (no experimental import). Verified end-to-end against the
OMat24-trained ``eqV2_31M_omat`` checkpoint: bulk Cu4 gives -3.74 eV/atom with a
lattice-constant minimum at ~3.6 A (Cu's equilibrium), i.e. physically correct
materials predictions.

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

#: Default published checkpoint: the MPtrj-fine-tuned EquiformerV3 gradient model
#: (materials, conservative forces). Ungated.
DEFAULT_CHECKPOINT = 'mptrj_gradient'

#: Hugging Face repo hosting the ungated EquiformerV3 checkpoints.
_EQV3_REPO = 'mirror-physics/equiformer_v3'
#: Hugging Face repo hosting the EquiformerV2 OMat24 checkpoints. Gated by manual
#: review: request access at https://huggingface.co/fairchem/OMAT24.
_EQV2_REPO = 'fairchem/OMAT24'

#: Published checkpoints, mapping a name to ``(hf_repo, filename_in_repo)``.
#: EquiformerV3 checkpoints (ungated) sit under ``checkpoint/`` in the mirror
#: repo; EquiformerV2 OMat24 checkpoints (gated) sit at the root of fairchem/OMAT24.
PUBLISHED_CHECKPOINTS = {
    # EquiformerV3 (ungated)
    'mptrj_gradient': (_EQV3_REPO, 'checkpoint/mptrj_gradient.pt'),
    'omat24_direct': (_EQV3_REPO, 'checkpoint/omat24_direct.pt'),
    'omat24_gradient': (_EQV3_REPO, 'checkpoint/omat24_gradient.pt'),
    'omat24-mptrj-salex_gradient': (
        _EQV3_REPO,
        'checkpoint/omat24-mptrj-salex_gradient.pt',
    ),
    # EquiformerV2 on OMat24 (gated repo; sizes 31M/86M/153M).
    'eqV2_31M_mp': (_EQV2_REPO, 'eqV2_31M_mp.pt'),
    'eqV2_31M_omat': (_EQV2_REPO, 'eqV2_31M_omat.pt'),
    'eqV2_31M_omat_mp_salex': (_EQV2_REPO, 'eqV2_31M_omat_mp_salex.pt'),
    'eqV2_86M_omat': (_EQV2_REPO, 'eqV2_86M_omat.pt'),
    'eqV2_153M_omat': (_EQV2_REPO, 'eqV2_153M_omat.pt'),
    'eqV2_153M_omat_mp_salex': (_EQV2_REPO, 'eqV2_153M_omat_mp_salex.pt'),
}


def published_checkpoint_filename(name: str) -> str:
    """Return the EquiformerV3 mirror-repo path for a checkpoint name.

    Used for names not in :data:`PUBLISHED_CHECKPOINTS` (assumed to follow the
    V3 mirror layout). Accepts a bare name (``'mptrj_gradient'``) or one already
    carrying the ``checkpoint/`` prefix and/or ``.pt`` suffix.
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
        return tuple(PUBLISHED_CHECKPOINTS)

    def resolve_pretrained_model(self, variant: str | None) -> str:
        """Return the checkpoint name for a variant.

        A variant is a published checkpoint name (``'mptrj_gradient'``) passed
        through unchanged; ``create_calculator`` downloads it, or accepts a
        local ``.pt`` path. None selects :attr:`default_pretrained_model`.
        """
        if variant is None:
            return self.default_pretrained_model
        return variant

    # -- MLIPDescriptorProvider ---------------------------------------------

    def generate_descriptors(
        self,
        database,
        model_path,
        settings: dict,
        outer_average: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        from atlas.active_learning.backends.equiformer.descriptors import (
            generate_descriptors_equiformer,
        )

        return generate_descriptors_equiformer(
            model_path=model_path,
            database=database,
            descriptor_settings=settings,
            outer_average=outer_average,
            verbose=verbose,
        )
