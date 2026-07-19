"""Orb descriptor generation.

Orb has no public descriptor API, but its graph-network encoder produces per-atom
invariant node features (256-dim for orb-v2/orb-d3-*). They are captured with a
forward hook on the model's encoder and used exactly like MACE descriptors for
latent-space / FPS structure selection.

The node features are rotation/translation invariant (verified: symmetry-
equivalent atoms give bit-identical vectors) and chemically discriminative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


def generate_descriptors_orb(
    model_path: str,
    database: list[Atoms],
    descriptor_settings: dict,
    outer_average: bool = False,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, list[str]]:
    """Generate per-structure Orb descriptors for a database.

    Parameters
    ----------
    model_path : str
        Published potential name (e.g. ``'orb-d3-xs-v2'``) or a checkpoint path.
        A non-existent path (the trained-model file convention, which Orb does
        not use) falls back to ``descriptor_settings['pretrained_model']`` and
        then to Orb's default.
    database : list[Atoms]
        Structures to describe.
    descriptor_settings : dict
        ``device`` and optionally ``pretrained_model``.
    outer_average : bool
        Average per-atom descriptors into one structure-level vector.
    verbose : bool
        Show a progress bar.

    Returns
    -------
    tuple[dict, np.ndarray, list[str]]
        ``(descriptor_dict, descriptor_arr, uuid_list)``; see
        ``descriptor_utils.assemble_structure_descriptors``.
    """
    from pathlib import Path

    from atlas.active_learning.backends.descriptor_utils import (
        assemble_structure_descriptors,
    )
    from atlas.active_learning.backends.orb.calculator import build_orb_forcefield

    device = descriptor_settings.get('device', 'cpu')

    # Orb produces no trained-model file, so the descriptor model is a pretrained
    # potential: an explicit ``pretrained_model`` setting, else a real checkpoint
    # path if one was passed, else Orb's default (via build_orb_forcefield(None)).
    # The trained-file path convention (e.g. 'curr_model.orb') never exists here.
    model_ref = descriptor_settings.get('pretrained_model')
    if model_ref is None and model_path is not None and Path(str(model_path)).exists():
        model_ref = model_path

    orbff = build_orb_forcefield(model_ref, device=device, compile_model=False)

    from orb_models.forcefield.calculator import ORBCalculator

    calculator = ORBCalculator(orbff, device=device)
    encoder = dict(orbff.named_modules())['model._encoder']

    captured: dict = {}
    handle = encoder.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__(
            'feat', out[0].detach().cpu().numpy()
        )
    )
    try:

        def extract(struct: Atoms) -> np.ndarray:
            struct.calc = calculator
            struct.get_potential_energy()
            return captured['feat']

        return assemble_structure_descriptors(
            database,
            extract,
            outer_average=outer_average,
            verbose=verbose,
            prepend='Orb:',
        )
    finally:
        handle.remove()
