"""EquiformerV3 descriptor generation.

EquiformerV3 has no public descriptor API, but its final equivariant norm feeds a
scalar energy head whose input is a per-atom invariant node feature (128-dim).
That vector is captured with a forward hook on the energy block and used exactly
like MACE descriptors for latent-space / FPS structure selection.

The features are invariant (verified: symmetry-equivalent atoms agree to float
precision) and chemically discriminative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


def generate_descriptors_equiformer(
    model_path: str,
    database: list[Atoms],
    descriptor_settings: dict,
    outer_average: bool = False,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, list[str]]:
    """Generate per-structure EquiformerV3 descriptors for a database.

    Parameters
    ----------
    model_path : str
        Published checkpoint name (e.g. ``'mptrj_gradient'``) or a checkpoint
        path. A non-existent trained-model path falls back to
        ``descriptor_settings['pretrained_model']`` and then to the default.
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
    from atlas.active_learning.backends.equiformer.calculator import (
        create_equiformer_calculator,
    )

    device = descriptor_settings.get('device', 'cpu')

    # No trained-model file exists for EquiformerV3, so the descriptor model is a
    # published checkpoint: an explicit ``pretrained_model`` setting, else a real
    # checkpoint path if passed, else the backend default (create_equiformer_
    # calculator(None)). The trained-file path convention never exists here.
    model_ref = descriptor_settings.get('pretrained_model')
    if model_ref is None and model_path is not None and Path(str(model_path)).exists():
        model_ref = model_path

    calculator = create_equiformer_calculator(model_ref, device=device)
    model = calculator.trainer.model
    energy_block = dict(model.named_modules())['energy_block']

    captured: dict = {}
    handle = energy_block.register_forward_hook(
        lambda _m, inputs, _o: captured.__setitem__(
            'feat', inputs[0].detach().cpu().numpy()
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
            prepend='EquiformerV3:',
        )
    finally:
        handle.remove()
