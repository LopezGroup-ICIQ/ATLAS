"""MACE ASE calculator creation.

Extracted from ``atlas.active_learning.active_learning_utils`` as part of
the MLIP-agnostic refactoring.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ase.calculators.calculator import Calculator


def create_mace_calculator(
    model_path: str | Path,
    device: str = 'cpu',
    dtype: str = 'float32',
    **kwargs,
) -> Calculator:
    """Create an ASE Calculator from a trained MACE model.

    Parameters
    ----------
    model_path : str | Path
        Path to the MACE ``.model`` file, or a foundation model
        identifier (e.g. ``"mace:mp-small"``).
    device : str
        Device for inference (``'cpu'`` or ``'cuda'``).
    dtype : str
        Data type for inference.
    **kwargs
        Additional keyword arguments passed to ``MACECalculator``.

    Returns
    -------
    Calculator
        An ASE-compatible MACE calculator.
    """
    from mace.calculators import MACECalculator

    model_path = str(model_path)

    # Handle foundation model identifiers
    if 'mace:mp-' in model_path:
        from mace.calculators import mace_mp

        model_variant = model_path.split('mace:mp-')[-1]
        return mace_mp(
            model=model_variant,
            device=device,
            default_dtype=dtype,
        )

    if 'mace:off-' in model_path:
        from mace.calculators import mace_off

        model_variant = model_path.split('mace:off-')[-1]
        return mace_off(
            model=model_variant,
            device=device,
            default_dtype=dtype,
        )

    try:
        model_loaded = torch.load(model_path, map_location=torch.device(device))
    except RuntimeError:
        model_loaded = torch.load(model_path, map_location=torch.device('cpu'))

    return MACECalculator(
        models=[model_loaded],
        device=device,
        default_dtype=dtype,
        **kwargs,
    )
