"""Allegro/NequIP ASE calculator creation."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator


def create_allegro_calculator(
    model_path: str | Path,
    device: str = 'cpu',
    dtype: str = 'float32',
    **kwargs,
) -> Calculator:
    """Create an ASE Calculator from a compiled Allegro/NequIP model.

    Targets the new nequip API: the model must be a compiled artifact
    produced by ``nequip-compile`` (see
    :func:`atlas.active_learning.backends.allegro.training.deploy_allegro_model`),
    loaded via ``NequIPCalculator.from_compiled_model``.

    Parameters
    ----------
    model_path : str | Path
        Path to a compiled model file (e.g. ``*.nequip.pt2``).
    device : str
        Device for inference (``'cpu'`` or ``'cuda'``).
    dtype : str
        Data type for inference (kept for interface compatibility).

    Returns
    -------
    Calculator
        An ASE-compatible NequIP calculator.
    """
    from nequip.ase import NequIPCalculator

    species_map = kwargs.pop('chemical_species_to_atom_type_map', None)

    calc_kwargs = {}
    if species_map is not None:
        calc_kwargs['chemical_species_to_atom_type_map'] = species_map

    return NequIPCalculator.from_compiled_model(
        compile_path=str(model_path),
        device=device,
        **calc_kwargs,
    )
