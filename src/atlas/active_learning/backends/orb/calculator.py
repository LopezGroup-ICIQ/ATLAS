"""Orb ASE calculator creation.

Orb (orb-models) ships pretrained universal potentials with an ASE calculator.
Unlike MACE/Allegro there is no ATLAS training path: the "model" is a published
identifier such as ``'orb-v2'`` or ``'orb-d3-xs-v2'`` (see ``OrbBackend``), which
orb-models resolves and downloads. A local checkpoint may also be supplied.

API verified against orb-models 0.5.5.
"""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator


def create_orb_calculator(
    model_path: str | Path | None = None,
    device: str = 'cpu',
    dtype: str = 'float32',
    **kwargs,
) -> Calculator:
    """Create an ASE calculator from an Orb pretrained potential or checkpoint.

    Parameters
    ----------
    model_path : str | Path | None
        A published potential name (e.g. ``'orb-v2'``, ``'orb-d3-xs-v2'``), a
        path to a downloaded checkpoint, or None for the default potential.
    device : str
        Device for inference (``'cpu'`` or ``'cuda'``).
    dtype : str
        Accepted for interface compatibility with the other backends; Orb does
        not expose a dtype knob, so it is ignored.
    **kwargs
        ``compile`` (bool, default False) enables ``torch.compile``; other keys
        (e.g. the MACE-only ``enable_cueq``) are accepted and ignored so the
        shared MD call site does not need to special-case this backend.

    Returns
    -------
    Calculator
        An ASE-compatible ``ORBCalculator``.
    """
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
    except ImportError as exc:
        raise ImportError(
            "The 'orb' MLIP backend requires the 'orb-models' package, which is "
            "not installed. Install it (pip install orb-models) to use "
            "model_type='orb'. Because Orb's dependencies conflict with the "
            'MACE/ATLAS environment, it is best run from its own container '
            f'image. Original error: {exc}'
        ) from exc

    # torch.compile is opt-in: it is faster on GPU but needs a working C++
    # toolchain (torch inductor), which is not guaranteed on every host.
    compile_model = kwargs.get('compile', False)

    if model_path is None:
        orbff = pretrained.orb_v2(device=device, compile=compile_model)
    elif _is_pretrained_name(model_path):
        factory = _resolve_pretrained_factory(str(model_path))
        orbff = factory(device=device, compile=compile_model)
    else:
        orbff = pretrained.orb_v2(
            weights_path=str(model_path), device=device, compile=compile_model
        )

    return ORBCalculator(orbff, device=device)


def _is_pretrained_name(model_path: str | Path) -> bool:
    """Return True if ``model_path`` is a published potential name, not a file."""
    return isinstance(model_path, str) and not Path(model_path).exists()


def _resolve_pretrained_factory(variant: str):
    """Return the orb-models loader for a published potential name.

    v3 potentials are registered in ``ORB_PRETRAINED_MODELS`` under dashed keys;
    older ones (``orb_v2``, ``orb_d3_*_v2``) are module-level functions with
    underscore names.
    """
    from orb_models.forcefield import pretrained

    registry = getattr(pretrained, 'ORB_PRETRAINED_MODELS', {})
    if variant in registry:
        return registry[variant]

    func_name = variant.replace('-', '_')
    if hasattr(pretrained, func_name):
        return getattr(pretrained, func_name)

    available = sorted(registry)
    raise ValueError(
        f"Unknown Orb pretrained potential '{variant}'. "
        f'Available: {available} plus module loaders such as orb_v2.'
    )
