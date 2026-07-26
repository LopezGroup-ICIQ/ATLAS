"""Orb ASE calculator creation.

Orb (orb-models) ships pretrained universal potentials with an ASE calculator.
Unlike MACE/Allegro there is no ATLAS training path: the "model" is a published
identifier such as ``'orb-v2'`` or ``'orb-d3-xs-v2'`` (see ``OrbBackend``), which
orb-models resolves and downloads. A local checkpoint may also be supplied.

API verified against orb-models 0.5.5.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

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
    calc, _ = build_orb_calculator_and_model(
        model_path, device=device, compile_model=kwargs.get('compile', False)
    )
    return calc


def build_orb_forcefield(
    model_path: str | Path | None = None,
    device: str = 'cpu',
    compile_model: bool = False,
):
    """Build the Orb forcefield model behind the calculator.

    Lowest-level loader shared by :func:`load_orb_model` (and through it
    :func:`build_orb_calculator_and_model` and :func:`create_orb_calculator`)
    and the descriptor/uncertainty providers, which need the model itself to
    hook its encoder / read its heads. ``compile_model`` (torch.compile) is
    opt-in: faster on GPU but needs a working C++ toolchain (torch inductor).
    """
    try:
        from orb_models.forcefield import pretrained
    except ImportError as exc:
        raise ImportError(
            "The 'orb' MLIP backend requires the 'orb-models' package, which is "
            'not installed. Install it (pip install orb-models) to use '
            "model_type='orb'. Because Orb's dependencies conflict with the "
            'MACE/ATLAS environment, it is best run from its own container '
            f'image. Original error: {exc}'
        ) from exc

    if model_path is None:
        return pretrained.orb_v2(device=device, compile=compile_model)
    if _is_pretrained_name(model_path):
        factory = _resolve_pretrained_factory(str(model_path))
        return factory(device=device, compile=compile_model)
    return pretrained.orb_v2(
        weights_path=str(model_path), device=device, compile=compile_model
    )


def load_orb_model(
    model_path: str | Path | None = None,
    device: str = 'cpu',
    compile_model: bool = False,
) -> tuple[Any, Any]:
    """Load an Orb forcefield and unwrap the ``(model, adapter)`` tuple.

    Mirrors the model-loading step MACE performs with ``torch.load``: it returns
    the bare model needed to hook the encoder (descriptors) or read the
    confidence heads (uncertainty), plus the atoms adapter orb-models >= 0.4
    bundles alongside it (``None`` for older versions and for v2 loaders that
    return the model directly).
    """
    orbff = build_orb_forcefield(
        model_path, device=device, compile_model=compile_model
    )
    model = orbff
    atoms_adapter = None
    if isinstance(orbff, tuple):
        model = orbff[0]
        if len(orbff) > 1:
            atoms_adapter = orbff[1]
    return model, atoms_adapter


def _wrap_orb_calculator(
    model: Any, atoms_adapter: Any | None, device: str = 'cpu'
) -> Calculator:
    """Wrap a bare Orb model in an ASE ``ORBCalculator`` (version-agnostic).

    Handles the import-path move (>= 0.4: ``inference.calculator``; <= 0.3:
    ``calculator``) and the ``atoms_adapter`` vs ``adapter`` kwarg rename, and
    falls back to a default ``AseAtomsAdapter`` when the loader did not supply
    one. Keep this the single place that knows about the orb-models API drift.
    """
    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ModuleNotFoundError:
        from orb_models.forcefield.calculator import ORBCalculator

    sig = inspect.signature(ORBCalculator.__init__)
    if 'atoms_adapter' in sig.parameters:
        if atoms_adapter is None:
            atoms_adapter = _get_default_atoms_adapter()
        return ORBCalculator(model, atoms_adapter=atoms_adapter, device=device)
    elif 'adapter' in sig.parameters:
        if atoms_adapter is None:
            atoms_adapter = _get_default_atoms_adapter()
        return ORBCalculator(model, adapter=atoms_adapter, device=device)

    return ORBCalculator(model, device=device)


def build_orb_calculator_and_model(
    model_path: str | Path | None = None,
    device: str = 'cpu',
    compile_model: bool = False,
) -> tuple[Calculator, Any]:
    """Build the Orb ASE calculator and return the bare model alongside it.

    Combines :func:`load_orb_model` (load + tuple unwrap) and
    :func:`_wrap_orb_calculator` (version-agnostic ``ORBCalculator``
    construction). Consumers that need to introspect the model (descriptors via
    a forward hook on the encoder; uncertainty via the confidence head) use this
    instead of :func:`create_orb_calculator`, which returns the calculator only.
    """
    model, atoms_adapter = load_orb_model(
        model_path, device=device, compile_model=compile_model
    )
    calc = _wrap_orb_calculator(model, atoms_adapter, device=device)
    return calc, model


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


def _get_default_atoms_adapter() -> Any:
    """Attempt to instantiate a default AseAtomsAdapter across orb-models versions."""
    for module_path in (
        'orb_models.forcefield.atomic_system',
        'orb_models.forcefield.inference.atoms_adapter',
        'orb_models.forcefield.inference.adapter',
        'orb_models.forcefield.inference.calculator',
        'orb_models.forcefield.atoms_adapter',
    ):
        try:
            mod = __import__(module_path, fromlist=['AseAtomsAdapter'])
            if hasattr(mod, 'AseAtomsAdapter'):
                return mod.AseAtomsAdapter()
        except (ImportError, ModuleNotFoundError):
            continue
    return None
