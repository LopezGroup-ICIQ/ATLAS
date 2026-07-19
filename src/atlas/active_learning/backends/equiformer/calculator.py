"""EquiformerV3 ASE calculator creation.

EquiformerV3 (atomicarchitects/equiformer_v3) is an SE(3)-equivariant graph
transformer trained on materials data (OMat24 / MPtrj / sAlex). It is used
through the OCP-era ``fairchem`` fork bundled in that repository, whose
``OCPCalculator`` rebuilds the model from a checkpoint and drives it as a
standard ASE calculator.

Two things distinguish this from the modern ``fairchem`` backend (UMA/eSEN):

- it uses the OCP-era API ``OCPCalculator(checkpoint_path=...)``, which was
  removed in fairchem-core 2.x (renamed to ``FAIRChemCalculator``); and
- the EquiformerV3 model class lives in the repo's ``experimental`` tree, not in
  the installed ``fairchem-core`` package, so it must be imported to register
  itself before a checkpoint can be rebuilt.

Both stacks therefore need their own container image (see per-backend container
settings). Verified end-to-end against the ``mptrj_gradient`` checkpoint:
bulk Cu4 fcc -> E = -16.39 eV, forces ~ 0.
"""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator

#: Hugging Face repo hosting the (ungated) published EquiformerV3 checkpoints.
_HF_REPO = 'mirror-physics/equiformer_v3'

#: Candidate import paths that register the EquiformerV3 model with fairchem's
#: model registry. The class lives in the repo's ``experimental`` tree, which a
#: deployment may expose under different top-level names, so several are tried.
_MODEL_REGISTER_MODULES = (
    'experimental.models.equiformer_v3.equiformer_v3',
    'equiformer_v3.experimental.models.equiformer_v3.equiformer_v3',
    'models.equiformer_v3.equiformer_v3',
)


def create_equiformer_calculator(
    model_path: str | Path | None = None,
    device: str = 'cpu',
    dtype: str = 'float32',
    **kwargs,
) -> Calculator:
    """Create an ASE calculator from an EquiformerV3 checkpoint.

    Parameters
    ----------
    model_path : str | Path | None
        A published checkpoint name (e.g. ``'mptrj_gradient'``), a path to a
        local ``.pt`` checkpoint, or None for the default checkpoint.
    device : str
        ``'cpu'`` or ``'cuda'`` (mapped to ``OCPCalculator(cpu=...)``).
    dtype : str
        Accepted for interface compatibility; not exposed by OCPCalculator, so
        it is ignored.
    **kwargs
        ``seed`` (int, default 0). Other keys (e.g. the MACE-only
        ``enable_cueq``) are accepted and ignored so the shared MD call site
        does not need to special-case this backend.

    Returns
    -------
    Calculator
        An ``OCPCalculator`` wrapping the EquiformerV3 model.
    """
    try:
        from fairchem.core import OCPCalculator
    except ImportError as exc:
        raise ImportError(
            "The 'equiformer' MLIP backend requires the OCP-era 'fairchem' fork "
            'bundled with atomicarchitects/equiformer_v3 (it provides '
            'OCPCalculator, which the modern fairchem-core 2.x removed). Install '
            'that fork and the equiformer_v3 package. Because its dependency '
            'stack (torch 2.4, e3nn 0.5, scipy<1.16) conflicts with the '
            'MACE/ATLAS environment, run it from its own container image. '
            f'Original error: {exc}'
        ) from exc

    _register_equiformer_model()

    checkpoint = _resolve_checkpoint(model_path)
    return OCPCalculator(
        checkpoint_path=str(checkpoint),
        cpu=not str(device).startswith('cuda'),
        seed=kwargs.get('seed', 0),
    )


def _register_equiformer_model() -> None:
    """Import the EquiformerV3 model module so it registers with fairchem.

    ``OCPCalculator`` rebuilds the model named in the checkpoint config from the
    fairchem model registry. The EquiformerV3 class is not part of the installed
    ``fairchem-core``; it lives in the equiformer_v3 repo's ``experimental``
    tree and only registers when imported.
    """
    import importlib

    for module_name in _MODEL_REGISTER_MODULES:
        try:
            importlib.import_module(module_name)
            return
        except ImportError:
            continue

    raise ImportError(
        'Could not import the EquiformerV3 model module to register it with '
        'fairchem. The equiformer_v3 repository (atomicarchitects/equiformer_v3) '
        'must be installed and importable so that its '
        "'experimental.models.equiformer_v3' module is on the path. Tried: "
        f'{", ".join(_MODEL_REGISTER_MODULES)}.'
    )


def _resolve_checkpoint(model_path: str | Path | None) -> str:
    """Return a local checkpoint path, downloading a published name if needed."""
    from atlas.active_learning.backends.equiformer import (
        DEFAULT_CHECKPOINT,
        published_checkpoint_filename,
    )

    if model_path is not None and Path(str(model_path)).exists():
        return str(model_path)

    name = str(model_path) if model_path else DEFAULT_CHECKPOINT
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Downloading a published EquiformerV3 checkpoint needs 'huggingface_hub'. "
            'Install it, or pass a local checkpoint path instead. '
            f'Original error: {exc}'
        ) from exc

    return hf_hub_download(_HF_REPO, published_checkpoint_filename(name))
