"""MLIP backend registry for model-agnostic active learning.

This module provides a registry pattern for MLIP backends. Each backend
(MACE, NequIP, Allegro, DeepMD, etc.) registers itself and can then be
looked up by name from the TOML configuration.

Usage
-----
    from atlas.active_learning.backends import get_backend

    backend = get_backend('mace')
    calculator = backend.create_calculator(model_path, device='cuda')
"""

from __future__ import annotations

from atlas.active_learning.backends._base import (
    MLIPCalculatorFactory,
    MLIPCommitteeEvaluator,
    MLIPDescriptorProvider,
    MLIPModelCompiler,
    MLIPTrainer,
)

_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str):
    """Decorator to register an MLIP backend class.

    Parameters
    ----------
    name : str
        Short name for the backend (e.g. 'mace', 'nequip', 'allegro').
    """

    def decorator(cls):
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str):
    """Instantiate and return a backend by name.

    Parameters
    ----------
    name : str
        Registered backend name.

    Raises
    ------
    ValueError
        If the backend name is not registered.
    """
    if name not in _BACKEND_REGISTRY:
        available = ', '.join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown MLIP backend '{name}'. Available backends: {available}"
        )
    return _BACKEND_REGISTRY[name]()


def list_backends() -> list[str]:
    """Return the list of registered backend names."""
    return sorted(_BACKEND_REGISTRY.keys())


# Model file extensions, longest/compound first so that e.g. a nequip package
# ('foo.nequip.zip') is matched as '.nequip.zip' rather than just '.zip'
# (which is all ``pathlib.Path.suffix`` would return).
_MODEL_FILE_EXTS = (
    '.nequip.zip',
    '.nequip.pt2',
    '.nequip.pth',
    '.ckpt',
    '.pth',
    '.model',
)


def model_file_ext(filename: str, default: str = '.model') -> str:
    """Return a model file's extension, honouring compound nequip suffixes.

    ``Path('m.nequip.zip').suffix`` is only ``'.zip'``; this returns the full
    ``'.nequip.zip'`` so calcjobs name copied model files consistently with
    what the remote scripts (which use ``backend.model_file_extension``) expect.
    """
    from pathlib import Path

    name = str(filename)
    for ext in _MODEL_FILE_EXTS:
        if name.endswith(ext):
            return ext
    return Path(name).suffix or default


def model_file_stem(filename: str) -> str:
    """Return a model filename with its (possibly compound) extension removed."""
    from pathlib import Path

    name = Path(str(filename)).name
    ext = model_file_ext(name, default='')
    if ext and name.endswith(ext):
        return name[: -len(ext)]
    return Path(name).stem


def find_inference_model(directory, stem: str, backend):
    """Return the model file to hand to ``backend.create_calculator``.

    Prefers a pre-compiled inference artifact (shipped by the optional
    compile-once step, e.g. ``{stem}.nequip.pt2``) when present in ``directory``,
    otherwise falls back to the raw model ``{stem}{backend.model_file_extension}``.

    Backends without compilation (no ``MLIPModelCompiler``) always resolve to the
    raw model, so behaviour is unchanged for them.
    """
    from pathlib import Path

    directory = Path(directory)
    if isinstance(backend, MLIPModelCompiler):
        compiled = directory / f'{stem}{backend.compiled_model_extension}'
        if compiled.exists():
            return compiled
    return directory / f'{stem}{backend.model_file_extension}'


def list_inference_models(directory, backend) -> list:
    """Return all model files in ``directory`` to use for inference.

    Prefers pre-compiled artifacts (``*.nequip.pt2``) when any are present,
    otherwise the raw models (``*{backend.model_file_extension}``). Used by the
    committee evaluation, which loads every committee member. Falls back to the
    raw extension for backends without compilation.
    """
    from pathlib import Path

    directory = Path(directory)
    if isinstance(backend, MLIPModelCompiler):
        compiled = sorted(directory.glob(f'*{backend.compiled_model_extension}'))
        if compiled:
            return compiled
    return sorted(directory.glob(f'*{backend.model_file_extension}'))


# Import backends so they register themselves. Each backend is optional: their
# ML stacks can be mutually exclusive in a single environment (e.g. mace-torch
# pins e3nn==0.4.4 while new nequip needs e3nn>=0.6), so a deployment may install
# only the backend it runs. Import defensively — a backend whose dependencies
# are missing simply isn't registered, without breaking the others. Only
# ImportError is swallowed so genuine bugs in a backend module still surface.
import contextlib as _contextlib  # noqa: E402
import importlib as _importlib  # noqa: E402

for _backend_module in (
    'atlas.active_learning.backends.allegro',
    'atlas.active_learning.backends.mace',
):
    with _contextlib.suppress(ImportError):
        _importlib.import_module(_backend_module)

__all__ = [
    'MLIPCalculatorFactory',
    'MLIPCommitteeEvaluator',
    'MLIPDescriptorProvider',
    'MLIPModelCompiler',
    'MLIPTrainer',
    'find_inference_model',
    'get_backend',
    'list_backends',
    'list_inference_models',
    'model_file_ext',
    'model_file_stem',
    'register_backend',
]
