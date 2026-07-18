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
    MLIPPretrainedModel,
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


def trainable_backends() -> list[str]:
    """Return the registered backends that can train models.

    Inference-only backends (pretrained foundation potentials) omit
    ``MLIPTrainer``; this is used to give an actionable error when one of them
    is configured as ``mlip.training_backend``.
    """
    return sorted(
        name
        for name in _BACKEND_REGISTRY
        if isinstance(_BACKEND_REGISTRY[name](), MLIPTrainer)
    )


def parse_model_spec(spec) -> tuple[str, str | None] | None:
    r"""Split a ``'<backend>:<variant>'`` model spec.

    Parameters
    ----------
    spec : Any
        Candidate spec, e.g. ``'mace'``, ``'mace:mp-small'``, ``'orb:orb-v2'``,
        or a filesystem path.

    Returns
    -------
    tuple[str, str | None] | None
        ``(backend_name, variant)`` when the part before the first ``':'`` is a
        registered backend, else None. Requiring a registered prefix (rather
        than merely containing a ``':'``) is what keeps filesystem paths, and
        Windows-style ``C:\\...`` paths, from being misread as specs.
    """
    if not isinstance(spec, str):
        return None
    name, sep, variant = spec.partition(':')
    if name not in _BACKEND_REGISTRY:
        return None
    if not sep:
        return (name, None)
    return (name, variant or None)


def resolve_model_spec(spec, default_backend: str | None = None):
    """Resolve a model spec into a ``(backend, pretrained_id)`` pair.

    Parameters
    ----------
    spec : Any
        ``'<backend>:<variant>'`` for a pretrained model, ``'<backend>'`` for a
        backend (its default pretrained model when it has one, else its trained
        model on disk), or None to fall back to ``default_backend``.
    default_backend : str | None
        Backend to use when ``spec`` is None or unparseable.

    Returns
    -------
    tuple[str, str | None]
        ``(backend_name, pretrained_id)``. ``pretrained_id`` is None when the
        model should be resolved from disk.

    Raises
    ------
    ValueError
        If a variant is given for a backend that serves no pretrained models,
        or the backend cannot resolve the variant.
    """
    parsed = parse_model_spec(spec)
    if parsed is None:
        return ((default_backend or spec or 'mace'), None)

    backend_name, variant = parsed
    backend = get_backend(backend_name)

    if not isinstance(backend, MLIPPretrainedModel):
        if variant is not None:
            raise ValueError(
                f"Backend '{backend_name}' does not provide pretrained models, "
                f"so '{spec}' is not valid. Use '{backend_name}' on its own to "
                'use a model trained by the active learning loop.'
            )
        return (backend_name, None)

    # A bare pretrained-only backend name falls back to its default model; a
    # trainable backend (e.g. mace) keeps resolving from disk unless a variant
    # is given, so existing configs are unaffected.
    if variant is None:
        if isinstance(backend, MLIPTrainer):
            return (backend_name, None)
        return (backend_name, backend.resolve_pretrained_model(None))

    return (backend_name, backend.resolve_pretrained_model(variant))


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


def find_inference_model(directory, stem: str, backend, pretrained_model=None):
    """Return the model reference to hand to ``backend.create_calculator``.

    Prefers a pre-compiled inference artifact (shipped by the optional
    compile-once step, e.g. ``{stem}.nequip.pt2``) when present in ``directory``,
    otherwise falls back to the raw model ``{stem}{backend.model_file_extension}``.

    Backends without compilation (no ``MLIPModelCompiler``) always resolve to the
    raw model, so behaviour is unchanged for them.

    Parameters
    ----------
    pretrained_model : str | None
        A published identifier (see ``MLIPPretrainedModel``). When given it wins
        outright, since it is an explicit user choice.

    Returns
    -------
    Path | str
        A ``str`` identifier for a pretrained model, else a ``Path``.
    """
    from pathlib import Path

    # Explicit configuration wins over anything on disk.
    if pretrained_model is not None:
        return pretrained_model

    directory = Path(directory)
    if isinstance(backend, MLIPModelCompiler):
        compiled = directory / f'{stem}{backend.compiled_model_extension}'
        if compiled.exists():
            return compiled

    # NB: deliberately no fallback to the backend's default pretrained model
    # when this path is missing. A missing trained model is a real failure and
    # must surface as one -- silently substituting a foundation model would run
    # the MD on the wrong potential and yield plausible-looking but wrong
    # results. Choosing a pretrained model is `resolve_model_spec`'s job, and it
    # only does so when the user asked for one.
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
# only the backend it runs. Import defensively a backend whose dependencies
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
    'MLIPPretrainedModel',
    'MLIPTrainer',
    'find_inference_model',
    'get_backend',
    'list_backends',
    'list_inference_models',
    'model_file_ext',
    'model_file_stem',
    'parse_model_spec',
    'register_backend',
    'resolve_model_spec',
    'trainable_backends',
]
