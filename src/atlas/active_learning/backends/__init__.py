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


# Import backends so they register themselves
import atlas.active_learning.backends.mace  # noqa: E402, F401

__all__ = [
    'MLIPCalculatorFactory',
    'MLIPCommitteeEvaluator',
    'MLIPDescriptorProvider',
    'MLIPTrainer',
    'get_backend',
    'list_backends',
    'register_backend',
]
