#!/usr/bin/env python3
"""Backend-agnostic MLIP training script.

Deployed as PortableCode to HPC nodes. Reads which backend to use
from settings.toml, then dispatches training via the backend registry.
Writes standardized training_results.json for the parser.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _patch_torch_load_if_needed():
    """Fix ``torch.load`` for e3nn < 0.5.2 + PyTorch >= 2.6.

    Sets ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` so that all processes
    (including forkserver multiprocessing workers spawned by nequip)
    default to ``weights_only=False``.
    """
    import os

    try:
        import torch
        from packaging.version import Version

        torch_version = Version(torch.__version__.split('+')[0])
        if torch_version < Version('2.6'):
            return

        import e3nn

        e3nn_version = Version(e3nn.__version__)
        if e3nn_version >= Version('0.5.2'):
            return

        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

        warnings.warn(
            f'Detected e3nn {e3nn.__version__} with PyTorch '
            f'{torch.__version__}. Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1. '
            f'Consider upgrading e3nn >= 0.5.2.',
            stacklevel=2,
        )
    except ImportError:
        pass


def main():
    """Run MLIP training and write standardized results."""
    _patch_torch_load_if_needed()

    from atlas.active_learning.backends import (
        get_backend,
        trainable_backends,
    )
    from atlas.active_learning.backends._base import MLIPTrainer

    with open('settings.toml', 'rb') as f:
        settings = tomllib.load(f)

    backend_name = settings.get('mlip', {}).get('training_backend', 'mace')
    backend = get_backend(backend_name)

    if not isinstance(backend, MLIPTrainer):
        raise TypeError(
            f"MLIP backend '{backend_name}' provides pretrained models only and "
            f'cannot train. Set mlip.training_backend to a trainable backend '
            f'({", ".join(trainable_backends())}), or use this backend for MD '
            f'only via [md.parameters].md_type = "{backend_name}:<variant>".'
        )

    backend.run_training(config_path='train_config.yaml')

    results = backend.parse_training_results(results_dir=Path('.'))
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    sys.exit(main())
