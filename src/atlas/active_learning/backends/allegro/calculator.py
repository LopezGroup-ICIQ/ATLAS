"""Allegro/NequIP ASE calculator creation."""

from __future__ import annotations

import contextlib
from pathlib import Path

from ase.calculators.calculator import Calculator


def create_allegro_calculator(
    model_path: str | Path,
    device: str = 'cpu',
    dtype: str = 'float32',
    **kwargs,
) -> Calculator:
    """Create an ASE Calculator from an Allegro/NequIP model.

    Targets the new nequip API, which loads inference models via
    ``NequIPCalculator.from_compiled_model``. Accepts either an already
    compiled artifact (``*.nequip.pt2``) or a raw Lightning checkpoint
    (``*.ckpt``); a checkpoint is compiled on demand with
    :func:`atlas.active_learning.backends.allegro.training.deploy_allegro_model`.

    Parameters
    ----------
    model_path : str | Path
        Path to a compiled model file (``*.nequip.pt2``) or a training
        checkpoint (``*.ckpt``) to compile on demand.
    device : str
        Device for inference (``'cpu'`` or ``'cuda'``).
    dtype : str
        Data type for inference (kept for interface compatibility).

    Returns
    -------
    Calculator
        An ASE-compatible NequIP calculator.
    """
    # Ensure ``torch.load`` tolerates e3nn/nequip checkpoints on
    # PyTorch >= 2.6 (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD when needed)
    # before importing nequip, which transitively imports e3nn.o3.
    from atlas.active_learning.backends.allegro import (
        _patch_torch_load_if_needed,
    )

    _patch_torch_load_if_needed()

    from nequip.ase import NequIPCalculator

    model_path = Path(model_path)

    species_map = kwargs.pop('chemical_species_to_atom_type_map', None)
    # ``aotinductor`` shells out to a C++ compiler; ``torchscript`` compiled
    # in-process but is removed in PyTorch >= 2.10, so aotinductor is the only
    # option on recent torch. Kept configurable via ``compile_mode`` for older
    # torch. Either way a working C++ compiler must be present for aotinductor.
    compile_mode = kwargs.pop('compile_mode', 'aotinductor')

    calc_kwargs = {}
    if species_map is not None:
        calc_kwargs['chemical_species_to_atom_type_map'] = species_map

    from atlas.active_learning.backends.allegro.training import (
        deploy_allegro_model,
    )

    def _compile(checkpoint):
        compiled = deploy_allegro_model(
            checkpoint_path=checkpoint,
            device=device,
            mode=compile_mode,
        )
        if compiled is None:
            raise RuntimeError(
                f"Failed to compile Allegro checkpoint '{checkpoint}' for "
                f'inference via nequip-compile (device={device!r}, '
                f'mode={compile_mode!r}). Check that nequip-compile is '
                'available and the checkpoint is valid.'
            )
        return compiled

    # ``NequIPCalculator.from_compiled_model`` consumes a *compiled* model
    # (``*.nequip.pt2`` for aotinductor, ``*.nequip.pth`` for torchscript),
    # not the raw Lightning checkpoint produced by ``nequip-train``. When
    # handed a checkpoint, compile it on demand for the requested device.
    was_precompiled = model_path.name.endswith(('.nequip.pt2', '.nequip.pth'))
    if not was_precompiled:
        model_path = _compile(model_path)

    try:
        return NequIPCalculator.from_compiled_model(
            compile_path=str(model_path),
            device=device,
            **calc_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        # A pre-compiled artifact may have been produced on a different node
        # (compilation is device- and arch-specific), so it can fail to load
        # here. Recover by recompiling on *this* node from a co-located raw
        # checkpoint when one is available; otherwise surface a clear error.
        if not was_precompiled:
            # We just compiled this on this node.
            raise
        raw_checkpoint = _find_sibling_checkpoint(model_path)
        if raw_checkpoint is None:
            raise RuntimeError(
                f"Failed to load pre-compiled Allegro artifact '{model_path}' "
                'on this node, and no co-located raw checkpoint was found to '
                'recompile from. The artifact may have been compiled for a '
                'different device/architecture.'
            ) from exc
        # Drop the stale artifact so ``deploy_allegro_model`` recompiles fresh.
        with contextlib.suppress(OSError):
            model_path.unlink()
        recompiled = _compile(raw_checkpoint)
        return NequIPCalculator.from_compiled_model(
            compile_path=str(recompiled),
            device=device,
            **calc_kwargs,
        )


def _find_sibling_checkpoint(compiled_path: Path) -> Path | None:
    """Return a raw checkpoint next to a compiled artifact, or None.

    Used to recompile locally when a pre-compiled ``*.nequip.pt2`` (possibly
    built on another node) fails to load. Looks for a same-stem raw model in the
    same directory.
    """
    from atlas.active_learning.backends import model_file_stem

    stem = model_file_stem(compiled_path.name)
    for ext in ('.nequip.zip', '.ckpt'):
        candidate = compiled_path.parent / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None
