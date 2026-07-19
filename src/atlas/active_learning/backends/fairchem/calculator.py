"""fairchem ASE calculator creation.

fairchem (fairchem-core) serves Meta's pretrained potentials -- UMA, eSEN,
AllSCAIP. As with Orb there is no ATLAS training path: the "model" is a
published identifier such as ``'uma-s-1p2'`` (see ``FairchemBackend``), resolved
and downloaded by fairchem, or a local checkpoint.

API verified against fairchem-core 2.21.0:
- ``pretrained_mlip.get_predict_unit(name, device=...)`` for a published model,
- ``pretrained_mlip.load_predict_unit(path, device=...)`` for a local checkpoint,
- ``FAIRChemCalculator(predict_unit, task_name=...)``.

UMA is multi-task, so it needs a ``task_name`` (``'omat'`` for bulk materials);
eSEN/AllSCAIP are single-task and select their one dataset automatically.
"""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator


def create_fairchem_calculator(
    model_path: str | Path | None = None,
    device: str = 'cpu',
    dtype: str = 'float32',
    **kwargs,
) -> Calculator:
    """Create an ASE calculator from a fairchem model or checkpoint.

    Parameters
    ----------
    model_path : str | Path | None
        A published model name (e.g. ``'uma-s-1p2'``), a path to a local
        checkpoint, or None for the default model.
    device : str
        ``'cpu'`` or ``'cuda'`` (``'cuda:N'`` is accepted and mapped to
        ``'cuda'``).
    dtype : str
        Accepted for interface compatibility; fairchem exposes no dtype knob, so
        it is ignored.
    **kwargs
        ``task_name`` selects the UMA task (default: ``'omat'`` when the model
        offers it, else the model's single task). Other keys (e.g. the MACE-only
        ``enable_cueq``) are accepted and ignored so the shared MD call site does
        not need to special-case this backend.

    Returns
    -------
    Calculator
        A ``FAIRChemCalculator``.
    """
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:
        raise ImportError(
            "The 'fairchem' MLIP backend requires the 'fairchem-core' package, "
            "which is not installed. Install it to use model_type='fairchem'. "
            'Because fairchem pulls torch/e3nn/numpy pins that conflict with the '
            'MACE/ATLAS environment, it is best run from its own container '
            f'image. Original error: {exc}'
        ) from exc

    fc_device = 'cuda' if str(device).startswith('cuda') else 'cpu'

    if model_path is not None and Path(str(model_path)).exists():
        predict_unit = pretrained_mlip.load_predict_unit(
            str(model_path), device=fc_device
        )
    else:
        model_name = str(model_path) if model_path else 'uma-s-1p2'
        predict_unit = _load_pretrained(pretrained_mlip, model_name, fc_device)

    task_name = _select_task(predict_unit, kwargs.get('task_name'))
    return FAIRChemCalculator(predict_unit, task_name=task_name)


def _load_pretrained(pretrained_mlip, model_name: str, device: str):
    """Resolve a published model, turning failures into actionable errors."""
    # Authenticate gated downloads with the ATLAS-managed HF token, if any.
    from atlas.core.code_utils import apply_hf_token

    apply_hf_token()
    try:
        return pretrained_mlip.get_predict_unit(model_name, device=device)
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
        available = ', '.join(pretrained_mlip.available_models)
        # UMA checkpoints are gated on Hugging Face; the raw error is an opaque
        # 401/403 deep in huggingface_hub.
        if _looks_like_hf_auth_error(exc):
            raise RuntimeError(
                f"Cannot download the gated fairchem model '{model_name}'. This "
                'needs a Hugging Face token (huggingface-cli login or the '
                'HF_TOKEN environment variable) AND access to the '
                'facebook/UMA repo. Access is granted by manual review, so after '
                'requesting it at https://huggingface.co/facebook/UMA the 403 '
                f'persists until the authors approve. Original error: {exc}'
            ) from exc
        raise ValueError(
            f"Could not load fairchem model '{model_name}'. Available models: "
            f'{available}. Original error: {exc}'
        ) from exc


def _select_task(predict_unit, task_name):
    """Choose the FAIRChemCalculator task for this model.

    Precedence: an explicit ``task_name`` wins; otherwise ``'omat'`` (bulk
    materials) when the model offers it; otherwise None, which makes
    FAIRChemCalculator auto-select a single-task model's only dataset and raise
    an informative error for a multi-task model.
    """
    if task_name is not None:
        return task_name
    valid = list(getattr(predict_unit, 'dataset_to_tasks', {}) or {})
    if 'omat' in valid:
        return 'omat'
    return None


def _looks_like_hf_auth_error(exc: Exception) -> bool:
    """Best-effort detection of a Hugging Face gated/auth error."""
    text = f'{type(exc).__name__}: {exc}'.lower()
    return any(
        token in text
        for token in ('401', '403', 'gated', 'unauthorized', 'authentication')
    )
