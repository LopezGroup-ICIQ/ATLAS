"""Orb per-structure uncertainty from the v3 confidence head.

Orb v3 models (``orb-v3-*``) carry a ``ConfidenceHead`` that estimates the mean
force-prediction error per atom, exposed as ``ORBCalculator.results['confidence']``
-- a per-atom probability distribution over 50 force-error bins spanning
0-0.3 eV/A. The v2 models have no such head.

Reducing that distribution to an expected force error (sum of bin_center * prob)
gives a physical, per-atom uncertainty; the max over atoms is the per-structure
score. Verified monotonic with disorder (a rattled crystal scores strictly higher
than the perfect lattice).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


def estimate_uncertainty_orb(
    structures: list[Atoms],
    model: str | None = None,
    device: str = 'cpu',
    reduction: str = 'max',
) -> list[float]:
    """Return the per-structure expected force error (eV/A) from Orb's head.

    Parameters
    ----------
    structures : list[Atoms]
        Structures to score.
    model : str | None
        An ``orb-v3-*`` model name (or checkpoint path); None uses the default.
        A v2 model has no confidence head and raises a clear error.
    device : str
        Inference device.
    reduction : str
        How to reduce per-atom expected error to a per-structure scalar:
        ``'max'`` (default, most sensitive) or ``'mean'``.

    Returns
    -------
    list[float]
        One non-negative uncertainty per structure, in input order.
    """
    from atlas.active_learning.backends.orb.calculator import (
        build_orb_calculator_and_model,
    )

    calculator, orb_model = build_orb_calculator_and_model(
        model, device=device, compile_model=False
    )

    heads = getattr(orb_model, 'heads', {})
    if 'confidence' not in heads:
        raise ValueError(
            f"The Orb model '{model or 'default'}' has no confidence head, so it "
            'cannot provide committee-free uncertainty. Use an orb-v3 model '
            "(e.g. 'orb-v3-conservative-inf-omat')."
        )

    # Bin centres of the confidence head's force-error histogram (eV/A).
    edges = heads['confidence'].bin_edges.detach().cpu().numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])

    reduce_fn = np.max if reduction == 'max' else np.mean

    scores: list[float] = []
    for struct in structures:
        atoms = struct.copy()
        atoms.calc = calculator
        atoms.get_potential_energy()
        # (n_atoms, n_bins) probabilities -> per-atom expected force error.
        probs = np.asarray(calculator.results['confidence'])
        per_atom_error = probs @ centers
        scores.append(float(reduce_fn(per_atom_error)))

    return scores
