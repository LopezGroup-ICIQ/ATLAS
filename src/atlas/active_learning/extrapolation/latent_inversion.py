"""Decode latent-space coordinates back into atomic structures.

The descriptor of a differentiable backend is a differentiable function of the
atomic positions, and the ATLAS autoencoder maps descriptors to a 2D latent
space. Composing the two gives a differentiable map from positions to latent
coordinates, which this module inverts by gradient-based optimization:

    L(R) = w_lat  * || mean_i enc(s(D_i(R))) - z* ||^2
         + w_rep  * sum_{i<j} barrier(d_ij ; f * (r_i + r_j))
         + w_coh  * sum_i max(0, min_coord - n_neighbours(i))^2
         + w_disp * mean_i max(0, |R_i - R_i^0| - d_free)^2

where ``s()`` is the standardization used to train the autoencoder, ``enc`` is
the (frozen) encoder, and ``r_i`` is the covalent radius of atom i. The
mean-over-atoms form reproduces exactly how training structures are placed in
the latent space, so target and data are on the same footing.

Inversion has to be constrained to be meaningful. The target is
two-dimensional while the structure has 3N degrees of freedom, so the problem
is heavily underdetermined and an unconstrained optimizer exploits that
freedom in two ways. It drives atoms into each other, since overlapping atoms
have distinctive descriptors; and it dissociates the cluster, since an
isolated atom's descriptor can drag the structure mean toward targets no
compact structure can reach. The repulsion and cohesion terms bracket the
physically sensible geometries from below and above. Both are one-sided,
contributing nothing wherever the structure is already acceptable, so neither
biases the search in well-behaved regions. The displacement restraint is a
search regularizer rather than a physical constraint and can be switched off
(``w_disp = 0``) for exploratory decoding.

This module is GUI-free and imports torch lazily, so it can be imported by
capability checks that must not pay for a torch import.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

    from atlas.active_learning.backends._base import (
        DifferentiableDescriptorHandle,
    )

#: Fraction of the summed covalent radii below which a contact is rejected.
#: ATLAS uses 0.7 as its DFT outlier filter (see
#: ``active_learning_utils``: ``(R_atA + R_atB) * 0.7``), which is the loosest
#: defensible value. The default here is tighter: at 0.7 the optimizer parks
#: atoms flush against the wall, giving Cu-Cu contacts near 1.85 A, and
#: structures that compressed are poor DFT candidates.
COVALENT_FRACTION = 0.85

#: Multiple of the summed covalent radii out to which two atoms count as
#: bonded, used for fragment detection and the cohesion term.
BOND_FACTOR = 1.25


# ---------------------------------------------------------------------------
# Geometry helpers (numpy; used for acceptance tests, not for gradients)
# ---------------------------------------------------------------------------


def covalent_min_matrix(atoms: Atoms, f: float = COVALENT_FRACTION) -> np.ndarray:
    """Minimum acceptable pair distances as an ``(N, N)`` array.

    Entry ``(i, j)`` is ``f * (r_i + r_j)`` with ``r`` the covalent radius. The
    diagonal is zero so that self-pairs never trigger a penalty.
    """
    from ase.data import covalent_radii

    r = covalent_radii[atoms.get_atomic_numbers()]
    rmin = f * (r[:, np.newaxis] + r[np.newaxis, :])
    np.fill_diagonal(rmin, 0.0)
    return rmin


def min_distance_ratio(atoms: Atoms) -> tuple[float, float, tuple[str, str] | None]:
    """Smallest ``d_ij / (r_i + r_j)`` over all pairs.

    Uses raw (non-PBC) pairwise distances, since clusters live in large boxes
    where minimum-image distances can be artificially small for atoms near
    opposite cell faces.

    Returns ``(ratio, distance, (symbol_i, symbol_j))``; ``(inf, inf, None)``
    for structures with fewer than two atoms.
    """
    from ase.data import covalent_radii
    from scipy.spatial.distance import pdist, squareform

    if len(atoms) < 2:
        return float('inf'), float('inf'), None

    d = squareform(pdist(atoms.positions))
    r = covalent_radii[atoms.get_atomic_numbers()]
    rsum = r[:, np.newaxis] + r[np.newaxis, :]
    ratio = d / rsum
    np.fill_diagonal(ratio, np.inf)

    i, j = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
    symbols = atoms.get_chemical_symbols()
    return float(ratio[i, j]), float(d[i, j]), (symbols[i], symbols[j])


def connectivity(atoms: Atoms, bond_factor: float = BOND_FACTOR) -> tuple[int, int]:
    """Fragment and isolated-atom counts for a cluster.

    Two atoms are bonded when ``d_ij <= bond_factor * (r_i + r_j)``. Returns
    ``(n_fragments, n_isolated)``; a compact cluster gives ``(1, 0)``.
    """
    from ase.data import covalent_radii
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import pdist, squareform

    if len(atoms) < 2:
        return 1, 0

    d = squareform(pdist(atoms.positions))
    np.fill_diagonal(d, np.inf)
    r = covalent_radii[atoms.get_atomic_numbers()]
    rmax = bond_factor * (r[:, np.newaxis] + r[np.newaxis, :])

    bonded = d <= rmax
    n_frag, _ = connected_components(csr_matrix(bonded), directed=False)
    return int(n_frag), int((~bonded.any(axis=1)).sum())


def check_physical(
    atoms: Atoms,
    f: float = COVALENT_FRACTION,
    bond_factor: float = BOND_FACTOR,
    max_fragments: int = 1,
):
    """Check a structure against the bond floor and the connectivity ceiling.

    A structure is accepted when no pair is closer than ``f * (r_i + r_j)`` and
    the cluster does not break into more than *max_fragments* pieces. The
    second test matters as much as the first: a dissociated cloud of atoms is
    just as unphysical as an overlapping one, and it is the failure mode that
    unconstrained latent matching produces.

    Returns ``(ok, ratio, distance, pair, n_fragments, n_isolated)``.
    """
    ratio, dist, pair = min_distance_ratio(atoms)
    n_frag, n_iso = connectivity(atoms, bond_factor)
    return (ratio >= f and n_frag <= max_fragments), ratio, dist, pair, n_frag, n_iso


# ---------------------------------------------------------------------------
# Differentiable penalty terms (torch)
# ---------------------------------------------------------------------------


def pair_distances(pos):
    """``(N, N)`` pair distances, with the diagonal set to 1 rather than 0.

    The gradient of ``||x||`` is undefined at ``x = 0``, which the self-pairs
    would otherwise hit, so the diagonal is offset before the square root.
    Callers must mask the diagonal through their own thresholds.
    """
    import torch

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    d2 = (diff**2).sum(-1)
    eye = torch.eye(pos.shape[0], device=pos.device, dtype=pos.dtype)
    return torch.sqrt(d2 + eye)


def repulsion_penalty(pos, rmin_t, form='inverse', power=6, max_ratio=10.0):
    """One-sided penalty on pair distances below *rmin_t*.

    Both forms are exactly zero, with zero gradient, at and above the floor, so
    the term is inactive wherever the geometry is already acceptable.

    ``'quadratic'``
        ``max(0, rmin - d)^2``. Soft: the latent term can outbid it, and the
        optimum parks flush against the wall.
    ``'inverse'``
        ``((rmin/d)^power - 1)^2`` below the floor. Matches the floor value and
        slope of the quadratic form at ``d = rmin`` but diverges as ``d -> 0``,
        so pushing further into the core costs unboundedly more. In practice
        this stops the optimizer trading overlap for latent accuracy.

    *max_ratio* clamps ``rmin/d`` so the barrier stays finite in float64 even
    if a line search briefly proposes near-coincident atoms.
    """
    import torch

    d = pair_distances(pos)

    if form == 'quadratic':
        viol = torch.clamp(rmin_t - d, min=0.0)
        # Each unique pair is counted twice in the full matrix; the diagonal is
        # zero because rmin_t has a zero diagonal.
        return 0.5 * (viol**2).sum()

    if form != 'inverse':
        raise ValueError(f'unknown repulsion form: {form}')

    # Clamp the ratio rather than the distance: rmin varies per pair, and the
    # diagonal (rmin = 0) must stay at ratio 0 so it contributes nothing.
    ratio = torch.clamp(rmin_t / d, max=max_ratio)
    excess = torch.clamp(ratio**power - 1.0, min=0.0)
    return 0.5 * (excess**2).sum()


def cohesion_penalty(pos, rmax_t, min_coord=2.0, width=0.25, beta=6.0):
    """Penalty on atoms that are losing contact -- the mirror of repulsion.

    Matching only the structure-mean latent coordinate leaves a degenerate
    escape route: an isolated atom has a very distinctive descriptor, so
    dissociating the cluster is a cheap way to drag the mean toward a target
    that no compact structure can reach. Repulsion cannot object, since the
    atoms are far apart, and a displacement restraint cannot tell an atom that
    moved to a new site from one that flew into vacuum.

    Two contributions, both zero for a well-connected cluster:

    coordination
        Neighbours are counted smoothly, as a sum of sigmoids switching on
        inside *rmax_t*, and atoms below *min_coord* are penalised. This is the
        sensitive term while an atom is still near the cutoff, but it saturates
        once the atom is well clear.
    separation
        A quadratic hinge on each atom's soft-minimum neighbour distance,
        expressed as a multiple of that pair's bond cutoff. Unlike the
        coordination term it never saturates, so an atom that has already
        broken away still feels a restoring gradient.
    """
    import torch

    d = pair_distances(pos)

    # The diagonal sits at d = 1 with rmax > 1, so it would count as a
    # self-neighbour; subtract its contribution explicitly.
    counts = torch.sigmoid((rmax_t - d) / width)
    n_neigh = counts.sum(dim=1) - torch.diagonal(counts)
    deficit = torch.clamp(min_coord - n_neigh, min=0.0)

    # Distances in units of each pair's bond cutoff, so heteroatomic pairs
    # share one scale. Self-pairs are pushed out of the soft minimum.
    eye = torch.eye(pos.shape[0], device=pos.device, dtype=pos.dtype)
    u = d / torch.clamp(rmax_t, min=1e-6) + eye * 1e6
    u_min = -torch.logsumexp(-beta * u, dim=1) / beta
    separation = torch.clamp(u_min - 1.0, min=0.0)

    return (deficit**2).sum() + (separation**2).sum()


def displacement_restraint(pos, pos0, d_free):
    """Flat-bottom restraint on per-atom displacement from *pos0*.

    Atoms move freely within *d_free* of their starting position; beyond that
    the penalty grows quadratically. Returns the mean over atoms so the weight
    does not have to be rescaled with system size.
    """
    import torch

    disp = torch.sqrt(((pos - pos0) ** 2).sum(-1) + 1e-12)
    excess = torch.clamp(disp - d_free, min=0.0)
    return (excess**2).mean()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class LatentInversionSettings:
    """Every knob of a latent traversal, in the style of ``AutoencoderSettings``.

    Persisted per project as JSON, so :meth:`from_dict` deliberately ignores
    unknown keys: a settings file written by a newer version must not stop an
    older one from opening the project.
    """

    # -- Path ---------------------------------------------------------------
    #: Structures decoded along the polyline through the marked waypoints.
    #: One by default, so a first run decodes exactly the marked points.
    n_structures: int = 1
    #: Place a sample exactly on each waypoint rather than on a uniform grid.
    snap_waypoints: bool = True
    #: Offer the previous point's result as a seed for the next one. Faster,
    #: and it makes the decoded path visibly smoother.
    chain_from_previous: bool = True

    # -- Seeding ------------------------------------------------------------
    #: 'nearest' | 'by_size' | 'fixed_size'
    seed_strategy: str = 'nearest'
    #: Target atom count for the size-aware strategies.
    target_n_atoms: int | None = None
    #: Permitted deviation from ``target_n_atoms`` ('fixed_size' forces 0).
    size_tolerance: int = 10
    #: Restarts per point; the objective is non-convex enough that a single
    #: start regularly lands in a poor minimum.
    n_starts: int = 3
    #: Exclude already-fragmented seeds. The optimizer moves atoms by a couple
    #: of Angstrom at most and cannot reassemble a gas.
    require_compact: bool = True

    # -- Optimizer ----------------------------------------------------------
    n_rounds: int = 8
    max_iter: int = 100
    lr: float = 0.05
    tolerance: float = 1e-4
    optimizer: str = 'lbfgs'
    device: str = 'cpu'
    #: float64 by default: strong-Wolfe on a stiff inverse barrier is far
    #: better conditioned in double precision.
    dtype: str = 'float64'

    # -- Objective ----------------------------------------------------------
    w_lat: float = 1.0
    w_rep: float = 10.0
    #: Dissociation is a strong attractor when the target is unreachable, so
    #: cohesion needs an order of magnitude more weight than repulsion: at 10
    #: the far end of a size traversal still came apart into 17 fragments; at
    #: 100 it stays compact for a latent-error cost of about 0.1.
    w_coh: float = 100.0
    #: Search regularizer. Set to 0 for exploratory decoding, which keeps the
    #: repulsion and cohesion terms but drops any preference for the seed.
    w_disp: float = 1.0
    rep_form: str = 'inverse'
    #: Multipliers applied to ``w_rep``/``w_coh`` over the first rounds.
    #: Ramping avoids stalling the strong-Wolfe line search on a stiff barrier
    #: from the first iteration. The last value is held for later rounds.
    rep_ramp: tuple[float, ...] = (0.1, 1.0, 10.0)

    # -- Geometry -----------------------------------------------------------
    covalent_fraction: float = COVALENT_FRACTION
    bond_factor: float = BOND_FACTOR
    #: Relative safety margin applied to ``covalent_fraction`` inside the
    #: penalty only. The hinge has zero gradient at the boundary, so the
    #: optimizer converges exactly onto the floor and rounding would decide
    #: the acceptance test. Optimizing against ``f * (1 + margin)`` leaves the
    #: converged geometry comfortably inside the acceptable region.
    margin: float = 0.02
    min_coord: float = 2.0
    d_free: float = 1.5
    max_fragments: int = 1

    # -- Cell ---------------------------------------------------------------
    relax_cell: bool = False
    max_strain: float = 0.10
    w_strain: float = 10.0

    # -- Bookkeeping --------------------------------------------------------
    #: Radius used to report how well the data covers a target point.
    neighbourhood_radius: float = 0.5
    #: Latent error at or below which a physical candidate ends the restarts.
    good_enough_latent_err: float = 1e-2

    def __post_init__(self):
        # JSON round-trips tuples as lists; normalise so callers can rely on
        # rep_ramp[-1] and on equality between a saved and a fresh instance.
        self.rep_ramp = tuple(float(x) for x in self.rep_ramp)

    def validate(self) -> None:
        """Raise ``ValueError`` on a combination that cannot run.

        Called before a traversal starts so that a bad dialog entry surfaces
        immediately rather than as a torch error eight rounds in.
        """
        if self.n_structures < 1:
            raise ValueError('n_structures must be at least 1')
        if self.n_starts < 1:
            raise ValueError('n_starts must be at least 1')
        if self.n_rounds < 1:
            raise ValueError('n_rounds must be at least 1')
        if self.max_iter < 1:
            raise ValueError('max_iter must be at least 1')
        if not self.rep_ramp:
            raise ValueError('rep_ramp must have at least one entry')
        if any(x <= 0 for x in self.rep_ramp):
            raise ValueError('rep_ramp entries must be positive')
        if self.rep_form not in ('inverse', 'quadratic'):
            raise ValueError(f'unknown repulsion form: {self.rep_form}')
        if self.optimizer != 'lbfgs':
            raise ValueError(f'unknown optimizer: {self.optimizer}')
        if self.seed_strategy not in ('nearest', 'by_size', 'fixed_size'):
            raise ValueError(f'unknown seed strategy: {self.seed_strategy}')
        if self.seed_strategy != 'nearest' and self.target_n_atoms is None:
            raise ValueError(
                f"seed strategy '{self.seed_strategy}' needs target_n_atoms"
            )
        if not 0.0 < self.covalent_fraction <= 1.0:
            raise ValueError('covalent_fraction must be in (0, 1]')
        if self.bond_factor < 1.0:
            raise ValueError('bond_factor must be at least 1.0')
        if self.d_free < 0:
            raise ValueError('d_free must be non-negative')
        if self.lr <= 0:
            raise ValueError('lr must be positive')

    @property
    def effective_size_tolerance(self) -> int:
        """Size window actually used, with 'fixed_size' pinned to zero."""
        if self.seed_strategy == 'fixed_size':
            return 0
        return int(self.size_tolerance)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['rep_ramp'] = list(self.rep_ramp)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> LatentInversionSettings:
        """Build from a (possibly stale or partial) mapping.

        Unknown keys are dropped rather than raising, so that settings saved by
        a different version still load.
        """
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


class InversionCancelled(RuntimeError):
    """Raised when ``should_stop`` asks an inversion to abort."""


def invert_to_latent(
    handle: DifferentiableDescriptorHandle,
    z_target,
    initial_atoms: Atoms,
    ae_model,
    mean_vals,
    std_vals,
    settings: LatentInversionSettings | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_round: Callable[[int, dict], None] | None = None,
) -> dict:
    """Optimize positions so the structure-mean latent coordinate matches z*.

    Parameters
    ----------
    handle : DifferentiableDescriptorHandle
        A gradient-capable descriptor handle (see
        ``backends.supports_descriptor_gradients``). Species are fixed; only
        positions are optimized.
    z_target : array-like
        Target latent coordinate, shape ``(bottleneck_dim,)``.
    initial_atoms : Atoms
        The seed structure. Never returned as a fallback: a failed inversion
        reports the optimized geometry with ``physical=False``, so that a bad
        result can never be mistaken for a good one.
    ae_model : Autoencoder
        Trained ATLAS autoencoder. Only ``ae_model.encoder`` is used, and it is
        frozen and restored around the optimization -- the same instance is
        shared with other code paths.
    mean_vals, std_vals : np.ndarray
        Standardization parameters used when training the autoencoder. These
        must be the ones the latent map was built with; standardizing with
        anything else silently relocates every target.
    settings : LatentInversionSettings | None
        Defaults are used when omitted.
    should_stop : callable | None
        Polled between L-BFGS rounds. Raises :class:`InversionCancelled` when
        it returns True -- ``QThread.terminate()`` on a torch worker is not an
        option, so cancellation has to be cooperative.
    on_round : callable | None
        ``(round_index, diagnostics)`` after each round, for progress display.

    Returns
    -------
    dict
        ``atoms`` plus the full diagnostic set: achieved latent coordinate and
        error, the four penalty terms and their weighted contributions,
        ``physical``/``min_ratio``/``n_fragments`` acceptance data,
        displacement statistics, and timings.
    """
    import torch

    settings = settings or LatentInversionSettings()
    settings.validate()

    if settings.relax_cell:
        # The geometry terms above are dense, non-periodic pdist sums: with a
        # cell free to move they would miss image contacts entirely and call a
        # perfect crystal many fragments. Refusing is better than reporting a
        # confidently wrong structure.
        raise NotImplementedError(
            'Cell relaxation is not implemented yet. The geometry penalties '
            'assume a cluster in vacuum, so relaxing the cell needs the '
            'PBC-aware edge-list terms.'
        )

    encoder = ae_model.encoder
    expected_dim = _encoder_input_dim(encoder)
    if expected_dim is not None and handle.descriptor_dim != expected_dim:
        raise ValueError(
            f'Descriptor width {handle.descriptor_dim} does not match the '
            f'autoencoder input width {expected_dim}. The latent map was '
            'built with a different descriptor or model; re-run the boundary '
            'generation, or point the traversal at the model that built it.'
        )

    t0 = time.time()
    dtype = handle.dtype
    device = handle.device

    # Freeze the encoder: only positions are optimized. Restored in `finally`,
    # since this model instance is shared with other GUI code paths.
    encoder_was_training = encoder.training
    encoder_device = next(encoder.parameters()).device
    if encoder_device != torch.device(device):
        # The descriptors live on the handle's device, so the encoder has to
        # as well; the CPU default never moves it, but a caller that asks for
        # CUDA would otherwise fail on the first forward pass with a device
        # mismatch. Restored below, since the model may be shared.
        encoder.to(device)
    encoder.eval()
    saved_requires_grad = [p.requires_grad for p in encoder.parameters()]
    for p in encoder.parameters():
        p.requires_grad_(False)

    try:
        z_t = torch.as_tensor(
            np.asarray(z_target, dtype=float).reshape(-1),
            dtype=dtype,
            device=device,
        )
        mean_t = torch.as_tensor(np.asarray(mean_vals), dtype=dtype, device=device)
        std_t = torch.as_tensor(np.asarray(std_vals), dtype=dtype, device=device)
        rmin_t = torch.as_tensor(
            covalent_min_matrix(
                initial_atoms, settings.covalent_fraction * (1.0 + settings.margin)
            ),
            dtype=dtype,
            device=device,
        )
        rmax_t = torch.as_tensor(
            covalent_min_matrix(initial_atoms, settings.bond_factor),
            dtype=dtype,
            device=device,
        )
        enc_dtype = next(encoder.parameters()).dtype
        pos0 = torch.as_tensor(initial_atoms.positions, dtype=dtype, device=device)

        def latent_of(state):
            """Structure-mean latent coordinate, differentiable in positions.

            The mean over atoms is what places training structures on the map,
            so target and data stay on the same footing.
            """
            desc = handle.descriptors(state)
            desc_std = (desc - mean_t) / std_t
            z_atoms = encoder(desc_std.to(enc_dtype))
            return z_atoms.to(dtype).mean(dim=0)

        def fresh_state(positions):
            """A batch with the neighbour list rebuilt at *positions*.

            The neighbour list is fixed when the batch is built. Atoms move far
            enough here that reusing it across the whole optimization silently
            corrupts the descriptor: at 1.5 A of displacement the per-atom
            error reaches the scale of the descriptor norm itself. Rebuilding
            each round keeps the objective, and the reported result, honest.
            """
            work = initial_atoms.copy()
            work.positions = positions
            return handle.prepare(work)

        def penalties(pos):
            return (
                repulsion_penalty(pos, rmin_t, form=settings.rep_form),
                cohesion_penalty(pos, rmax_t, settings.min_coord),
                displacement_restraint(pos, pos0, settings.d_free),
            )

        current = initial_atoms.positions.copy()
        latent_loss = float('inf')
        total_iters = 0
        rounds_used = 0

        for round_i in range(settings.n_rounds):
            if should_stop is not None and should_stop():
                raise InversionCancelled('inversion cancelled')

            ramp = settings.rep_ramp[min(round_i, len(settings.rep_ramp) - 1)]
            w_rep_i = settings.w_rep * ramp
            w_coh_i = settings.w_coh * ramp

            state = fresh_state(current)
            pos = state.free_positions
            optimizer = torch.optim.LBFGS(
                [pos],
                lr=settings.lr,
                max_iter=settings.max_iter,
                line_search_fn='strong_wolfe',
            )

            # Everything the closure reads from the enclosing loop is bound
            # as a default, so a stale round can never leak into it.
            def closure(
                state=state,
                pos=pos,
                optimizer=optimizer,
                w_rep_i=w_rep_i,
                w_coh_i=w_coh_i,
            ):
                optimizer.zero_grad()
                lat = ((latent_of(state) - z_t) ** 2).sum()
                rep, coh, dis = penalties(pos)
                loss = (
                    settings.w_lat * lat
                    + w_rep_i * rep
                    + w_coh_i * coh
                    + settings.w_disp * dis
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            total_iters += settings.max_iter
            rounds_used = round_i + 1
            current = pos.detach().cpu().numpy()

            # Evaluate against a rebuilt neighbour list, so the convergence
            # test sees the same descriptor an independent call would.
            eval_state = fresh_state(current)
            with torch.no_grad():
                latent_loss = float(((latent_of(eval_state) - z_t) ** 2).sum())
                rep_now, coh_now, disp_now = (
                    float(x) for x in penalties(eval_state.free_positions)
                )

            if on_round is not None:
                on_round(
                    round_i,
                    {
                        'latent_loss': latent_loss,
                        'rep_penalty': rep_now,
                        'coh_penalty': coh_now,
                        'disp_penalty': disp_now,
                        'w_rep': w_rep_i,
                        'w_coh': w_coh_i,
                        'elapsed_s': time.time() - t0,
                    },
                )

            # Only stop early once the barriers are at full strength; otherwise
            # a geometry that is still overlapping, or already coming apart,
            # could satisfy the test. Both penalties settle at a small positive
            # residual rather than exactly zero.
            at_full_strength = ramp >= settings.rep_ramp[-1]
            if (
                latent_loss < settings.tolerance
                and rep_now < 1e-8
                and coh_now < 1e-3
                and at_full_strength
            ):
                break

        inverted = initial_atoms.copy()
        inverted.positions = current

        final_state = fresh_state(current)
        with torch.no_grad():
            z_final = latent_of(final_state).detach().cpu().numpy()
            rep_final, coh_final, disp_final = (
                float(x) for x in penalties(final_state.free_positions)
            )
    finally:
        for p, rg in zip(encoder.parameters(), saved_requires_grad, strict=False):
            p.requires_grad_(rg)
        if encoder_was_training:
            encoder.train()
        encoder.to(encoder_device)

    ok, ratio, dist, pair, n_frag, n_iso = check_physical(
        inverted,
        settings.covalent_fraction,
        settings.bond_factor,
        settings.max_fragments,
    )
    disp = np.linalg.norm(inverted.positions - initial_atoms.positions, axis=1)
    w_rep_final = settings.w_rep * settings.rep_ramp[-1]
    w_coh_final = settings.w_coh * settings.rep_ramp[-1]
    objective_latent = settings.w_lat * latent_loss
    objective_repulsion = w_rep_final * rep_final
    objective_cohesion = w_coh_final * coh_final
    objective_displacement = settings.w_disp * disp_final

    return {
        'atoms': inverted,
        'latent': z_final,
        'latent_err': float(
            np.linalg.norm(z_final - np.asarray(z_target, dtype=float).reshape(-1))
        ),
        'latent_loss': latent_loss,
        'rep_penalty': rep_final,
        'coh_penalty': coh_final,
        'disp_penalty': disp_final,
        'w_rep_final': w_rep_final,
        'w_coh_final': w_coh_final,
        'objective_latent': objective_latent,
        'objective_repulsion': objective_repulsion,
        'objective_cohesion': objective_cohesion,
        'objective_displacement': objective_displacement,
        'objective_total': (
            objective_latent
            + objective_repulsion
            + objective_cohesion
            + objective_displacement
        ),
        'physical': bool(ok),
        'min_ratio': ratio,
        'min_dist': dist,
        'min_pair': pair,
        'n_fragments': n_frag,
        'n_isolated': n_iso,
        'max_disp': float(disp.max()) if len(disp) else 0.0,
        'mean_disp': float(disp.mean()) if len(disp) else 0.0,
        'n_moved': int((disp > 0.1).sum()),
        'n_rounds_used': rounds_used,
        'n_iters': total_iters,
        'time_s': time.time() - t0,
    }


def _encoder_input_dim(encoder) -> int | None:
    """Input width of an ``nn.Sequential`` encoder, or None if undiscoverable.

    Returning None rather than raising keeps the check advisory: a custom
    encoder that is not a plain Sequential is still usable, it just does not
    get the early, readable dimension error.
    """
    for layer in encoder:
        in_features = getattr(layer, 'in_features', None)
        if in_features is not None:
            return int(in_features)
    return None


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------


@dataclass
class SeedPool:
    """The training structures an inversion may start from.

    Deliberately lazy about geometry. The latent coordinates and atom counts
    are cheap and always present, but converting every row of an initial
    database to ``Atoms`` costs far more memory than a GUI can spend, so atoms
    are materialised only for the handful of shortlisted candidates.
    """

    #: ``(n_structures, bottleneck_dim)`` per-structure latent coordinates.
    latent: np.ndarray
    #: ``(n_structures,)`` identifiers, 1:1 with :attr:`latent`.
    unique_ids: np.ndarray
    #: ``(n_structures,)`` atom counts.
    sizes: np.ndarray
    #: ``index -> Atoms``; called at most once per index.
    loader: Callable[[int], Atoms]

    _atoms_cache: dict = field(default_factory=dict, repr=False)
    _compact_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.latent = np.asarray(self.latent, dtype=float)
        self.unique_ids = np.asarray(self.unique_ids)
        self.sizes = np.asarray(self.sizes, dtype=int)
        n = len(self.latent)
        if not (len(self.unique_ids) == len(self.sizes) == n):
            raise ValueError(
                'SeedPool latent, unique_ids and sizes must have equal length '
                f'(got {n}, {len(self.unique_ids)}, {len(self.sizes)})'
            )

    def __len__(self) -> int:
        return len(self.latent)

    def atoms(self, index: int) -> Atoms:
        """Materialise (and cache) the structure at *index*."""
        if index not in self._atoms_cache:
            self._atoms_cache[index] = self.loader(index)
        return self._atoms_cache[index]

    def is_compact(self, index: int, bond_factor: float = BOND_FACTOR) -> bool:
        """Whether structure *index* forms a single fragment.

        Cached: the scan is O(N^2) per structure, and databases of MD snapshots
        are often largely fragmented, so the shortlist gets probed repeatedly.
        """
        key = (index, bond_factor)
        if key not in self._compact_cache:
            self._compact_cache[key] = (
                connectivity(self.atoms(index), bond_factor)[0] == 1
            )
        return self._compact_cache[key]

    @classmethod
    def from_structures(cls, latent, structures, unique_ids=None) -> SeedPool:
        """Build from an in-memory list of ``Atoms`` (tests, scripts)."""
        structures = list(structures)
        if unique_ids is None:
            unique_ids = np.arange(len(structures))
        return cls(
            latent=latent,
            unique_ids=unique_ids,
            sizes=[len(s) for s in structures],
            loader=lambda i: structures[i],
        )

    @classmethod
    def from_dataframe(cls, latent, unique_ids, df, id_column=None) -> SeedPool:
        """Build from an initial-database dataframe, converting lazily.

        *unique_ids* fixes the ordering, since it is what the latent-space
        artifact stores alongside the coordinates. Ids missing from *df* are
        dropped from the pool rather than raising, so a map built before rows
        were deleted still seeds a traversal.

        Returns a pool that may be shorter than *unique_ids*; callers that
        index back into the original arrays must use
        :attr:`SeedPool.unique_ids`.
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        if id_column is None:
            id_column = 'atl_id' if 'atl_id' in df.columns else 'unique_id'
        indexed = df.set_index(id_column)

        latent = np.asarray(latent, dtype=float)
        # numpy unicode scalars compare unequal to Python str in a pandas
        # index lookup, so normalise before matching.
        wanted = [str(u) for u in np.asarray(unique_ids)]
        present = indexed.index.astype(str)
        position = {key: i for i, key in enumerate(present)}

        keep_rows, keep_ids, keep_latent = [], [], []
        for latent_index, key in enumerate(wanted):
            row_index = position.get(key)
            if row_index is None:
                continue
            keep_rows.append(row_index)
            keep_ids.append(key)
            keep_latent.append(latent[latent_index])

        structures = indexed['structure'].to_numpy()
        sizes = [structures[i].num_sites for i in keep_rows]

        def loader(i: int) -> Atoms:
            return AseAtomsAdaptor.get_atoms(structures[keep_rows[i]])

        return cls(
            latent=np.asarray(keep_latent, dtype=float).reshape(len(keep_ids), -1),
            unique_ids=np.asarray(keep_ids),
            sizes=sizes,
            loader=loader,
        )


@dataclass
class Seed:
    """One candidate starting structure for an inversion."""

    index: int
    unique_id: Any
    distance: float
    atoms: Atoms


def select_seeds(
    pool: SeedPool,
    target_latent,
    settings: LatentInversionSettings | None = None,
    shortlist_factor: int = 5,
) -> list[Seed]:
    """Pick the structures to start an inversion from, nearest first.

    The descriptor encodes the local environment within the cutoff, not the
    total number of atoms, so the seed must already contain enough atoms to
    provide the environment being decoded: a 2-atom seed can only produce
    dimer-like descriptors however far it is optimized. The size-aware
    strategies therefore restrict the pool to a window around
    ``settings.target_n_atoms``.

    Several candidates are returned because the objective is strongly
    non-convex and a single start regularly lands in a poor minimum.

    Geometry is touched only for a shortlist of ``shortlist_factor *
    n_starts`` candidates, since the compactness test needs real coordinates.
    """
    settings = settings or LatentInversionSettings()
    target = np.asarray(target_latent, dtype=float).reshape(-1)

    if len(pool) == 0:
        return []

    distances = np.linalg.norm(pool.latent - target, axis=1)
    mask = np.ones(len(pool), dtype=bool)

    if settings.seed_strategy != 'nearest' and settings.target_n_atoms is not None:
        target_n = int(settings.target_n_atoms)
        tolerance = settings.effective_size_tolerance
        # The lower bound is also held at half the target, so a small target
        # does not admit a dimer: an absolute tolerance of 10 around 9 atoms
        # would otherwise let the traversal decode a 2-atom "cluster".
        lo = max(2, target_n - tolerance, int(np.ceil(0.5 * target_n)))
        size_mask = (pool.sizes >= lo) & (pool.sizes <= target_n + tolerance)
        if not size_mask.any():
            size_mask = pool.sizes >= 2
        mask = size_mask

    if not mask.any():
        mask = np.ones(len(pool), dtype=bool)

    order = np.argsort(np.where(mask, distances, np.inf))
    order = [int(i) for i in order if mask[i]]

    if not settings.require_compact:
        chosen = order[: settings.n_starts]
    else:
        # Only materialise geometry for the shortlist. A dissociated seed stays
        # dissociated -- the optimizer moves atoms by a couple of Angstrom at
        # most and cannot reassemble a gas -- so fragmented candidates are
        # skipped unless that would leave nothing to start from.
        chosen = []
        n_shortlist = max(shortlist_factor * settings.n_starts, settings.n_starts)
        for index in order[:n_shortlist]:
            if pool.is_compact(index, settings.bond_factor):
                chosen.append(index)
            if len(chosen) == settings.n_starts:
                break
        if not chosen:
            chosen = order[: settings.n_starts]

    return [
        Seed(
            index=index,
            unique_id=pool.unique_ids[index],
            distance=float(distances[index]),
            atoms=pool.atoms(index),
        )
        for index in chosen
    ]


# ---------------------------------------------------------------------------
# Path sampling and coverage
# ---------------------------------------------------------------------------


def sample_path(waypoints, n_structures: int, snap_waypoints: bool = True):
    """Sample *n_structures* points along the polyline through *waypoints*.

    Uniform in arc length, so the decoded structures are evenly spaced in
    latent distance rather than bunched on the short legs. Endpoints are always
    included exactly; with *snap_waypoints* each interior waypoint replaces its
    nearest sample, so a marked point is guaranteed to be decoded.

    Raises
    ------
    ValueError
        If *n_structures* is smaller than the number of waypoints while
        snapping is requested, since not every waypoint could then be hit.
    """
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError('waypoints must be a non-empty (n_points, dim) array')
    if n_structures < 1:
        raise ValueError('n_structures must be at least 1')

    n_way = points.shape[0]
    if n_way == 1:
        # A single marked point is a request to decode that point, not to build
        # a path through it.
        return points.copy()
    if snap_waypoints and n_structures < n_way:
        raise ValueError(
            f'n_structures ({n_structures}) is smaller than the number of '
            f'waypoints ({n_way}); increase it or disable waypoint snapping'
        )

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total <= 0:
        # Every waypoint coincides; arc length carries no information.
        return np.repeat(points[:1], n_structures, axis=0)

    targets = np.linspace(0.0, total, n_structures)
    samples = np.column_stack(
        [np.interp(targets, cumulative, points[:, d]) for d in range(points.shape[1])]
    )

    if snap_waypoints:
        # Endpoints are already exact. Snap each interior waypoint onto its
        # nearest free sample, nearest-first so that a crowded region does not
        # let an early waypoint steal a later one's best slot.
        taken = {0, n_structures - 1}
        samples[0] = points[0]
        samples[-1] = points[-1]
        for way_index in range(1, n_way - 1):
            candidates = np.argsort(np.abs(targets - cumulative[way_index]))
            for slot in candidates:
                if int(slot) not in taken:
                    samples[int(slot)] = points[way_index]
                    taken.add(int(slot))
                    break

    return samples


def latent_neighbourhood(target_latent, latent_space, radius: float = 0.5, tree=None):
    """Distance to the nearest training point, and the count within *radius*.

    Traversal points can leave the region the data actually covers. When they
    do, no physical structure reproduces the target and the residual saturates
    at roughly this distance -- which makes the pair (residual, neighbourhood)
    a direct read-out of the coverage gap rather than an optimizer failure.
    Recording it per point lets the two be told apart afterwards.

    Pass *tree* (a prebuilt ``cKDTree``) to avoid rebuilding it per point.
    """
    from scipy.spatial import cKDTree

    if tree is None:
        tree = cKDTree(np.asarray(latent_space, dtype=float))
    target = np.asarray(target_latent, dtype=float).reshape(-1)
    distance, _ = tree.query(target)
    return float(distance), int(len(tree.query_ball_point(target, radius)))


def rank_key(result: dict) -> tuple:
    """Sort key for multi-start candidates: feasible first, then accurate.

    Among structures that pass both the bond floor and connectivity, prefer the
    one closest to the target latent coordinate. If none pass, prefer the least
    fragmented and then the least overlapping, so the reported failure is the
    best attempt rather than an arbitrary one.
    """
    if result['physical']:
        return (0, 0, result['latent_err'])
    return (1, result['n_fragments'], -result['min_ratio'])


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

#: Diagnostics copied from a result into ``atoms.info``. Kept explicit so an
#: exported extxyz stays readable and every field is one ase can serialise.
_INFO_FIELDS = (
    'latent_err',
    'latent_loss',
    'min_ratio',
    'min_dist',
    'n_fragments',
    'n_isolated',
    'max_disp',
    'mean_disp',
    'n_moved',
    'rep_penalty',
    'coh_penalty',
    'disp_penalty',
    'objective_latent',
    'objective_repulsion',
    'objective_cohesion',
    'objective_displacement',
    'objective_total',
    'physical',
    'n_rounds_used',
    'time_s',
)


def invert_one(
    handle: DifferentiableDescriptorHandle,
    target_latent,
    ae_model,
    mean_vals,
    std_vals,
    seeds: Sequence[Seed],
    settings: LatentInversionSettings | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_round: Callable[[int, dict], None] | None = None,
) -> dict:
    """Decode one latent point from several seeds and keep the best result.

    The returned structure is always an optimized one. Failures are flagged in
    ``atoms.info['physical']``, never silently replaced by the seed -- a caller
    that cannot tell a decode from its starting point would be adding untouched
    training structures back into the database as if they were new.
    """
    settings = settings or LatentInversionSettings()
    if not seeds:
        raise ValueError('no seed structures available for this latent point')

    best = None
    for seed in seeds:
        result = invert_to_latent(
            handle=handle,
            z_target=target_latent,
            initial_atoms=seed.atoms,
            ae_model=ae_model,
            mean_vals=mean_vals,
            std_vals=std_vals,
            settings=settings,
            should_stop=should_stop,
            on_round=on_round,
        )
        result['seed_index'] = seed.index
        result['seed_unique_id'] = seed.unique_id
        result['seed_distance'] = seed.distance
        if best is None or rank_key(result) < rank_key(best):
            best = result
        # A feasible, accurate solution is good enough; skip the remaining
        # restarts rather than spending the same effort again.
        if best['physical'] and best['latent_err'] < settings.good_enough_latent_err:
            break

    best['target_latent'] = np.asarray(target_latent, dtype=float).reshape(-1)
    return best


def annotate_atoms(result: dict) -> Atoms:
    """Return a copy of the decoded structure carrying its diagnostics.

    Everything worth reporting is written into ``atoms.info`` so that it
    survives an extxyz export and can be read back without the result dict.
    """
    atoms = result['atoms'].copy()
    atoms.info['latent_decoded'] = True
    atoms.info['target_latent'] = np.asarray(result['target_latent'], dtype=float)
    atoms.info['latent_achieved'] = np.asarray(result['latent'], dtype=float)
    pair = result.get('min_pair')
    atoms.info['min_pair'] = '-'.join(pair) if pair else 'none'
    for key in _INFO_FIELDS:
        if key in result:
            atoms.info[key] = result[key]
    for key in (
        'seed_index',
        'seed_unique_id',
        'seed_distance',
        'target_nn_dist',
        'target_n_within',
    ):
        if result.get(key) is not None:
            atoms.info[key] = result[key]
    return atoms


def traverse_latent_path(
    handle: DifferentiableDescriptorHandle,
    targets,
    pool: SeedPool,
    ae_model,
    mean_vals,
    std_vals,
    settings: LatentInversionSettings | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_result: Callable[[int, dict], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
):
    """Decode every point of *targets*, yielding one result per point.

    A generator so a GUI can draw each structure as it appears rather than
    waiting for the whole path. Cancellation is cooperative: *should_stop* is
    polled between points and inside each inversion, because terminating a
    thread mid-optimization is not safe with torch.

    With ``settings.chain_from_previous`` the previous decode is offered as an
    extra seed. It usually wins on latent distance, which makes the path
    visibly smoother and avoids re-converging from a distant training structure
    at every step.
    """
    from scipy.spatial import cKDTree

    settings = settings or LatentInversionSettings()
    settings.validate()

    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)

    tree = cKDTree(pool.latent) if len(pool) else None
    previous = None

    for point_index, target in enumerate(targets):
        if should_stop is not None and should_stop():
            return

        if on_progress is not None:
            on_progress(
                point_index,
                len(targets),
                f'Decoding point {point_index + 1} of {len(targets)}',
            )

        seeds = select_seeds(pool, target, settings)
        if previous is not None and settings.chain_from_previous:
            chained = Seed(
                index=-1,
                unique_id=None,
                distance=float(np.linalg.norm(np.asarray(previous['latent']) - target)),
                atoms=previous['atoms'],
            )
            # First, so that a chained seed is tried before falling back to the
            # database; rank_key still arbitrates the final choice.
            seeds = [chained, *seeds]
        if not seeds:
            raise ValueError(
                'No seed structures are available. The latent map has no '
                'entries that match the initial database.'
            )

        try:
            result = invert_one(
                handle=handle,
                target_latent=target,
                ae_model=ae_model,
                mean_vals=mean_vals,
                std_vals=std_vals,
                seeds=seeds,
                settings=settings,
                should_stop=should_stop,
            )
        except InversionCancelled:
            return

        if tree is not None:
            nn_dist, n_within = latent_neighbourhood(
                target, pool.latent, settings.neighbourhood_radius, tree=tree
            )
            result['target_nn_dist'] = nn_dist
            result['target_n_within'] = n_within

        result['point_index'] = point_index
        result['annotated_atoms'] = annotate_atoms(result)
        previous = result

        if on_result is not None:
            on_result(point_index, result)
        yield result
