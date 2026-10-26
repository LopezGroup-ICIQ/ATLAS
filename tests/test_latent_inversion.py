"""Tests for latent-space inversion.

Everything here runs without MACE or a GPU. The optimizer is exercised against
a *fake* differentiable descriptor handle -- a toy but genuinely differentiable
function of the positions -- so the round loop, the neighbour-list rebuild, the
encoder freeze and cancellation are all covered without loading a 30 MB model.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from ase import Atoms  # noqa: E402

from atlas.active_learning.extrapolation.latent_inversion import (  # noqa: E402
    BOND_FACTOR,
    COVALENT_FRACTION,
    InversionCancelled,
    LatentInversionSettings,
    Seed,
    SeedPool,
    annotate_atoms,
    check_physical,
    cohesion_penalty,
    connectivity,
    covalent_min_matrix,
    displacement_restraint,
    invert_one,
    invert_to_latent,
    latent_neighbourhood,
    min_distance_ratio,
    rank_key,
    repulsion_penalty,
    sample_path,
    select_seeds,
    traverse_latent_path,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

#: Width of the toy descriptor; small enough to train an autoencoder instantly.
FAKE_DIM = 6


class FakeBatchState:
    """Minimal ``DescriptorBatchState``: a leaf and the positions it feeds."""

    def __init__(self, positions):
        self.free_positions = positions
        self.positions = positions
        self.strain = None
        self.cell = None
        self.edge_index = None
        self.shifts = None


class FakeHandle:
    """A differentiable per-atom descriptor with no external dependencies.

    Each atom is described by smooth, decreasing functions of its distances to
    the others. That is enough to make the composed structure-to-latent map
    non-trivial and genuinely invertible by gradient descent, which is what the
    optimizer tests need -- the real MACE handle is verified separately in
    ``tests/test_backends.py``.
    """

    def __init__(self, dim: int = FAKE_DIM):
        self.device = 'cpu'
        self.dtype = torch.float64
        self.descriptor_dim = dim
        self.supports_cell_gradients = False
        self.n_prepare_calls = 0

    def prepare(self, atoms, strain=None):
        self.n_prepare_calls += 1
        positions = torch.tensor(
            np.asarray(atoms.positions), dtype=self.dtype, requires_grad=True
        )
        return FakeBatchState(positions)

    def descriptors(self, state):
        pos = state.positions
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        d = torch.sqrt((diff**2).sum(-1) + 1e-9)
        eye = torch.eye(pos.shape[0], dtype=self.dtype)
        # Mask self-pairs out of every reduction below.
        d = d + eye * 1e6
        columns = []
        for k in range(self.descriptor_dim):
            decay = torch.exp(-d / (1.5 + 0.5 * k))
            columns.append(decay.sum(dim=1))
        return torch.stack(columns, dim=1)


def _autoencoder(input_dim: int = FAKE_DIM, seed: int = 0):
    from atlas.active_learning.extrapolation.autoencoder import Autoencoder

    torch.manual_seed(seed)
    model = Autoencoder(input_dim=input_dim, l1_dim=16, l2_dim=8, bottleneck_dim=2)
    return model.double()


def _cluster(n: int = 6, spacing: float = 2.55, seed: int = 0) -> Atoms:
    """A compact, well-coordinated Cu cluster to seed inversions from.

    Carved out of an fcc lattice rather than grown by a random walk: a walk
    produces branching chains whose end atoms have a single neighbour, which
    the cohesion term correctly -- but unhelpfully, for a fixture -- penalises.
    """
    from ase.build import bulk

    lattice = bulk('Cu', 'fcc', a=spacing * np.sqrt(2.0), cubic=True) * (3, 3, 3)
    lattice.set_pbc(False)
    positions = lattice.positions - lattice.positions.mean(axis=0)
    nearest = np.argsort(np.linalg.norm(positions, axis=1))[:n]

    atoms = Atoms(f'Cu{n}', positions=positions[nearest])
    if seed:
        atoms.rattle(0.05, seed=seed)
    # A generous vacuum box, so the structure survives a pymatgen round-trip.
    atoms.set_cell(np.eye(3) * 30.0)
    atoms.center()
    return atoms


def _latent_of(handle, model, atoms, mean, std):
    with torch.no_grad():
        desc = handle.descriptors(handle.prepare(atoms))
        z = model.encoder((desc - torch.tensor(mean)) / torch.tensor(std))
        return z.mean(dim=0).numpy()


@pytest.fixture
def inversion_setup():
    """A handle, an autoencoder, standardization stats and a seed cluster."""
    handle = FakeHandle()
    model = _autoencoder()
    atoms = _cluster()
    desc = handle.descriptors(handle.prepare(atoms)).detach().numpy()
    mean = desc.mean(axis=0)
    std = np.where(desc.std(axis=0) == 0, 1.0, desc.std(axis=0))
    return handle, model, atoms, mean, std


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    """Acceptance tests that decide whether a decoded structure is usable."""

    def test_covalent_matrix_has_zero_diagonal(self):
        atoms = Atoms('CuO', positions=[[0, 0, 0], [2.0, 0, 0]])
        rmin = covalent_min_matrix(atoms)
        # A non-zero diagonal would make every atom repel itself.
        assert np.allclose(np.diag(rmin), 0.0)
        assert rmin[0, 1] == pytest.approx(rmin[1, 0])
        assert rmin[0, 1] > 0

    def test_min_distance_ratio_finds_the_closest_pair(self):
        atoms = Atoms('Cu3', positions=[[0, 0, 0], [5.0, 0, 0], [0.5, 0, 0]])
        ratio, dist, pair = min_distance_ratio(atoms)
        assert dist == pytest.approx(0.5)
        assert pair == ('Cu', 'Cu')
        assert ratio < 0.5

    def test_single_atom_has_no_pairs(self):
        ratio, dist, pair = min_distance_ratio(Atoms('Cu', positions=[[0, 0, 0]]))
        assert np.isinf(ratio) and np.isinf(dist) and pair is None
        assert connectivity(Atoms('Cu', positions=[[0, 0, 0]])) == (1, 0)

    def test_compact_cluster_is_one_fragment(self):
        n_frag, n_iso = connectivity(_cluster(6, spacing=2.4))
        assert n_frag == 1
        assert n_iso == 0

    def test_dissociated_cluster_is_many_fragments(self):
        atoms = Atoms('Cu3', positions=[[0, 0, 0], [30.0, 0, 0], [0, 30.0, 0]])
        n_frag, n_iso = connectivity(atoms)
        assert n_frag == 3
        assert n_iso == 3

    def test_check_physical_rejects_overlap(self):
        atoms = Atoms('Cu2', positions=[[0, 0, 0], [0.4, 0, 0]])
        ok, ratio, _dist, _pair, _frag, _iso = check_physical(atoms)
        assert not ok
        assert ratio < COVALENT_FRACTION

    def test_check_physical_rejects_dissociation(self):
        atoms = Atoms('Cu2', positions=[[0, 0, 0], [30.0, 0, 0]])
        ok, ratio, _dist, _pair, n_frag, _iso = check_physical(atoms)
        # Well separated, so no overlap -- rejected purely on connectivity.
        assert ratio > COVALENT_FRACTION
        assert n_frag == 2
        assert not ok

    def test_check_physical_accepts_a_reasonable_cluster(self):
        ok, ratio, _dist, _pair, n_frag, _iso = check_physical(_cluster(5, spacing=2.4))
        assert ok
        assert ratio >= COVALENT_FRACTION
        assert n_frag == 1

    def test_max_fragments_can_be_relaxed(self):
        atoms = Atoms('Cu2', positions=[[0, 0, 0], [30.0, 0, 0]])
        assert not check_physical(atoms, max_fragments=1)[0]
        assert check_physical(atoms, max_fragments=2)[0]


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def _pair(distance: float):
    return torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )


def _floor(value: float = 2.0):
    return torch.tensor([[0.0, value], [value, 0.0]], dtype=torch.float64)


class TestRepulsionPenalty:
    """The barrier must be inert above the floor and steep below it."""

    FORMS = ('inverse', 'quadratic')

    @pytest.mark.parametrize('form', FORMS)
    def test_zero_above_the_floor(self, form):
        value = repulsion_penalty(_pair(3.0), _floor(2.0), form=form)
        assert float(value) == pytest.approx(0.0)

    @pytest.mark.parametrize('form', FORMS)
    def test_zero_gradient_above_the_floor(self, form):
        """A one-sided term must not bias the search where it is satisfied."""
        pos = _pair(3.0)
        repulsion_penalty(pos, _floor(2.0), form=form).backward()
        assert np.abs(pos.grad.numpy()).max() == pytest.approx(0.0)

    @pytest.mark.parametrize('form', FORMS)
    def test_zero_at_the_floor(self, form):
        value = repulsion_penalty(_pair(2.0), _floor(2.0), form=form)
        assert float(value) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize('form', FORMS)
    def test_increases_monotonically_below_the_floor(self, form):
        values = [
            float(repulsion_penalty(_pair(d), _floor(2.0), form=form))
            for d in (1.9, 1.5, 1.0, 0.5)
        ]
        assert all(b > a for a, b in zip(values, values[1:], strict=False))

    def test_inverse_form_is_far_stiffer_than_quadratic(self):
        """Why 'inverse' is the default: it cannot be outbid near contact."""
        inverse = float(repulsion_penalty(_pair(1.0), _floor(2.0), form='inverse'))
        quadratic = float(repulsion_penalty(_pair(1.0), _floor(2.0), form='quadratic'))
        assert inverse > 100 * quadratic

    def test_inverse_form_stays_finite_at_contact(self):
        """A line search may briefly propose coincident atoms in float64."""
        value = float(repulsion_penalty(_pair(1e-6), _floor(2.0), form='inverse'))
        assert np.isfinite(value)

    def test_unknown_form_raises(self):
        with pytest.raises(ValueError, match='unknown repulsion form'):
            repulsion_penalty(_pair(1.0), _floor(2.0), form='gaussian')


class TestCohesionPenalty:
    """The term that stops the optimizer dissociating the cluster."""

    def test_small_for_a_well_connected_cluster(self):
        atoms = _cluster(6, spacing=2.4)
        pos = torch.tensor(atoms.positions, dtype=torch.float64)
        rmax = torch.tensor(covalent_min_matrix(atoms, BOND_FACTOR))
        assert float(cohesion_penalty(pos, rmax, min_coord=2.0)) < 1e-3

    def test_large_for_a_dissociated_cluster(self):
        atoms = Atoms('Cu3', positions=[[0, 0, 0], [30.0, 0, 0], [0, 30.0, 0]])
        pos = torch.tensor(atoms.positions, dtype=torch.float64)
        rmax = torch.tensor(covalent_min_matrix(atoms, BOND_FACTOR))
        assert float(cohesion_penalty(pos, rmax, min_coord=2.0)) > 1.0

    def test_keeps_pulling_on_an_atom_that_already_broke_away(self):
        """The separation term must not saturate, or escape becomes free.

        The coordination term alone saturates once an atom is well outside the
        cutoff, which would leave a dissociating atom with no restoring
        gradient at exactly the point it is needed.
        """
        far = Atoms('Cu3', positions=[[0, 0, 0], [2.4, 0, 0], [40.0, 0, 0]])
        farther = Atoms('Cu3', positions=[[0, 0, 0], [2.4, 0, 0], [60.0, 0, 0]])
        rmax = torch.tensor(covalent_min_matrix(far, BOND_FACTOR))
        a = float(
            cohesion_penalty(torch.tensor(far.positions, dtype=torch.float64), rmax)
        )
        b = float(
            cohesion_penalty(torch.tensor(farther.positions, dtype=torch.float64), rmax)
        )
        assert b > a


class TestDisplacementRestraint:
    """The optional search regularizer around the seed geometry."""

    def test_flat_bottom_inside_the_free_radius(self):
        pos0 = torch.zeros(3, 3, dtype=torch.float64)
        pos = pos0.clone().requires_grad_(True)
        with torch.no_grad():
            pos[0, 0] = 0.9
        value = displacement_restraint(pos, pos0, d_free=1.5)
        assert float(value) == pytest.approx(0.0)

    def test_grows_beyond_the_free_radius(self):
        pos0 = torch.zeros(3, 3, dtype=torch.float64)
        near = pos0.clone()
        near[0, 0] = 2.0
        far = pos0.clone()
        far[0, 0] = 4.0
        assert float(displacement_restraint(far, pos0, 1.5)) > float(
            displacement_restraint(near, pos0, 1.5)
        )

    def test_is_a_mean_so_it_does_not_scale_with_system_size(self):
        small = torch.zeros(2, 3, dtype=torch.float64)
        big = torch.zeros(20, 3, dtype=torch.float64)
        moved_small = small.clone()
        moved_small[0, 0] = 4.0
        moved_big = big.clone()
        moved_big[0, 0] = 4.0
        # Same single displacement in a 10x bigger system -> 10x smaller mean.
        assert float(displacement_restraint(moved_small, small, 1.5)) > float(
            displacement_restraint(moved_big, big, 1.5)
        )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestLatentInversionSettings:
    """Persistence and validation of the traversal configuration."""

    def test_round_trips_through_a_dict(self):
        original = LatentInversionSettings(n_structures=7, w_disp=0.0)
        restored = LatentInversionSettings.from_dict(original.to_dict())
        assert restored == original

    def test_unknown_keys_are_ignored(self):
        """Settings written by another version must not break loading."""
        restored = LatentInversionSettings.from_dict(
            {'n_structures': 5, 'from_the_future': 'nonsense'}
        )
        assert restored.n_structures == 5

    def test_empty_payload_gives_defaults(self):
        assert LatentInversionSettings.from_dict(None) == LatentInversionSettings()
        assert LatentInversionSettings.from_dict({}) == LatentInversionSettings()

    def test_default_decodes_one_structure(self):
        """A first run should decode exactly what was marked, not a path."""
        assert LatentInversionSettings().n_structures == 1

    def test_rep_ramp_is_normalised_to_a_tuple(self):
        """JSON round-trips tuples as lists; rep_ramp[-1] must still work."""
        restored = LatentInversionSettings.from_dict({'rep_ramp': [0.5, 2.0]})
        assert restored.rep_ramp == (0.5, 2.0)

    @pytest.mark.parametrize(
        ('overrides', 'message'),
        [
            ({'n_structures': 0}, 'n_structures'),
            ({'n_starts': 0}, 'n_starts'),
            ({'n_rounds': 0}, 'n_rounds'),
            ({'rep_ramp': ()}, 'rep_ramp'),
            ({'rep_ramp': (0.0, 1.0)}, 'positive'),
            ({'rep_form': 'gaussian'}, 'repulsion form'),
            ({'optimizer': 'adam'}, 'optimizer'),
            ({'seed_strategy': 'random'}, 'seed strategy'),
            ({'seed_strategy': 'by_size'}, 'target_n_atoms'),
            ({'covalent_fraction': 1.5}, 'covalent_fraction'),
            ({'bond_factor': 0.5}, 'bond_factor'),
            ({'lr': 0.0}, 'lr'),
        ],
    )
    def test_validate_rejects_bad_combinations(self, overrides, message):
        settings = LatentInversionSettings(**overrides)
        with pytest.raises(ValueError, match=message):
            settings.validate()

    def test_fixed_size_pins_the_tolerance_to_zero(self):
        settings = LatentInversionSettings(
            seed_strategy='fixed_size', target_n_atoms=8, size_tolerance=10
        )
        assert settings.effective_size_tolerance == 0


# ---------------------------------------------------------------------------
# Path sampling
# ---------------------------------------------------------------------------


class TestSamplePath:
    """Turning marked waypoints into the points that get decoded."""

    def test_endpoints_are_exact(self):
        way = np.array([[0.0, 0.0], [1.0, 2.0]])
        samples = sample_path(way, 5)
        assert np.allclose(samples[0], way[0])
        assert np.allclose(samples[-1], way[-1])
        assert samples.shape == (5, 2)

    def test_samples_are_uniform_in_arc_length(self):
        """Uneven legs must not bunch structures on the short one."""
        way = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        samples = sample_path(way, 10, snap_waypoints=False)
        steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        assert np.allclose(steps, steps[0])

    def test_interior_waypoints_are_hit_when_snapping(self):
        way = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        samples = sample_path(way, 10, snap_waypoints=True)
        for point in way:
            assert np.abs(samples - point).sum(axis=1).min() == pytest.approx(0.0)

    def test_snapping_does_not_collapse_two_waypoints_onto_one_slot(self):
        way = np.array([[0.0, 0.0], [4.9, 0.0], [5.1, 0.0], [10.0, 0.0]])
        samples = sample_path(way, 6, snap_waypoints=True)
        assert len({tuple(row) for row in samples}) == 6

    def test_a_single_waypoint_decodes_that_point(self):
        samples = sample_path(np.array([[1.0, 2.0]]), 5)
        assert samples.shape == (1, 2)
        assert np.allclose(samples[0], [1.0, 2.0])

    def test_coincident_waypoints_do_not_divide_by_zero(self):
        samples = sample_path(np.array([[1.0, 2.0], [1.0, 2.0]]), 4)
        assert samples.shape == (4, 2)
        assert np.allclose(samples, [1.0, 2.0])

    def test_too_few_structures_to_snap_every_waypoint_raises(self):
        way = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        with pytest.raises(ValueError, match='smaller than the number of'):
            sample_path(way, 3, snap_waypoints=True)
        # Without snapping there is nothing to guarantee, so it is allowed.
        assert sample_path(way, 3, snap_waypoints=False).shape == (3, 2)

    def test_empty_waypoints_raise(self):
        with pytest.raises(ValueError, match='non-empty'):
            sample_path(np.zeros((0, 2)), 4)


# ---------------------------------------------------------------------------
# Ranking and coverage
# ---------------------------------------------------------------------------


def _result(physical, latent_err=0.1, n_fragments=1, min_ratio=0.9):
    return {
        'physical': physical,
        'latent_err': latent_err,
        'n_fragments': n_fragments,
        'min_ratio': min_ratio,
    }


class TestRankKey:
    """How multi-start candidates are compared."""

    def test_physical_beats_non_physical_however_accurate(self):
        """Accuracy must never buy its way past the feasibility test."""
        physical = _result(True, latent_err=5.0)
        accurate = _result(False, latent_err=1e-9)
        assert rank_key(physical) < rank_key(accurate)

    def test_among_physical_the_most_accurate_wins(self):
        assert rank_key(_result(True, 0.01)) < rank_key(_result(True, 0.5))

    def test_among_failures_the_least_fragmented_wins(self):
        assert rank_key(_result(False, n_fragments=1)) < rank_key(
            _result(False, n_fragments=4)
        )

    def test_among_equally_fragmented_failures_the_least_overlapping_wins(self):
        assert rank_key(_result(False, n_fragments=2, min_ratio=0.8)) < rank_key(
            _result(False, n_fragments=2, min_ratio=0.3)
        )


class TestLatentNeighbourhood:
    """Coverage read-out that separates a data gap from an optimizer failure."""

    def test_reports_distance_and_count(self):
        cloud = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        distance, count = latent_neighbourhood([0.0, 0.0], cloud, radius=0.5)
        assert distance == pytest.approx(0.0)
        assert count == 2

    def test_a_point_outside_the_data_reports_an_empty_neighbourhood(self):
        cloud = np.array([[0.0, 0.0], [0.1, 0.0]])
        distance, count = latent_neighbourhood([20.0, 20.0], cloud, radius=0.5)
        assert distance > 20
        assert count == 0


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------


def _pool(sizes, latents, spacing=2.4):
    structures = [_cluster(n, spacing=spacing, seed=i) for i, n in enumerate(sizes)]
    return SeedPool.from_structures(np.asarray(latents, dtype=float), structures)


class TestSeedPool:
    """The lazily materialised pool of candidate starting structures."""

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match='equal length'):
            SeedPool(
                latent=np.zeros((3, 2)),
                unique_ids=np.array(['a', 'b']),
                sizes=np.array([1, 2, 3]),
                loader=lambda i: None,
            )

    def test_atoms_are_loaded_lazily_and_cached(self):
        calls = []

        def loader(index):
            calls.append(index)
            return _cluster(4)

        pool = SeedPool(
            latent=np.zeros((3, 2)),
            unique_ids=np.array(['a', 'b', 'c']),
            sizes=np.array([4, 4, 4]),
            loader=loader,
        )
        assert calls == []
        pool.atoms(1)
        pool.atoms(1)
        assert calls == [1]

    def test_from_dataframe_drops_ids_missing_from_the_database(self):
        """A map built before rows were deleted must still seed a traversal."""
        pandas = pytest.importorskip('pandas')
        pytest.importorskip('pymatgen')
        from pymatgen.io.ase import AseAtomsAdaptor

        structures = [AseAtomsAdaptor.get_structure(_cluster(4)) for _ in range(2)]
        df = pandas.DataFrame(
            {'unique_id': ['keep-a', 'keep-b'], 'structure': structures}
        )
        latent = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pool = SeedPool.from_dataframe(latent, ['keep-a', 'gone', 'keep-b'], df)
        assert list(pool.unique_ids) == ['keep-a', 'keep-b']
        # The surviving latent rows must stay paired with their own ids.
        assert np.allclose(pool.latent, [[0.0, 0.0], [2.0, 2.0]])


class TestSelectSeeds:
    """Choosing which structures an inversion starts from."""

    def test_nearest_strategy_returns_candidates_in_distance_order(self):
        pool = _pool([5, 5, 5], [[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
        seeds = select_seeds(pool, [0.0, 0.0], LatentInversionSettings(n_starts=3))
        assert [s.index for s in seeds] == [0, 2, 1]
        assert [s.distance for s in seeds] == pytest.approx([0.0, 1.0, 3.0])

    def test_returns_several_starts_because_the_objective_is_non_convex(self):
        pool = _pool([5] * 5, [[float(i), 0.0] for i in range(5)])
        seeds = select_seeds(pool, [0.0, 0.0], LatentInversionSettings(n_starts=3))
        assert len(seeds) == 3

    def test_size_strategy_excludes_structures_outside_the_window(self):
        pool = _pool([2, 12, 3], [[0.0, 0.0], [5.0, 0.0], [0.1, 0.0]])
        settings = LatentInversionSettings(
            seed_strategy='by_size', target_n_atoms=12, size_tolerance=2, n_starts=3
        )
        seeds = select_seeds(pool, [0.0, 0.0], settings)
        # Only the 12-atom structure fits, despite being the furthest away.
        assert [s.index for s in seeds] == [1]

    def test_small_targets_do_not_admit_a_dimer(self):
        """The half-target floor: tolerance 10 around 9 atoms must exclude 2."""
        pool = _pool([2, 9], [[0.0, 0.0], [5.0, 0.0]])
        settings = LatentInversionSettings(
            seed_strategy='by_size', target_n_atoms=9, size_tolerance=10, n_starts=2
        )
        seeds = select_seeds(pool, [0.0, 0.0], settings)
        assert [s.index for s in seeds] == [1]

    def test_fragmented_seeds_are_skipped(self):
        """A dissociated start stays dissociated; the optimizer cannot heal it."""
        fragmented = Atoms('Cu3', positions=[[0, 0, 0], [30.0, 0, 0], [0, 30.0, 0]])
        pool = SeedPool.from_structures(
            np.array([[0.0, 0.0], [2.0, 0.0]]), [fragmented, _cluster(5)]
        )
        seeds = select_seeds(pool, [0.0, 0.0], LatentInversionSettings(n_starts=1))
        assert [s.index for s in seeds] == [1]

    def test_compactness_is_dropped_rather_than_returning_nothing(self):
        fragmented = Atoms('Cu2', positions=[[0, 0, 0], [30.0, 0, 0]])
        pool = SeedPool.from_structures(np.array([[0.0, 0.0]]), [fragmented])
        seeds = select_seeds(pool, [0.0, 0.0], LatentInversionSettings(n_starts=1))
        assert len(seeds) == 1

    def test_compactness_check_can_be_disabled(self):
        fragmented = Atoms('Cu3', positions=[[0, 0, 0], [30.0, 0, 0], [0, 30.0, 0]])
        pool = SeedPool.from_structures(
            np.array([[0.0, 0.0], [2.0, 0.0]]), [fragmented, _cluster(5)]
        )
        seeds = select_seeds(
            pool,
            [0.0, 0.0],
            LatentInversionSettings(n_starts=1, require_compact=False),
        )
        assert [s.index for s in seeds] == [0]

    def test_an_empty_pool_yields_no_seeds(self):
        pool = SeedPool.from_structures(np.zeros((0, 2)), [])
        assert select_seeds(pool, [0.0, 0.0]) == []


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------


def _reachable_target(handle, model, mean, std, atoms, shift=0.15):
    """A latent target near, but not at, a real structure's coordinate.

    Targets picked out of thin air may be unreachable by any physical geometry,
    which would make an "error decreased" assertion a coin flip. Perturbing a
    reachable point keeps the test about the optimizer, not about coverage.
    """
    base = _latent_of(handle, model, atoms, mean, std)
    return base + shift


class TestInvertToLatent:
    """The constrained optimizer itself."""

    def test_reduces_the_latent_error(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        target = _reachable_target(handle, model, mean, std, atoms)
        start_err = float(
            np.linalg.norm(_latent_of(handle, model, atoms, mean, std) - target)
        )

        result = invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=4, max_iter=40),
        )
        assert result['latent_err'] < start_err

    def test_result_stays_physical(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        target = _reachable_target(handle, model, mean, std, atoms)
        result = invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=4, max_iter=40),
        )
        assert result['physical']
        assert result['min_ratio'] >= COVALENT_FRACTION
        assert result['n_fragments'] == 1

    def test_the_seed_is_never_returned_unchanged(self, inversion_setup):
        """A decode must be distinguishable from its starting structure.

        Silently falling back to the seed would let untouched training
        structures be added to the database as if they were new candidates.
        """
        handle, model, atoms, mean, std = inversion_setup
        target = _reachable_target(handle, model, mean, std, atoms, shift=0.3)
        result = invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=3, max_iter=40),
        )
        assert result['atoms'] is not atoms
        assert result['max_disp'] > 1e-6

    def test_rebuilds_the_neighbour_list_every_round(self, inversion_setup):
        """A stale neighbour list silently corrupts the descriptor."""
        handle, model, atoms, mean, std = inversion_setup
        target = _reachable_target(handle, model, mean, std, atoms)
        before = handle.n_prepare_calls
        invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=3, max_iter=5, tolerance=0.0),
        )
        # Three rounds, each with an optimization batch and an evaluation
        # batch, plus one final evaluation.
        assert handle.n_prepare_calls - before >= 7

    def test_restores_the_encoder_grad_flags(self, inversion_setup):
        """The autoencoder instance is shared with other code paths."""
        handle, model, atoms, mean, std = inversion_setup
        for p in model.encoder.parameters():
            p.requires_grad_(True)
        model.encoder.train()

        invert_to_latent(
            handle,
            [0.0, 0.0],
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=1, max_iter=5),
        )
        assert all(p.requires_grad for p in model.encoder.parameters())
        assert model.encoder.training

    def test_restores_the_encoder_even_when_cancelled(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        for p in model.encoder.parameters():
            p.requires_grad_(True)

        with pytest.raises(InversionCancelled):
            invert_to_latent(
                handle,
                [0.0, 0.0],
                atoms,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=3, max_iter=5),
                should_stop=lambda: True,
            )
        assert all(p.requires_grad for p in model.encoder.parameters())

    def test_should_stop_aborts_between_rounds(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        rounds_seen = []

        def stop():
            return len(rounds_seen) >= 2

        with pytest.raises(InversionCancelled):
            invert_to_latent(
                handle,
                [0.0, 0.0],
                atoms,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=10, max_iter=5),
                should_stop=stop,
                on_round=lambda i, d: rounds_seen.append(i),
            )
        assert len(rounds_seen) == 2

    def test_on_round_reports_every_term(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        seen = []
        invert_to_latent(
            handle,
            [0.0, 0.0],
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=2, max_iter=5, tolerance=0.0),
            on_round=lambda i, d: seen.append((i, d)),
        )
        assert [i for i, _ in seen] == [0, 1]
        for _, diagnostics in seen:
            assert {
                'latent_loss',
                'rep_penalty',
                'coh_penalty',
                'disp_penalty',
                'w_rep',
                'w_coh',
                'elapsed_s',
            } <= set(diagnostics)

    def test_ramp_raises_the_barrier_weight_over_rounds(self, inversion_setup):
        """Ramping keeps strong-Wolfe off a stiff barrier in early rounds."""
        handle, model, atoms, mean, std = inversion_setup
        seen = []
        invert_to_latent(
            handle,
            [0.0, 0.0],
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(
                n_rounds=4, max_iter=5, tolerance=0.0, rep_ramp=(0.1, 1.0, 10.0)
            ),
            on_round=lambda i, d: seen.append(d['w_rep']),
        )
        assert seen == pytest.approx([1.0, 10.0, 100.0, 100.0])

    def test_does_not_stop_early_before_full_barrier_strength(self, inversion_setup):
        """A still-overlapping geometry can satisfy a loose latent tolerance.

        The early exit is guarded on the ramp having reached full strength, so
        a structure cannot be accepted while the barriers are still soft.
        """
        handle, model, atoms, mean, std = inversion_setup
        target = _latent_of(handle, model, atoms, mean, std)
        seen = []
        invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(
                n_rounds=5, max_iter=5, tolerance=1e9, rep_ramp=(0.1, 1.0, 10.0)
            ),
            on_round=lambda i, d: seen.append(i),
        )
        # Tolerance is satisfied from round 0, so only the ramp guard can be
        # holding it: the run must reach the round where the ramp saturates.
        assert seen == [0, 1, 2]

    def test_zero_disp_weight_allows_a_larger_rearrangement(self, inversion_setup):
        """w_disp=0 is the exploratory decode described in the thesis."""
        handle, model, atoms, mean, std = inversion_setup
        target = _reachable_target(handle, model, mean, std, atoms, shift=0.8)
        common = {'n_rounds': 4, 'max_iter': 40, 'd_free': 0.1}

        restrained = invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(w_disp=1000.0, **common),
        )
        free = invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(w_disp=0.0, **common),
        )
        assert free['max_disp'] > restrained['max_disp']

    def test_descriptor_width_mismatch_is_reported_clearly(self, inversion_setup):
        handle, _model, atoms, mean, std = inversion_setup
        wrong = _autoencoder(input_dim=FAKE_DIM + 3)
        with pytest.raises(ValueError, match='does not match the autoencoder'):
            invert_to_latent(handle, [0.0, 0.0], atoms, wrong, mean, std)

    def test_cell_relaxation_refuses_rather_than_silently_ignoring(
        self, inversion_setup
    ):
        handle, model, atoms, mean, std = inversion_setup
        with pytest.raises(NotImplementedError, match='Cell relaxation'):
            invert_to_latent(
                handle,
                [0.0, 0.0],
                atoms,
                model,
                mean,
                std,
                LatentInversionSettings(relax_cell=True),
            )

    def test_invalid_settings_are_rejected_before_any_work(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        before = handle.n_prepare_calls
        with pytest.raises(ValueError):
            invert_to_latent(
                handle,
                [0.0, 0.0],
                atoms,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=0),
            )
        assert handle.n_prepare_calls == before

    def test_result_carries_the_full_diagnostic_set(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        result = invert_to_latent(
            handle,
            [0.0, 0.0],
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=1, max_iter=5),
        )
        assert {
            'atoms',
            'latent',
            'latent_err',
            'latent_loss',
            'rep_penalty',
            'coh_penalty',
            'disp_penalty',
            'objective_total',
            'physical',
            'min_ratio',
            'min_dist',
            'min_pair',
            'n_fragments',
            'n_isolated',
            'max_disp',
            'mean_disp',
            'n_moved',
            'n_rounds_used',
            'n_iters',
            'time_s',
        } <= set(result)
        assert result['objective_total'] == pytest.approx(
            result['objective_latent']
            + result['objective_repulsion']
            + result['objective_cohesion']
            + result['objective_displacement']
        )


class TestInvertOne:
    """Multi-start decoding of a single latent point."""

    def test_keeps_the_best_candidate_by_rank(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        fragmented = Atoms(
            'Cu6',
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [40, 0, 0],
                [0, 40, 0],
                [0, 0, 40],
                [40, 40, 0],
            ],
        )
        seeds = [
            Seed(0, 'bad', 0.0, fragmented),
            Seed(1, 'good', 1.0, atoms),
        ]
        result = invert_one(
            handle,
            _latent_of(handle, model, atoms, mean, std),
            model,
            mean,
            std,
            seeds,
            LatentInversionSettings(n_rounds=2, max_iter=20),
        )
        # The fragmented seed is nearer in latent space but cannot produce a
        # physical result, so the compact one has to win.
        assert result['seed_unique_id'] == 'good'
        assert result['physical']

    def test_stops_restarting_once_a_good_candidate_is_found(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        target = _latent_of(handle, model, atoms, mean, std)
        seeds = [Seed(i, f's{i}', 0.0, atoms.copy()) for i in range(4)]
        result = invert_one(
            handle,
            target,
            model,
            mean,
            std,
            seeds,
            LatentInversionSettings(
                n_rounds=3, max_iter=30, good_enough_latent_err=1.0
            ),
        )
        # The first start already satisfies the loose threshold.
        assert result['seed_unique_id'] == 's0'

    def test_no_seeds_is_an_error_not_a_silent_skip(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        with pytest.raises(ValueError, match='no seed structures'):
            invert_one(handle, [0.0, 0.0], model, mean, std, [])

    def test_records_the_target(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        result = invert_one(
            handle,
            [0.25, -0.5],
            model,
            mean,
            std,
            [Seed(0, 'a', 0.0, atoms)],
            LatentInversionSettings(n_rounds=1, max_iter=5),
        )
        assert np.allclose(result['target_latent'], [0.25, -0.5])


class TestAnnotateAtoms:
    """Diagnostics carried on the decoded structure."""

    def test_diagnostics_survive_an_extxyz_round_trip(self, inversion_setup, tmp_path):
        pytest.importorskip('ase.io')
        from ase.io import read, write

        handle, model, atoms, mean, std = inversion_setup
        result = invert_one(
            handle,
            [0.1, 0.2],
            model,
            mean,
            std,
            [Seed(0, 'a', 0.0, atoms)],
            LatentInversionSettings(n_rounds=1, max_iter=5),
        )
        annotated = annotate_atoms(result)
        assert annotated.info['latent_decoded'] is True
        assert annotated.info['physical'] in (True, False)

        path = tmp_path / 'decoded.xyz'
        write(path, annotated, format='extxyz')
        loaded = read(path, format='extxyz')
        assert loaded.info['latent_err'] == pytest.approx(result['latent_err'])
        assert np.allclose(loaded.info['target_latent'], [0.1, 0.2])

    def test_does_not_mutate_the_result_structure(self, inversion_setup):
        handle, model, atoms, mean, std = inversion_setup
        result = invert_one(
            handle,
            [0.1, 0.2],
            model,
            mean,
            std,
            [Seed(0, 'a', 0.0, atoms)],
            LatentInversionSettings(n_rounds=1, max_iter=5),
        )
        annotate_atoms(result)
        assert 'latent_decoded' not in result['atoms'].info


class TestTraverseLatentPath:
    """Decoding a whole path, with progress and cancellation."""

    @staticmethod
    def _pool_and_targets(handle, model, mean, std):
        structures = [_cluster(6, seed=i) for i in range(4)]
        latent = np.array([_latent_of(handle, model, s, mean, std) for s in structures])
        pool = SeedPool.from_structures(latent, structures)
        targets = sample_path(latent[[0, -1]], 3)
        return pool, targets

    def test_yields_one_result_per_point(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, targets = self._pool_and_targets(handle, model, mean, std)
        results = list(
            traverse_latent_path(
                handle,
                targets,
                pool,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=2, max_iter=20, n_starts=1),
            )
        )
        assert len(results) == len(targets)
        assert [r['point_index'] for r in results] == [0, 1, 2]

    def test_each_result_is_annotated_and_carries_coverage(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, targets = self._pool_and_targets(handle, model, mean, std)
        results = list(
            traverse_latent_path(
                handle,
                targets,
                pool,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=1, max_iter=10, n_starts=1),
            )
        )
        for result in results:
            assert 'annotated_atoms' in result
            assert 'target_nn_dist' in result
            assert 'target_n_within' in result

    def test_chaining_offers_the_previous_result_as_a_seed(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, targets = self._pool_and_targets(handle, model, mean, std)
        results = list(
            traverse_latent_path(
                handle,
                targets,
                pool,
                model,
                mean,
                std,
                LatentInversionSettings(
                    n_rounds=1,
                    max_iter=10,
                    n_starts=1,
                    chain_from_previous=True,
                    good_enough_latent_err=1e9,
                ),
            )
        )
        # A chained seed is index -1; the first point has no predecessor.
        assert results[0]['seed_index'] != -1
        assert any(r['seed_index'] == -1 for r in results[1:])

    def test_chaining_can_be_disabled(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, targets = self._pool_and_targets(handle, model, mean, std)
        results = list(
            traverse_latent_path(
                handle,
                targets,
                pool,
                model,
                mean,
                std,
                LatentInversionSettings(
                    n_rounds=1, max_iter=10, n_starts=1, chain_from_previous=False
                ),
            )
        )
        assert all(r['seed_index'] != -1 for r in results)

    def test_cancellation_stops_the_generator(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, targets = self._pool_and_targets(handle, model, mean, std)
        produced = []

        def stop():
            return len(produced) >= 1

        generator = traverse_latent_path(
            handle,
            targets,
            pool,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=1, max_iter=10, n_starts=1),
            should_stop=stop,
        )
        for result in generator:
            produced.append(result)
        assert len(produced) == 1

    def test_progress_is_reported_per_point(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, targets = self._pool_and_targets(handle, model, mean, std)
        seen = []
        list(
            traverse_latent_path(
                handle,
                targets,
                pool,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=1, max_iter=10, n_starts=1),
                on_progress=lambda i, n, msg: seen.append((i, n)),
            )
        )
        assert seen == [(0, 3), (1, 3), (2, 3)]

    def test_a_single_target_is_accepted_as_a_bare_point(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        pool, _targets = self._pool_and_targets(handle, model, mean, std)
        results = list(
            traverse_latent_path(
                handle,
                pool.latent[0],
                pool,
                model,
                mean,
                std,
                LatentInversionSettings(n_rounds=1, max_iter=10, n_starts=1),
            )
        )
        assert len(results) == 1

    def test_an_empty_pool_reports_why(self, inversion_setup):
        handle, model, _atoms, mean, std = inversion_setup
        empty = SeedPool.from_structures(np.zeros((0, 2)), [])
        with pytest.raises(ValueError, match='No seed structures'):
            list(
                traverse_latent_path(
                    handle, np.array([[0.0, 0.0]]), empty, model, mean, std
                )
            )


class TestLatentInversionLive:
    """End-to-end inversion with the real MACE handle, skipped without mace."""

    def test_decodes_a_reachable_target_with_mace_descriptors(self):
        """The full composed map: MACE -> standardize -> encoder -> latent.

        The fake-handle tests cover the optimizer; this covers the seam between
        it and a real backend -- descriptor width, dtype, the leaf exposed by
        ``prepare``, and the per-round neighbour-list rebuild on a real graph.
        """
        pytest.importorskip('mace')
        from atlas.active_learning.backends import get_backend

        handle = get_backend('mace').differentiable_descriptors(
            'mace:mp-small', {'device': 'cpu', 'dtype': 'float64'}
        )
        model = _autoencoder(input_dim=handle.descriptor_dim, seed=3)
        atoms = _cluster(6)

        with torch.no_grad():
            descriptors = handle.descriptors(handle.prepare(atoms)).numpy()
        mean = descriptors.mean(axis=0)
        std = np.where(descriptors.std(axis=0) == 0, 1.0, descriptors.std(axis=0))

        target = _reachable_target(handle, model, mean, std, atoms, shift=0.1)
        start_err = float(
            np.linalg.norm(_latent_of(handle, model, atoms, mean, std) - target)
        )

        result = invert_to_latent(
            handle,
            target,
            atoms,
            model,
            mean,
            std,
            LatentInversionSettings(n_rounds=3, max_iter=20),
        )

        assert result['latent_err'] < start_err
        assert result['physical']
        assert result['min_ratio'] >= COVALENT_FRACTION
