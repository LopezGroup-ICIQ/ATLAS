"""Instant local surrogates for the remote active-learning calcjobs.

Each public ``@calcfunction`` mirrors the *outputs* of the real CalcJob it stands
in for, so the workchain's reader methods (``get_mlip_train_output``,
``get_descriptor_results``, ``send_calc_or_remove_structures``,
``return_seed_dft_and_model``) consume the resulting nodes unchanged. The heavy
lifting lives in plain-Python ``*_core`` helpers so it can be unit-tested without
an AiiDA profile.

Enabled via ``[debug.mock]`` in the settings TOML; see ``stage_is_mocked``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io import read as ase_read
from ase.io import write as ase_write

# 2**(1/6): ratio between the LJ equilibrium distance and sigma.
_LJ_EQUILIBRIUM_FACTOR = 2.0 ** (1.0 / 6.0)

# Canonical active-learning stages that can be mocked.
MOCK_STAGES = ('md', 'dft', 'training', 'descriptors', 'test_db')

# Defaults for the synthetic RMSE curves (meV/atom for E, meV/Å for F). Chosen to
# start in a realistic "early training" range and decay geometrically per iteration.
_DEFAULT_RMSE = {
    'train_rmse_e_start': 90.0,
    'train_rmse_f_start': 275.0,
    'train_rmse_decay': 0.7,
}
_RMSE_E_FLOOR = 5.0
_RMSE_F_FLOOR = 30.0


def _decayed_rmse(iteration: int, start: float, floor: float, decay: float) -> float:
    """Geometric decay from ``start`` toward ``floor`` over active-learning steps."""
    return floor + (start - floor) * (decay**iteration)


def stage_is_mocked(mock_cfg: dict | None, stage: str) -> bool:
    """Return whether ``stage`` should be mocked given the ``[debug.mock]`` config.

    ``mock_cfg`` is the ``debug.mock`` sub-dict of the settings TOML. Mock mode is
    active when ``enable`` is true; an empty/omitted ``stages`` list means *all*
    stages are mocked.
    """
    if not mock_cfg or not mock_cfg.get('enable', False):
        return False
    stages = mock_cfg.get('stages') or []
    return not stages or stage in stages


def _atoms_to_extxyz_singlefile(
    images: Atoms | list[Atoms], filename: str
) -> orm.SinglefileData:
    """Serialise ASE atoms to an extxyz ``SinglefileData`` (matching workchain I/O)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        ase_write(filename='-', format='extxyz', images=images)
    node = orm.SinglefileData(
        file=io.BytesIO(buffer.getvalue().encode()),
        filename=filename,
    )
    return node


def _read_extxyz_singlefile(node: orm.SinglefileData) -> list[Atoms]:
    """Read all frames from an extxyz ``SinglefileData``."""
    with node.as_path() as path:
        return ase_read(filename=path, format='extxyz', index=':')


# --------------------------------------------------------------------------- #
# MD: perturb the seed frame(s)                                               #
# --------------------------------------------------------------------------- #
def mock_md_process_core(
    seed_frames: list[Atoms], rattle_stdev: float, n_frames: int
) -> list[Atoms]:
    """Return random perturbations of the seed frame(s), preserving ``info``.

    Each output frame is a copy of a seed frame with positions rattled by a
    Gaussian of std ``rattle_stdev`` (Å). ``atoms.info``/``atoms.arrays`` are kept
    so downstream keys (``atl_id``, ``atl_db_index``, ``atl_struct_type``, ...)
    survive, mimicking a "structure that extrapolated during MD".
    """
    out: list[Atoms] = []
    for frame_idx, seed in enumerate(seed_frames):
        for copy_idx in range(max(1, n_frames)):
            perturbed = seed.copy()
            # Preserve metadata that ase copy() keeps on info but re-attach to be safe.
            perturbed.info = dict(seed.info)
            perturbed.rattle(stdev=rattle_stdev, seed=frame_idx * 1000 + copy_idx)
            out.append(perturbed)
    return out


@calcfunction
def mock_md_process(md_structure, rattle_stdev, n_frames):
    """Mock of ``ProcessMDSeedStructCalculation`` (``atl-process-md-seed-struct``).

    Outputs ``extrapolating_structures`` (extxyz ``SinglefileData``) exactly like
    the real MD calcjob, so the workchain treats the perturbed frames as the
    structures that left the training domain.
    """
    seed_frames = _read_extxyz_singlefile(md_structure)
    perturbed = mock_md_process_core(
        seed_frames=seed_frames,
        rattle_stdev=rattle_stdev.value,
        n_frames=n_frames.value,
    )
    return {
        'extrapolating_structures': _atoms_to_extxyz_singlefile(
            perturbed, 'extrapolating_structures.xyz'
        )
    }


# --------------------------------------------------------------------------- #
# DFT: cheap classical energies/forces                                         #
# --------------------------------------------------------------------------- #
def _lj_calculator_for(atoms: Atoms) -> LennardJones:
    """Build a Lennard-Jones calculator with sigma derived from the geometry.

    Sigma is set so the LJ equilibrium distance matches the structure's smallest
    interatomic separation, keeping energies/forces finite and non-degenerate.
    """
    if len(atoms) > 1:
        dists = atoms.get_all_distances(mic=any(atoms.pbc))
        nonzero = dists[dists > 1e-6]
        nn_dist = float(nonzero.min()) if nonzero.size else 2.5
    else:
        nn_dist = 2.5
    sigma = nn_dist / _LJ_EQUILIBRIUM_FACTOR
    return LennardJones(sigma=sigma, epsilon=1.0)


def mock_dft_label_core(structures: list[Atoms], potential: str) -> list[Atoms]:
    """Attach classical ``REF_energy``/``REF_forces`` to each structure.

    ``potential='emt'`` uses ASE EMT (falling back to Lennard-Jones per structure
    when EMT does not support an element); ``potential='lj'`` always uses LJ.
    """
    labeled: list[Atoms] = []
    for atoms in structures:
        work = atoms.copy()
        work.info = dict(atoms.info)
        calc = None
        if potential == 'emt':
            try:
                work.calc = EMT()
                energy = work.get_potential_energy()
                forces = work.get_forces()
                calc = 'emt'
            except Exception:  # noqa: BLE001 - EMT supports only a few elements
                calc = None
        if calc is None:
            work.calc = _lj_calculator_for(work)
            energy = work.get_potential_energy()
            forces = work.get_forces()

        work.info['REF_energy'] = float(energy)
        work.arrays['REF_forces'] = np.asarray(forces, dtype=float)
        work.calc = None
        labeled.append(work)
    return labeled


@calcfunction
def mock_dft_label(structures, potential):
    """Mock of the DFT labelling step (VASP / ``mace-eval``).

    Outputs ``labeled_structures`` (extxyz ``SinglefileData``) carrying
    ``REF_energy`` (info) and ``REF_forces`` (arrays), the DB reference convention.
    """
    frames = _read_extxyz_singlefile(structures)
    labeled = mock_dft_label_core(frames, potential.value)
    return {
        'labeled_structures': _atoms_to_extxyz_singlefile(
            labeled, 'labeled_structures.xyz'
        )
    }


# --------------------------------------------------------------------------- #
# Training: placeholder model + realistic decaying committee RMSE               #
# --------------------------------------------------------------------------- #
def mock_train_rmse_core(
    iteration: int, model_index: int, cfg: dict | None = None
) -> tuple[float, float]:
    """Synthetic per-member ``(rmse_e, rmse_f)`` that decay over iterations.

    Values start in a realistic early-training range and decay geometrically with
    ``iteration``. A small deterministic per-member jitter (seeded by
    ``(iteration, model_index)``) makes committee members differ, so the workchain's
    weighted-argmin picks a plausible best model (M0).
    """
    cfg = cfg or {}
    e_start = cfg.get('train_rmse_e_start', _DEFAULT_RMSE['train_rmse_e_start'])
    f_start = cfg.get('train_rmse_f_start', _DEFAULT_RMSE['train_rmse_f_start'])
    decay = cfg.get('train_rmse_decay', _DEFAULT_RMSE['train_rmse_decay'])

    base_e = _decayed_rmse(iteration, e_start, _RMSE_E_FLOOR, decay)
    base_f = _decayed_rmse(iteration, f_start, _RMSE_F_FLOOR, decay)

    # Deterministic ±10% jitter per committee member.
    rng = np.random.default_rng(seed=(iteration * 1000 + model_index))
    jitter_e, jitter_f = rng.uniform(0.9, 1.1, size=2)
    return float(base_e * jitter_e), float(base_f * jitter_f)


@calcfunction
def mock_train_model(model_name, iteration, model_index, cfg):
    """Mock of ``TrainMLIPCalculation`` (``atl-train-mlip``).

    Outputs ``model_file`` (a tiny placeholder ``SinglefileData`` — mocked MD
    ignores model content), plus ``m_rmse_e``/``m_rmse_f`` that differ per committee
    member and decay over iterations. Matches what ``get_mlip_train_output`` reads.
    """
    rmse_e, rmse_f = mock_train_rmse_core(
        iteration.value, model_index.value, cfg.get_dict()
    )
    placeholder = orm.SinglefileData(
        file=io.BytesIO(f'mock-model:{model_name.value}\n'.encode()),
        filename='mock_model.model',
    )
    return {
        'model_file': placeholder,
        'm_rmse_e': orm.Float(rmse_e),
        'm_rmse_f': orm.Float(rmse_f),
    }


# --------------------------------------------------------------------------- #
# Test-DB evaluation: synthetic decaying error metrics + placeholder plot       #
# --------------------------------------------------------------------------- #
def mock_test_db_eval_core(
    iteration: int, cfg: dict | None = None
) -> tuple[float, float, float, float]:
    """Synthetic ``(rmse_e, rmse_f, mae_e, mae_f)`` decaying over iterations.

    Uses the same decay curve as training (no per-member jitter — a single model is
    evaluated); MAE is set to ~0.75x the RMSE, as is typical.
    """
    cfg = cfg or {}
    e_start = cfg.get('train_rmse_e_start', _DEFAULT_RMSE['train_rmse_e_start'])
    f_start = cfg.get('train_rmse_f_start', _DEFAULT_RMSE['train_rmse_f_start'])
    decay = cfg.get('train_rmse_decay', _DEFAULT_RMSE['train_rmse_decay'])

    rmse_e = _decayed_rmse(iteration, e_start, _RMSE_E_FLOOR, decay)
    rmse_f = _decayed_rmse(iteration, f_start, _RMSE_F_FLOOR, decay)
    return rmse_e, rmse_f, 0.75 * rmse_e, 0.75 * rmse_f


@calcfunction
def mock_test_db_eval(iteration, cfg):
    """Mock of ``EvalTestDatabaseCalculation`` (``atl-eval-test``).

    Outputs ``rmse_e``/``rmse_f``/``mae_e``/``mae_f`` (``Float``, decaying over
    iterations) and ``eval_plot`` (a placeholder PNG ``SinglefileData``), matching
    every output ``get_test_db_results`` reads.
    """
    rmse_e, rmse_f, mae_e, mae_f = mock_test_db_eval_core(
        iteration.value, cfg.get_dict()
    )
    plot = orm.SinglefileData(
        file=io.BytesIO(b'mock-test-db-eval-plot\n'),
        filename='eval_plot.png',
    )
    return {
        'rmse_e': orm.Float(rmse_e),
        'rmse_f': orm.Float(rmse_f),
        'mae_e': orm.Float(mae_e),
        'mae_f': orm.Float(mae_f),
        'eval_plot': plot,
    }


# --------------------------------------------------------------------------- #
# Descriptors: trivial min/max arrays                                          #
# --------------------------------------------------------------------------- #
@calcfunction
def mock_descriptors(training_database_path):
    """Mock of ``GetDescriptorsCombinedCalculation`` (``atl-descriptors-combined``).

    Outputs ``descriptor_max``/``descriptor_min`` (``ArrayData``), matching what
    ``get_descriptor_results`` reads. No concave hull is produced, so downstream
    extrapolation checks fall back to their disabled behaviour.
    """
    return {
        'descriptor_max': orm.ArrayData(np.array([1.0, 1.0])),
        'descriptor_min': orm.ArrayData(np.array([0.0, 0.0])),
    }
