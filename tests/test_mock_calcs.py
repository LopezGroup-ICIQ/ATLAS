"""Tests for the active-learning mock surrogates (atlas.active_learning.mock).

These exercise the plain-Python cores (no AiiDA profile required) plus the pure
``stage_is_mocked`` gate. The ``@calcfunction`` wrappers are thin and covered by
the end-to-end smoke run documented in the plan.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from atlas.active_learning.mock.mock_calcs import (
    _RMSE_E_FLOOR,
    _RMSE_F_FLOOR,
    _decayed_rmse,
    mock_dft_label_core,
    mock_md_process_core,
    mock_test_db_eval_core,
    mock_train_rmse_core,
    stage_is_mocked,
)


@pytest.mark.parametrize(
    'cfg, stage, expected',
    [
        ({}, 'md', False),
        (None, 'md', False),
        ({'enable': False, 'stages': ['md']}, 'md', False),
        ({'enable': True}, 'md', True),  # empty stages => all mocked
        ({'enable': True, 'stages': []}, 'dft', True),
        ({'enable': True, 'stages': ['md']}, 'md', True),
        ({'enable': True, 'stages': ['dft']}, 'md', False),
        ({'enable': True, 'stages': ['md', 'dft']}, 'training', False),
    ],
)
def test_stage_is_mocked(cfg, stage, expected):
    assert stage_is_mocked(cfg, stage) is expected


def _cu_frame():
    atoms = bulk('Cu', 'fcc', a=3.6)
    atoms.info = {'atl_id': 'x1', 'atl_db_index': 7, 'atl_struct_type': 'bulk'}
    return atoms


def test_mock_md_preserves_info_and_perturbs():
    seed = _cu_frame()
    orig = seed.get_positions().copy()
    out, stats = mock_md_process_core([seed], rattle_stdev=0.1, n_frames=2)

    assert len(out) == 2
    for frame in out:
        # Downstream keys must survive.
        assert frame.info['atl_id'] == 'x1'
        assert frame.info['atl_db_index'] == 7
        assert frame.info['atl_struct_type'] == 'bulk'
        disp = np.linalg.norm(frame.get_positions() - orig, axis=1)
        assert disp.max() > 0  # actually perturbed
        assert disp.max() < 1.0  # but within a sensible range of the stdev

    # The mock marks every produced frame as extrapolating (out-of-domain),
    # matching the contract that all returned frames are sent to DFT.
    assert stats['total_frames'] == 2
    assert stats['frames_after_filters'] == 2
    assert stats['extrapolation_error_frames'] == 2
    assert stats['interpolation_error_frames'] == 0
    assert stats['out_of_domain_frames'] == 2
    assert stats['seed_atl_id'] == 'x1'
    assert stats['per_temperature'] == []


def test_mock_md_stats_empty_seed():
    # No seed frames -> empty output and a stats dict with zero counts and an
    # 'unknown' seed id (mirrors the real CalcJob's fallback).
    out, stats = mock_md_process_core([], rattle_stdev=0.1, n_frames=2)
    assert out == []
    assert stats['total_frames'] == 0
    assert stats['out_of_domain_frames'] == 0
    assert stats['seed_atl_id'] == 'unknown'


def test_mock_dft_emt_labels_cu():
    lab = mock_dft_label_core([_cu_frame()], 'emt')
    assert len(lab) == 1
    assert np.isfinite(lab[0].info['REF_energy'])
    assert lab[0].arrays['REF_forces'].shape == (len(lab[0]), 3)
    assert np.all(np.isfinite(lab[0].arrays['REF_forces']))


def test_mock_dft_lj_fallback_for_unsupported_element():
    # Tungsten is not supported by ASE EMT; must fall back to LJ without raising.
    w = Atoms('W2', positions=[[0, 0, 0], [0, 0, 2.5]], cell=[10, 10, 10], pbc=True)
    lab = mock_dft_label_core([w], 'emt')
    assert np.isfinite(lab[0].info['REF_energy'])
    assert lab[0].arrays['REF_forces'].shape == (2, 3)


def test_mock_dft_lj_explicit():
    lab = mock_dft_label_core([_cu_frame()], 'lj')
    assert np.isfinite(lab[0].info['REF_energy'])


def test_decayed_rmse_monotonic_toward_floor():
    start, floor, decay = 90.0, 5.0, 0.7
    vals = [_decayed_rmse(it, start, floor, decay) for it in range(6)]
    assert vals[0] == pytest.approx(start)
    # Strictly decreasing, never below the floor, approaching it.
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    assert all(v >= floor for v in vals)
    assert _decayed_rmse(100, start, floor, decay) == pytest.approx(floor)


def test_mock_train_rmse_members_differ_and_decay():
    # Two committee members at the same iteration must differ (per-member jitter).
    e0, f0 = mock_train_rmse_core(0, model_index=0)
    e1, f1 = mock_train_rmse_core(0, model_index=1)
    assert e0 != e1 and f0 != f1
    # Realistic magnitudes at iter 0 (within jitter of the defaults).
    assert 80.0 < e0 < 100.0
    assert 245.0 < f0 < 305.0
    # Decays with iteration (compare same member index, no jitter cancellation).
    e_early, _ = mock_train_rmse_core(0, model_index=0)
    e_late, _ = mock_train_rmse_core(5, model_index=0)
    assert e_late < e_early


def test_mock_test_db_eval_core_decays_and_mae_below_rmse():
    rmse_e0, rmse_f0, mae_e0, mae_f0 = mock_test_db_eval_core(0)
    rmse_e2, rmse_f2, _, _ = mock_test_db_eval_core(2)
    assert all(np.isfinite(v) for v in (rmse_e0, rmse_f0, mae_e0, mae_f0))
    assert rmse_e2 < rmse_e0 and rmse_f2 < rmse_f0  # decays
    assert mae_e0 < rmse_e0 and mae_f0 < rmse_f0  # MAE below RMSE
    assert rmse_e0 >= _RMSE_E_FLOOR and rmse_f0 >= _RMSE_F_FLOOR


def test_mock_train_rmse_config_override():
    cfg = {
        'train_rmse_e_start': 50.0,
        'train_rmse_f_start': 150.0,
        'train_rmse_decay': 0.5,
    }
    e0, f0 = mock_train_rmse_core(0, model_index=0, cfg=cfg)
    assert 44.0 < e0 < 56.0  # ~50 within jitter
    assert 133.0 < f0 < 167.0  # ~150 within jitter
