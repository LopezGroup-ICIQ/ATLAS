"""Per-step interpolating/extrapolating counts: workchain helpers and GUI side.

Covers the workchain's counter helpers (including the safeguard key layout),
the AL log parser for the "Extrapolation statistics (AL step N)" report line,
workchain PK capture from the ``atl_active_learning run`` output, and how the
monitor merges AiiDA, log and SQLite sources. No AiiDA profile is needed; the
live AiiDA path is exercised in ``test_mock_loop_integration.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('aiida')
pytest.importorskip('PySide6')

from atlas.core.gui.widgets.al_monitor_panel import (  # noqa: E402
    gather_run_stats,
    parse_al_log,
    parse_submitted_pk,
)
from atlas.workflows.simple_active_learning import (  # noqa: E402
    EXTRAP_COUNT_KEYS,
    calc_stats_to_counts,
    empty_extrap_counts,
    format_extrap_counts,
)


def _step_line(al_step: int, counts: dict, submitted: int) -> str:
    """Return the step report line exactly as the workchain writes it."""
    return (
        f'[10/25/26 12:00:00] [REPORT] - [1|SimpleActiveLearningWorkChain|'
        f'send_calc_or_remove_structures]: Extrapolation statistics '
        f'(AL step {al_step}): {format_extrap_counts(counts)} '
        f'Submitted to DFT: {submitted}, removed from seed DB: 0, '
        f'MD calculations: 3.'
    )


class _FakeProject:
    """Minimal stand-in for ``Project`` with persisted iterations."""

    def __init__(self, persisted=None):
        self._persisted = persisted or []

    def list_al_iterations(self, run_id):
        return self._persisted


class TestCounterHelpers:
    """Workchain counter helpers shared by step, base and safeguard."""

    def test_calc_stats_to_counts_maps_script_keys(self):
        counts = calc_stats_to_counts(
            {
                'total_frames': 10,
                'frames_after_filters': 8,
                'interpolation_error_frames': 2,
                'extrapolation_error_frames': 3,
                'out_of_domain_frames': 4,
            }
        )
        assert counts == {
            'total_frames': 10,
            'frames_after_filters': 8,
            'interpolating_frames': 2,
            'extrapolating_frames': 3,
            'out_of_domain_frames': 4,
            'in_domain_frames': 4,
        }

    def test_calc_stats_to_counts_safeguard_layout(self):
        # Safeguard MD has no committee: no interpolation key, None filters
        counts = calc_stats_to_counts(
            {
                'total_frames': 5,
                'frames_after_filters': None,
                'extrapolation_error_frames': 1,
                'out_of_domain_frames': 1,
            }
        )
        assert counts['interpolating_frames'] == 0
        assert counts['extrapolating_frames'] == 1
        assert counts['in_domain_frames'] == 0
        assert set(counts) == set(EXTRAP_COUNT_KEYS)

    def test_format_handles_zero_frames(self):
        assert 'n/a (0 frames)' in format_extrap_counts(empty_extrap_counts())


class TestParseAlLog:
    """Log parsing of the per-step extrapolation statistics line."""

    def test_domain_counts_attached_to_step(self, tmp_path):
        counts = {
            'total_frames': 40,
            'frames_after_filters': 30,
            'interpolating_frames': 4,
            'extrapolating_frames': 6,
            'out_of_domain_frames': 8,
            'in_domain_frames': 22,
        }
        log = tmp_path / 'atl_output.log'
        log.write_text(
            '\n'.join(
                [
                    'Starting AL Loop iteration 1/3...',
                    _step_line(1, counts, submitted=5),
                    'Iteration 1: seed_gen_db 12, training_db: 40 entries',
                    'Starting AL Loop iteration 2/3...',
                    'Safeguard extrapolation statistics (after AL step 1): '
                    + format_extrap_counts(counts),
                ]
            )
        )
        stats = parse_al_log(log)
        assert stats[1]['domain'] == {**counts, 'submitted_to_dft': 5}
        assert stats[1]['finished']
        assert 'domain' not in stats[2]


class TestParseSubmittedPk:
    """Workchain PK capture from the CLI submission output."""

    @pytest.mark.parametrize(
        ('line', 'pk'),
        [
            ('| - The active learning loop has been launched (PK: 1234) |', 1234),
            ('\x1b[1m(PK: \x1b[0m\x1b[1;34m77\x1b[0m)', 77),
            ('│ Process PK  │ 4321 │', 4321),
            ('Starting AL Loop iteration 1/3...', None),
        ],
    )
    def test_lines(self, line, pk):
        assert parse_submitted_pk(line) == pk


class TestGatherRunStats:
    """Merging of AiiDA, workchain log, CLI log and SQLite sources."""

    def _write_log(self, path, step_counts=None):
        lines = [
            'Starting AL Loop iteration 1/2...',
            'Iteration 1: seed_gen_db 9, training_db: 20 entries',
        ]
        if step_counts is not None:
            lines.insert(1, _step_line(1, step_counts, submitted=1))
        path.write_text('\n'.join(lines))
        return path

    def test_workchain_log_preferred_over_cli_log(self, tmp_path):
        cli_log = tmp_path / 'atl_active_learning.log'
        cli_log.write_text('The active learning loop has been launched (PK: 5)')
        wc_log = self._write_log(tmp_path / 'atl_output.log')
        details = SimpleNamespace(log_path=str(wc_log), domain_stats={})

        stats = gather_run_stats(_FakeProject(), 1, cli_log, details)
        assert stats[1]['train_db_size'] == 20

    def test_aiida_counts_override_log(self, tmp_path):
        log_counts = dict.fromkeys(EXTRAP_COUNT_KEYS, 1)
        wc_log = self._write_log(tmp_path / 'atl_output.log', log_counts)
        aiida_step = {
            'al_step': 1,
            **dict.fromkeys(EXTRAP_COUNT_KEYS, 7),
            'submitted_to_dft': 3,
            'per_calculation': {},
        }
        details = SimpleNamespace(log_path=str(wc_log), domain_stats={1: aiida_step})

        stats = gather_run_stats(_FakeProject(), 1, None, details)
        assert stats[1]['domain']['out_of_domain_frames'] == 7
        assert stats[1]['domain']['submitted_to_dft'] == 3
        assert 'per_calculation' not in stats[1]['domain']

    def test_sqlite_fallback_when_logs_have_no_iterations(self, tmp_path):
        cli_log = tmp_path / 'atl_active_learning.log'
        cli_log.write_text('The active learning loop has been launched (PK: 5)')
        project = _FakeProject(
            [{'iteration': 1, 'rmse_e': 2.0, 'rmse_f': 50.0, 'db_size': 30}]
        )

        stats = gather_run_stats(project, 1, cli_log, None)
        assert stats[1]['train_db_size'] == 30
        assert 'domain' not in stats[1]
