"""Active learning monitoring panel.

Parses the AL process log file in real time to extract per-iteration
metrics (RMSE E/F, database sizes, interpolating/extrapolating frame
counts) and displays learning curves, database growth and domain-check
charts.  When the AL base workchain PK is known, per-step domain counts
and the run status are read from AiiDA instead, with the log as fallback.
Auto-refreshes via ``QTimer`` while a process is running; can also be
refreshed manually.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

COLORS = ['#6aa1f4', '#ffa6c8', '#3ed04e', '#e50e3f']
CHEM_ACC_E = 43.37  # meV/atom, chemical accuracy reference

# Per-step domain counts shown in the monitor chart: (key, label, color)
DOMAIN_SERIES = [
    ('interpolating_frames', 'Interpolating', COLORS[0]),
    ('extrapolating_frames', 'Extrapolating', COLORS[3]),
    ('out_of_domain_frames', 'Out-of-domain (union)', COLORS[2]),
]
DOMAIN_KEYS = (
    'interpolating_frames',
    'extrapolating_frames',
    'out_of_domain_frames',
    'in_domain_frames',
    'frames_after_filters',
    'total_frames',
    'submitted_to_dft',
)
# Written by SimpleActiveLearningWorkChain.send_calc_or_remove_structures
_DOMAIN_STATS_RE = re.compile(
    r'Extrapolation statistics \(AL step (\d+)\): '
    r'(\d+) interpolating .*?, (\d+) extrapolating .*?, '
    r'(\d+) out-of-domain .*?, (\d+) in-domain, '
    r'(\d+) after filters / (\d+) total frames\.'
    r'.*?Submitted to DFT: (\d+)'
)
_SUBMITTED_PK_RE = re.compile(r'\(PK:\s*(\d+)\)|Process PK\W+(\d+)')
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


# ------------------------------------------------------------------ parser


def parse_al_log(log_path: Path) -> dict:
    """Parse an ATLAS AL log file and return per-iteration stats.

    Returns a dict keyed by iteration number, each value being a dict
    with keys: ``it_idx``, ``mace_e``, ``mace_f``, ``train_db_size``,
    ``seed_gen_db_size``, ``finished``, ``max_iterations`` and, once the
    step has reported its extrapolation statistics, ``domain`` (a dict
    with the ``DOMAIN_KEYS`` counts).
    """
    if not log_path.exists():
        return {}

    text = log_path.read_text(encoding='utf-8', errors='replace')

    # Initial DB size
    ini_match = re.search(r'initial database containing\s+(\S+)', text)
    ini_db_size = (
        int(ini_match.group(1).replace("'", '').replace(',', '')) if ini_match else 0
    )

    stats: dict[int, dict] = {
        0: {
            'it_idx': 0,
            'mace_e': None,
            'mace_f': None,
            'train_db_size': ini_db_size,
            'seed_gen_db_size': ini_db_size,
            'finished': True,
            'max_iterations': None,
        },
    }

    # "Starting AL Loop iteration X/Y"
    it_indices: list[int] = []
    max_iter = None
    for m in re.finditer(r'Starting AL Loop iteration (\d+)/(\d+)', text):
        it_indices.append(int(m.group(1)))
        max_iter = int(m.group(2))

    # "Iteration X: seed_gen_db Y, training_db: Z entries"
    db_by_iter: dict[int, dict] = {}
    for m in re.finditer(
        r'Iteration (\d+): seed_gen_db (\S+), training_db: (\S+) entries',
        text,
    ):
        db_by_iter[int(m.group(1))] = {
            'seed_gen_db_size': int(m.group(2).replace(',', '')),
            'train_db_size': int(m.group(3).replace(',', '')),
        }

    # "Best model of current step ... RMSE E: X meV/at, RMSE F: Y meV/Å"
    mace_list: list[dict] = []
    for m in re.finditer(
        r'Best model of current step .* RMSE E: (\S+) meV/at, RMSE F: (\S+) meV',
        text,
    ):
        mace_list.append(
            {
                'mace_e': float(m.group(1)),
                'mace_f': float(m.group(2)),
            }
        )

    # "Extrapolation statistics (AL step X): A interpolating ..., B extrapolating
    # ..., C out-of-domain ..., D in-domain, E after filters / F total frames.
    # ... Submitted to DFT: G, ..."
    domain_by_iter: dict[int, dict] = {}
    for m in _DOMAIN_STATS_RE.finditer(text):
        counts = [int(g) for g in m.groups()[1:]]
        domain_by_iter[int(m.group(1))] = dict(zip(DOMAIN_KEYS, counts, strict=True))

    for i, it_num in enumerate(it_indices):
        entry: dict = {
            'it_idx': it_num,
            'mace_e': None,
            'mace_f': None,
            'train_db_size': None,
            'seed_gen_db_size': None,
            'finished': False,
            'max_iterations': max_iter,
        }
        if i < len(mace_list):
            entry['mace_e'] = mace_list[i]['mace_e']
            entry['mace_f'] = mace_list[i]['mace_f']
        if it_num in db_by_iter:
            entry['train_db_size'] = db_by_iter[it_num]['train_db_size']
            entry['seed_gen_db_size'] = db_by_iter[it_num]['seed_gen_db_size']
            entry['finished'] = True
        if it_num in domain_by_iter:
            entry['domain'] = domain_by_iter[it_num]
        stats[it_num] = entry

    if max_iter is not None:
        stats[0]['max_iterations'] = max_iter

    return stats


def parse_submitted_pk(line: str) -> int | None:
    """Return the workchain PK printed by ``atl_active_learning run``, if any."""
    m = _SUBMITTED_PK_RE.search(_ANSI_RE.sub('', line))
    if m is None:
        return None
    return int(m.group(1) or m.group(2))


def fetch_run_details(project: Project, run_id: int):
    """Return the AiiDA ``AlRunDetails`` of a run, or ``None``.

    ``None`` means the base workchain PK is unknown or AiiDA cannot be
    queried (not installed, no profile loaded, node missing).
    """
    run = project.get_al_run(run_id)
    pk = run.get('base_workchain_pk') if run else None
    if pk is None:
        return None
    try:
        from atlas.core.gui.project.aiida_sync import fetch_al_run_details

        return fetch_al_run_details(pk)
    except Exception:
        return None


def _stats_from_persisted(project: Project, run_id: int) -> dict:
    """Rebuild a parsed-stats dict from the project ``al_iterations`` table."""
    stats = {}
    for r in project.list_al_iterations(run_id):
        it = r['iteration']
        stats[it] = {
            'it_idx': it,
            'mace_e': r.get('rmse_e'),
            'mace_f': r.get('rmse_f'),
            'train_db_size': r.get('db_size'),
            'seed_gen_db_size': None,
            'finished': True,
            'max_iterations': None,
        }
    return stats


def gather_run_stats(
    project: Project,
    run_id: int,
    log_path: Path | None = None,
    details=None,
) -> dict:
    """Collect per-iteration stats of an AL run from all available sources.

    The workchain log (``details.log_path``) is preferred over *log_path*
    (the GUI log of the ``atl_active_learning`` command, which only holds
    the submission summary for daemon runs). If no log yields iterations,
    the persisted ``al_iterations`` rows are used. Per-step domain counts
    from AiiDA (*details*) take precedence over the log-parsed ones.
    """
    candidates = []
    if details is not None and details.log_path:
        candidates.append(Path(details.log_path))
    if log_path is not None:
        candidates.append(log_path)

    stats: dict = {}
    for path in candidates:
        if path.exists():
            stats = parse_al_log(path)
            if len(stats) > 1:
                break

    if len(stats) <= 1:
        persisted = _stats_from_persisted(project, run_id)
        if persisted:
            stats = persisted

    if details is not None:
        for step, step_stats in details.domain_stats.items():
            entry = stats.setdefault(
                step,
                {
                    'it_idx': step,
                    'mace_e': None,
                    'mace_f': None,
                    'train_db_size': None,
                    'seed_gen_db_size': None,
                    'finished': True,
                    'max_iterations': None,
                },
            )
            entry['domain'] = {key: step_stats.get(key) for key in DOMAIN_KEYS}

    return stats


def _stats_to_iteration_dicts(stats: dict) -> list[dict]:
    """Convert parsed stats into a list suitable for Project.upsert_al_iterations."""
    rows = []
    for it_num, s in stats.items():
        rows.append(
            {
                'iteration': it_num,
                'db_size': s.get('train_db_size'),
                'rmse_e': s.get('mace_e'),
                'rmse_f': s.get('mace_f'),
            }
        )
    return rows


# ------------------------------------------------------------------ chart


class _Chart(QWidget):
    """Single matplotlib chart canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 3), dpi=100, tight_layout=True)
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    @property
    def fig(self) -> Figure:
        return self._fig

    def redraw(self) -> None:
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()


# ------------------------------------------------------------ monitor panel


class AlMonitorPanel(QWidget):
    """Live AL monitoring panel showing learning curves and DB growth."""

    REFRESH_INTERVAL_MS = 10_000  # 10 seconds

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._log_path: Path | None = None
        self._run_id: int | None = None
        self._auto_refresh = False

        # --- run selector ---
        self._run_combo = QComboBox()
        self._run_combo.setMinimumWidth(200)
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        self._status_label = QLabel()
        self._status_label.setStyleSheet('color: #555; padding: 0 8px;')

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel('AL Run:'))
        top_bar.addWidget(self._run_combo)
        top_bar.addWidget(self._status_label, 1)
        top_bar.addWidget(refresh_btn)

        # --- summary cards ---
        self._iteration_label = self._make_card('Iteration', '—')
        self._rmse_e_label = self._make_card('RMSE E', '—')
        self._rmse_f_label = self._make_card('RMSE F', '—')
        self._db_size_label = self._make_card('Train DB', '—')
        self._out_of_domain_label = self._make_card('Out-of-domain', '—')

        cards = QHBoxLayout()
        cards.setSpacing(12)
        for card in (
            self._iteration_label,
            self._rmse_e_label,
            self._rmse_f_label,
            self._db_size_label,
            self._out_of_domain_label,
        ):
            cards.addWidget(card)
        cards.addStretch()

        # --- charts ---
        self._rmse_chart = _Chart()
        self._db_chart = _Chart()

        chart_splitter = QSplitter(Qt.Horizontal)
        chart_splitter.addWidget(self._rmse_chart)
        chart_splitter.addWidget(self._db_chart)
        chart_splitter.setStretchFactor(0, 1)
        chart_splitter.setStretchFactor(1, 1)

        self._domain_chart = _Chart()

        chart_grid = QSplitter(Qt.Vertical)
        chart_grid.addWidget(chart_splitter)
        chart_grid.addWidget(self._domain_chart)
        chart_grid.setStretchFactor(0, 1)
        chart_grid.setStretchFactor(1, 1)

        # --- empty state ---
        self._empty_label = QLabel(
            'No AL runs recorded yet.\n'
            'Start an active learning loop from the Config tab to see '
            'live monitoring here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('color: #6c757d; padding: 40px;')

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(cards)
        content_layout.addWidget(chart_grid, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._content, 1)
        layout.addWidget(self._empty_label)

        # --- auto-refresh timer ---
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._auto_refresh_tick)

    # ---------------------------------------------------------- public API

    def set_active_run(self, run_id: int, log_path: Path | None) -> None:
        """Point the monitor at a specific run and its log file."""
        self._run_id = run_id
        self._log_path = log_path
        self._refresh_run_combo()
        # Select the just-started run
        for i in range(self._run_combo.count()):
            if self._run_combo.itemData(i) == run_id:
                self._run_combo.setCurrentIndex(i)
                break
        self.refresh()

    def start_auto_refresh(self) -> None:
        self._auto_refresh = True
        if not self._timer.isActive():
            self._timer.start(self.REFRESH_INTERVAL_MS)

    def stop_auto_refresh(self) -> None:
        self._auto_refresh = False
        self._timer.stop()
        self.refresh()

    def refresh(self) -> None:
        self._refresh_run_combo()
        run_id = self._current_run_id()
        if run_id is None:
            self._content.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._content.setVisible(True)
        self._empty_label.setVisible(False)

        details = fetch_run_details(self._project, run_id)
        if details is not None:
            self._sync_run_status(run_id, details)

        stats = self._gather_stats(run_id, details)
        if not stats:
            self._status_label.setText('No iteration data available.')
            self._clear_cards()
            return

        self._persist_iterations(run_id, stats)
        self._update_cards(stats)
        if details is not None and details.terminated:
            self._status_label.setText(f'Run {details.status}.')
        self._draw_rmse_chart(stats)
        self._draw_db_chart(stats)
        self._draw_domain_chart(stats)

    # ---------------------------------------------------------- internals

    def _current_run_id(self) -> int | None:
        idx = self._run_combo.currentIndex()
        if idx < 0:
            return None
        return self._run_combo.itemData(idx)

    def _refresh_run_combo(self) -> None:
        current_id = self._current_run_id()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        runs = self._project.list_al_runs()
        for r in runs:
            rid = r['id']
            status = r.get('status', '')
            started = (r.get('started_at') or '')[:16]
            self._run_combo.addItem(
                f'#{rid}  {status}  ({started})',
                rid,
            )
        # Restore previous selection
        if current_id is not None:
            for i in range(self._run_combo.count()):
                if self._run_combo.itemData(i) == current_id:
                    self._run_combo.setCurrentIndex(i)
                    break
        self._run_combo.blockSignals(False)

    def _on_run_selected(self, _idx: int) -> None:
        run_id = self._current_run_id()
        if run_id is None:
            return
        # Try to find log file for this run
        self._log_path = self._find_log_for_run(run_id)
        self.refresh()

    def _find_log_for_run(self, run_id: int) -> Path | None:
        """Best-effort: find the log file for the given run."""
        # If we already have a log path and the run matches, keep it
        if self._log_path and self._run_id == run_id and self._log_path.exists():
            return self._log_path
        # Search logs directory for atl_active_learning logs
        log_dir = self._project.dir / 'logs'
        if not log_dir.is_dir():
            return None
        candidates = sorted(
            log_dir.glob('atl_active_learning.*.log'),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
        return None

    def _gather_stats(self, run_id: int, details=None) -> dict:
        """Try the log files first, fall back to SQLite; overlay AiiDA counts."""
        log_path = self._log_path or self._find_log_for_run(run_id)
        return gather_run_stats(self._project, run_id, log_path, details)

    def _sync_run_status(self, run_id: int, details) -> None:
        """Store the AiiDA status of the run; stop polling once it ends."""
        run = self._project.get_al_run(run_id)
        if run is not None and run.get('status') != details.status:
            with contextlib.suppress(Exception):
                self._project.update_al_run_status(
                    run_id,
                    details.status,
                    finished=details.terminated,
                )
            self._refresh_run_combo()
        if details.terminated and self._auto_refresh:
            self._auto_refresh = False
            self._timer.stop()

    def _persist_iterations(self, run_id: int, stats: dict) -> None:
        """Write parsed iteration data to the project DB for persistence."""
        dicts = _stats_to_iteration_dicts(stats)
        with contextlib.suppress(Exception):
            self._project.upsert_al_iterations(run_id, dicts)

    def _auto_refresh_tick(self) -> None:
        if self._auto_refresh:
            self.refresh()

    # ---------------------------------------------------------- cards

    @staticmethod
    def _make_card(title: str, value: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            'QWidget { background: #f8f9fa; border: 1px solid #dee2e6;'
            ' border-radius: 6px; padding: 8px 16px; }'
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            'font-size: 11px; color: #6c757d; border: none; background: transparent;'
        )
        value_lbl = QLabel(value)
        value_lbl.setObjectName('card_value')
        value_lbl.setStyleSheet(
            'font-size: 18px; font-weight: bold; color: #212529;'
            ' border: none; background: transparent;'
        )
        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        return card

    def _update_cards(self, stats: dict) -> None:
        finished = [s for s in stats.values() if s.get('finished')]
        if not finished:
            self._clear_cards()
            return

        latest = max(finished, key=lambda s: s['it_idx'])
        max_it = latest.get('max_iterations')

        it_text = str(latest['it_idx'])
        if max_it:
            it_text += f' / {max_it}'
        self._set_card(self._iteration_label, it_text)

        e = latest.get('mace_e')
        self._set_card(self._rmse_e_label, f'{e:.2f} meV' if e is not None else '—')

        f = latest.get('mace_f')
        self._set_card(self._rmse_f_label, f'{f:.1f} meV/A' if f is not None else '—')

        db = latest.get('train_db_size')
        self._set_card(self._db_size_label, f'{db:,}' if db is not None else '—')

        with_domain = [s for s in stats.values() if s.get('domain')]
        if with_domain:
            domain = max(with_domain, key=lambda s: s['it_idx'])['domain']
            n_out = domain.get('out_of_domain_frames') or 0
            n_total = domain.get('total_frames') or 0
            text = f'{n_out:,} / {n_total:,}'
            if n_total:
                text += f' ({n_out / n_total * 100:.1f}%)'
            self._set_card(self._out_of_domain_label, text)
        else:
            self._set_card(self._out_of_domain_label, '—')

        # Status text
        total_finished = len(
            [s for s in stats.values() if s['it_idx'] > 0 and s.get('finished')]
        )
        total_started = len([s for s in stats.values() if s['it_idx'] > 0])
        if total_started > total_finished:
            self._status_label.setText(f'Iteration {total_started} in progress…')
        elif max_it and latest['it_idx'] >= max_it:
            self._status_label.setText('All iterations complete.')
        else:
            self._status_label.setText(f'{total_finished} iteration(s) completed.')

    def _clear_cards(self) -> None:
        for card in (
            self._iteration_label,
            self._rmse_e_label,
            self._rmse_f_label,
            self._db_size_label,
            self._out_of_domain_label,
        ):
            self._set_card(card, '—')
        self._status_label.setText('')

    @staticmethod
    def _set_card(card: QWidget, text: str) -> None:
        lbl = card.findChild(QLabel, 'card_value')
        if lbl:
            lbl.setText(text)

    # ---------------------------------------------------------- charts

    def _draw_rmse_chart(self, stats: dict) -> None:
        fig = self._rmse_chart.fig
        fig.clear()

        iters_with_rmse = sorted(
            (
                s
                for s in stats.values()
                if s.get('mace_e') is not None and s['it_idx'] > 0
            ),
            key=lambda s: s['it_idx'],
        )
        if not iters_with_rmse:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'Waiting for first training result…',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._rmse_chart.redraw()
            return

        x = [s['it_idx'] for s in iters_with_rmse]
        e = [s['mace_e'] for s in iters_with_rmse]
        f = [s['mace_f'] for s in iters_with_rmse]

        ax_e = fig.add_subplot(111)
        ax_f = ax_e.twinx()

        ln1 = ax_e.plot(x, e, 'o-', color=COLORS[2], label='RMSE E', markersize=5)
        ln2 = ax_f.plot(x, f, 's-', color=COLORS[3], label='RMSE F', markersize=5)

        # Chemical accuracy line
        ax_e.axhline(y=CHEM_ACC_E, color='#28282833', ls='--', lw=1)
        ax_e.text(x[0], CHEM_ACC_E + 1, 'Chem. Acc.', fontsize=7, color='#888')

        ax_e.set_xlabel('AL Iteration')
        ax_e.set_ylabel('RMSE E [meV/atom]', color=COLORS[2])
        ax_f.set_ylabel('RMSE F [meV/A]', color=COLORS[3])
        ax_e.tick_params(axis='y', labelcolor=COLORS[2])
        ax_f.tick_params(axis='y', labelcolor=COLORS[3])

        lines = ln1 + ln2
        labels = [line.get_label() for line in lines]
        ax_e.legend(lines, labels, fontsize=8, loc='upper right')

        ax_e.set_title('Model Performance', fontsize=10, fontweight='bold')
        ax_e.spines['top'].set_visible(False)

        self._rmse_chart.redraw()

    def _draw_db_chart(self, stats: dict) -> None:
        fig = self._db_chart.fig
        fig.clear()

        iters_with_db = sorted(
            (s for s in stats.values() if s.get('train_db_size') is not None),
            key=lambda s: s['it_idx'],
        )
        if not iters_with_db:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'Waiting for iteration data…',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._db_chart.redraw()
            return

        x = np.array([s['it_idx'] for s in iters_with_db])
        train = [s['train_db_size'] for s in iters_with_db]
        seed = [s.get('seed_gen_db_size') for s in iters_with_db]
        has_seed = any(s is not None for s in seed)

        ax = fig.add_subplot(111)
        width = max(0.25, 0.8 / (2 if has_seed else 1))

        ax.bar(
            x,
            train,
            width=width,
            label='Train DB',
            color=COLORS[0],
            edgecolor='#282828',
            linewidth=0.8,
        )
        if has_seed:
            seed_clean = [s if s is not None else 0 for s in seed]
            ax.bar(
                x + width,
                seed_clean,
                width=width,
                label='Seed DB',
                color=COLORS[1],
                edgecolor='#282828',
                linewidth=0.8,
            )

        ax.set_xlabel('AL Iteration')
        ax.set_ylabel('Number of Structures')
        ax.set_title('Database Growth', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._db_chart.redraw()

    def _draw_domain_chart(self, stats: dict) -> None:
        fig = self._domain_chart.fig
        fig.clear()

        iters_with_domain = sorted(
            (s for s in stats.values() if s.get('domain')),
            key=lambda s: s['it_idx'],
        )
        if not iters_with_domain:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'Waiting for the first MD/domain-check results...',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._domain_chart.redraw()
            return

        x = np.arange(len(iters_with_domain))
        width = 0.8 / len(DOMAIN_SERIES)
        # Value labels only while they fit; the Outputs table has every count
        show_labels = len(iters_with_domain) <= 10

        ax = fig.add_subplot(111)
        for i, (key, label, color) in enumerate(DOMAIN_SERIES):
            values = [s['domain'].get(key) or 0 for s in iters_with_domain]
            bars = ax.bar(
                x + (i - (len(DOMAIN_SERIES) - 1) / 2) * width,
                values,
                width=width,
                label=label,
                color=color,
                edgecolor='#282828',
                linewidth=0.8,
            )
            if show_labels:
                ax.bar_label(bars, fontsize=7, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels([s['it_idx'] for s in iters_with_domain])
        ax.set_xlabel('AL Iteration')
        ax.set_ylabel('Number of MD Frames')
        ax.set_title(
            'Domain Checks per AL Step (committee / descriptor boundary)',
            fontsize=10,
            fontweight='bold',
        )
        # Headroom for the single-row legend above the tallest bar
        ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
        ax.legend(fontsize=8, ncols=len(DOMAIN_SERIES), loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._domain_chart.redraw()
