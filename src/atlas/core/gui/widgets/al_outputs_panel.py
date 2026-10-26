"""Active learning outputs panel.

Post-hoc analysis of completed AL runs: 4-panel chart (DB evolution,
DB delta, RMSE E, RMSE F), summary cards, and an iteration table with
the per-step interpolating/extrapolating frame counts. Data comes from
AiiDA (when the base workchain PK is known), the parsed log files and/or
the project SQLite DB.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project
from atlas.core.gui.widgets.al_monitor_panel import (
    fetch_run_details,
    gather_run_stats,
)

COLORS = ['#6aa1f4', '#ffa6c8', '#3ed04e', '#e50e3f']
LINE_COLOR = '#28282855'
CHEM_ACC_E = 43.37  # meV/atom


class _Chart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 3.5), dpi=100, tight_layout=True)
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


class AlOutputsPanel(QWidget):
    """Post-hoc AL results viewer: charts + iteration table."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

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
        self._cards_widget = QWidget()
        cards_layout = QHBoxLayout(self._cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)
        self._card_iterations = self._make_card('Iterations')
        self._card_final_e = self._make_card('Final RMSE E')
        self._card_final_f = self._make_card('Final RMSE F')
        self._card_final_db = self._make_card('Final Train DB')
        self._card_db_growth = self._make_card('DB Growth')
        for c in (
            self._card_iterations,
            self._card_final_e,
            self._card_final_f,
            self._card_final_db,
            self._card_db_growth,
        ):
            cards_layout.addWidget(c)
        cards_layout.addStretch()

        # --- 4-panel chart grid ---
        self._chart_db_evol = _Chart()
        self._chart_db_delta = _Chart()
        self._chart_rmse_e = _Chart()
        self._chart_rmse_f = _Chart()

        chart_top = QSplitter(Qt.Horizontal)
        chart_top.addWidget(self._chart_db_evol)
        chart_top.addWidget(self._chart_db_delta)
        chart_top.setStretchFactor(0, 1)
        chart_top.setStretchFactor(1, 1)

        chart_bottom = QSplitter(Qt.Horizontal)
        chart_bottom.addWidget(self._chart_rmse_e)
        chart_bottom.addWidget(self._chart_rmse_f)
        chart_bottom.setStretchFactor(0, 1)
        chart_bottom.setStretchFactor(1, 1)

        chart_grid = QSplitter(Qt.Vertical)
        chart_grid.addWidget(chart_top)
        chart_grid.addWidget(chart_bottom)
        chart_grid.setStretchFactor(0, 1)
        chart_grid.setStretchFactor(1, 1)

        # --- iteration table ---
        self._table = QTableWidget()
        self._table.setColumnCount(11)
        self._table.setHorizontalHeaderLabels(
            [
                'Iteration',
                'Train DB',
                'Seed DB',
                'RMSE E (meV)',
                'RMSE F (meV/A)',
                'DB Delta',
                'Interpolating',
                'Extrapolating',
                'Out-of-domain',
                'To DFT',
                'Status',
            ]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)

        # --- split: charts (top) + table (bottom) ---
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(chart_grid)
        main_splitter.addWidget(self._table)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        # --- empty state ---
        self._empty_label = QLabel(
            'No AL run data available.\n'
            'Complete an active learning run to see outputs here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('color: #6c757d; padding: 40px;')

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._cards_widget)
        content_layout.addWidget(main_splitter, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._content, 1)
        layout.addWidget(self._empty_label)

    # ---------------------------------------------------------- public API

    def refresh(self) -> None:
        self._refresh_run_combo()
        run_id = self._current_run_id()
        if run_id is None:
            self._content.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._content.setVisible(True)
        self._empty_label.setVisible(False)

        stats = self._gather_stats(run_id)
        if not stats:
            self._status_label.setText('No iteration data available.')
            self._content.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._update_cards(stats)
        self._draw_db_evolution(stats)
        self._draw_db_delta(stats)
        self._draw_rmse_e(stats)
        self._draw_rmse_f(stats)
        self._populate_table(stats)

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
            self._run_combo.addItem(f'#{rid}  {status}  ({started})', rid)
        if current_id is not None:
            for i in range(self._run_combo.count()):
                if self._run_combo.itemData(i) == current_id:
                    self._run_combo.setCurrentIndex(i)
                    break
        self._run_combo.blockSignals(False)

    def _on_run_selected(self, _idx: int) -> None:
        self.refresh()

    def _gather_stats(self, run_id: int) -> dict:
        details = fetch_run_details(self._project, run_id)
        log_path = self._find_log_for_run(run_id)
        return gather_run_stats(self._project, run_id, log_path, details)

    def _find_log_for_run(self, run_id: int) -> Path | None:
        log_dir = self._project.dir / 'logs'
        if not log_dir.is_dir():
            return None
        candidates = sorted(
            log_dir.glob('atl_active_learning.*.log'),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None

    # ---------------------------------------------------------- cards

    @staticmethod
    def _make_card(title: str) -> QWidget:
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
        value_lbl = QLabel('—')
        value_lbl.setObjectName('card_value')
        value_lbl.setStyleSheet(
            'font-size: 18px; font-weight: bold; color: #212529;'
            ' border: none; background: transparent;'
        )
        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        return card

    @staticmethod
    def _set_card(card: QWidget, text: str) -> None:
        lbl = card.findChild(QLabel, 'card_value')
        if lbl:
            lbl.setText(text)

    def _update_cards(self, stats: dict) -> None:
        finished = sorted(
            (s for s in stats.values() if s.get('finished')),
            key=lambda s: s['it_idx'],
        )
        if not finished:
            for c in (
                self._card_iterations,
                self._card_final_e,
                self._card_final_f,
                self._card_final_db,
                self._card_db_growth,
            ):
                self._set_card(c, '—')
            return

        latest = finished[-1]
        first = finished[0]

        num_it = len([s for s in finished if s['it_idx'] > 0])
        max_it = latest.get('max_iterations')
        it_text = str(num_it)
        if max_it:
            it_text += f' / {max_it}'
        self._set_card(self._card_iterations, it_text)

        e = latest.get('mace_e')
        self._set_card(self._card_final_e, f'{e:.2f} meV' if e is not None else '—')

        f = latest.get('mace_f')
        self._set_card(self._card_final_f, f'{f:.1f} meV/A' if f is not None else '—')

        db = latest.get('train_db_size')
        self._set_card(self._card_final_db, f'{db:,}' if db is not None else '—')

        db_first = first.get('train_db_size')
        if db is not None and db_first is not None and db_first > 0:
            growth = db - db_first
            pct = growth / db_first * 100
            self._set_card(self._card_db_growth, f'+{growth:,} ({pct:.0f}%)')
        else:
            self._set_card(self._card_db_growth, '—')

        total_finished = num_it
        total_started = len([s for s in stats.values() if s['it_idx'] > 0])
        if total_started > total_finished:
            self._status_label.setText(f'Iteration {total_started} in progress...')
        elif max_it and latest['it_idx'] >= max_it:
            self._status_label.setText('All iterations complete.')
        else:
            self._status_label.setText(f'{total_finished} iteration(s) completed.')

    # ---------------------------------------------------------- charts

    def _extract_db_series(self, stats: dict):
        """Extract sorted iteration indices and DB sizes, filtering None values."""
        entries = sorted(
            (s for s in stats.values() if s.get('train_db_size') is not None),
            key=lambda s: s['it_idx'],
        )
        it_idx = [s['it_idx'] for s in entries]
        train = [s['train_db_size'] for s in entries]
        seed = [s.get('seed_gen_db_size') for s in entries]
        return it_idx, train, seed

    def _draw_db_evolution(self, stats: dict) -> None:
        fig = self._chart_db_evol.fig
        fig.clear()

        it_idx, train, seed = self._extract_db_series(stats)
        if not it_idx:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'No DB data',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._chart_db_evol.redraw()
            return

        ax = fig.add_subplot(111)
        x = np.arange(len(it_idx))
        has_seed = any(s is not None for s in seed)
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

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(it_idx)
        ax.set_xlabel('AL Iteration')
        ax.set_ylabel('Number of Structures')
        ax.set_title('Database Evolution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._chart_db_evol.redraw()

    def _draw_db_delta(self, stats: dict) -> None:
        fig = self._chart_db_delta.fig
        fig.clear()

        it_idx, train, seed = self._extract_db_series(stats)
        if len(it_idx) < 2:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'Need 2+ iterations',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._chart_db_delta.redraw()
            return

        train_delta = [0] + [train[i] - train[i - 1] for i in range(1, len(train))]
        has_seed = any(s is not None for s in seed)
        if has_seed:
            seed_clean = [s if s is not None else 0 for s in seed]
            seed_delta = [0] + [
                seed_clean[i] - seed_clean[i - 1] for i in range(1, len(seed_clean))
            ]

        ax = fig.add_subplot(111)
        x = np.arange(len(it_idx))
        width = max(0.25, 0.8 / (2 if has_seed else 1))

        ax.bar(
            x,
            train_delta,
            width=width,
            label='Train DB',
            color=COLORS[0],
            edgecolor='#282828',
            linewidth=0.8,
        )
        if has_seed:
            ax.bar(
                x + width,
                seed_delta,
                width=width,
                label='Seed DB',
                color=COLORS[1],
                edgecolor='#282828',
                linewidth=0.8,
            )

        ax.axhline(y=0, color=LINE_COLOR, linestyle='--')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(it_idx)
        ax.set_xlabel('AL Iteration')
        ax.set_ylabel('Δ Structures')
        ax.set_title('Per-Iteration Change', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._chart_db_delta.redraw()

    def _draw_rmse_e(self, stats: dict) -> None:
        fig = self._chart_rmse_e.fig
        fig.clear()

        entries = sorted(
            (
                s
                for s in stats.values()
                if s.get('mace_e') is not None and s['it_idx'] > 0
            ),
            key=lambda s: s['it_idx'],
        )
        if not entries:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'Waiting for training data...',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._chart_rmse_e.redraw()
            return

        x = [s['it_idx'] for s in entries]
        e = [s['mace_e'] for s in entries]

        ax = fig.add_subplot(111)
        ax.plot(x, e, 'o-', color=COLORS[2], markersize=5, label='RMSE E')
        ax.axhline(y=CHEM_ACC_E, color=LINE_COLOR, ls='--', lw=1)
        ax.text(x[0], CHEM_ACC_E + 1, 'Chem. Acc.', fontsize=7, color='#888')

        ax.set_xlabel('AL Iteration')
        ax.set_ylabel('RMSE E [meV/atom]')
        ax.set_title('Energy RMSE Evolution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._chart_rmse_e.redraw()

    def _draw_rmse_f(self, stats: dict) -> None:
        fig = self._chart_rmse_f.fig
        fig.clear()

        entries = sorted(
            (
                s
                for s in stats.values()
                if s.get('mace_f') is not None and s['it_idx'] > 0
            ),
            key=lambda s: s['it_idx'],
        )
        if not entries:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                'Waiting for training data...',
                ha='center',
                va='center',
                transform=ax.transAxes,
                color='#6c757d',
            )
            ax.set_axis_off()
            self._chart_rmse_f.redraw()
            return

        x = [s['it_idx'] for s in entries]
        f = [s['mace_f'] for s in entries]

        ax = fig.add_subplot(111)
        ax.plot(x, f, 's-', color=COLORS[3], markersize=5, label='RMSE F')

        ax.set_xlabel('AL Iteration')
        ax.set_ylabel('RMSE F [meV/A]')
        ax.set_title('Force RMSE Evolution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._chart_rmse_f.redraw()

    # ---------------------------------------------------------- table

    def _populate_table(self, stats: dict) -> None:
        entries = sorted(stats.values(), key=lambda s: s['it_idx'])

        self._table.setRowCount(len(entries))
        for i, s in enumerate(entries):
            it = s['it_idx']
            train = s.get('train_db_size')
            seed = s.get('seed_gen_db_size')
            e = s.get('mace_e')
            f = s.get('mace_f')

            # DB delta
            prev = [
                p
                for p in entries
                if p['it_idx'] < it and p.get('train_db_size') is not None
            ]
            if prev and train is not None:
                delta = train - prev[-1]['train_db_size']
                delta_str = f'+{delta}' if delta >= 0 else str(delta)
            else:
                delta_str = '—'

            status = 'Done' if s.get('finished') else 'In progress'

            self._table.setItem(i, 0, _item(str(it)))
            self._table.setItem(i, 1, _item(f'{train:,}' if train is not None else '—'))
            self._table.setItem(i, 2, _item(f'{seed:,}' if seed is not None else '—'))
            self._table.setItem(i, 3, _item(f'{e:.2f}' if e is not None else '—'))
            self._table.setItem(i, 4, _item(f'{f:.1f}' if f is not None else '—'))
            self._table.setItem(i, 5, _item(delta_str))

            domain = s.get('domain') or {}
            for col, key in enumerate(
                (
                    'interpolating_frames',
                    'extrapolating_frames',
                    'out_of_domain_frames',
                    'submitted_to_dft',
                ),
                start=6,
            ):
                value = domain.get(key)
                self._table.setItem(
                    i, col, _item(f'{value:,}' if value is not None else '—')
                )

            status_item = _item(status)
            if status == 'Done':
                status_item.setForeground(Qt.darkGreen)
            else:
                status_item.setForeground(Qt.darkYellow)
            self._table.setItem(i, 10, status_item)


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item
