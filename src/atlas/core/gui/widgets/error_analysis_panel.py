"""Error analysis panel, DFT vs NN parity plots.

Evaluates a MACE model against a labelled database and renders
energy and force parity scatter plots with RMSE/MAE statistics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

COLORS = ['#6aa1f4', '#e50e3f', '#3ed04e']


class _Chart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 5), dpi=100, tight_layout=True)
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


class ErrorAnalysisPanel(QWidget):
    """DFT vs NN parity plots for energy and forces."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        # --- file pickers ---
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText('Path to MACE model (.model)')
        model_browse = QPushButton('Browse…')
        model_browse.clicked.connect(self._browse_model)

        self._db_edit = QLineEdit()
        self._db_edit.setPlaceholderText('Path to labelled database (.xyz / .xz)')
        db_browse = QPushButton('Browse…')
        db_browse.clicked.connect(self._browse_db)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Model:'))
        model_row.addWidget(self._model_edit, 1)
        model_row.addWidget(model_browse)

        db_row = QHBoxLayout()
        db_row.addWidget(QLabel('Database:'))
        db_row.addWidget(self._db_edit, 1)
        db_row.addWidget(db_browse)

        self._run_btn = QPushButton('Evaluate Model')
        self._run_btn.setFixedWidth(160)
        self._run_btn.clicked.connect(self._run_evaluation)

        self._export_btn = QPushButton('Export PNG…')
        self._export_btn.setFixedWidth(120)
        self._export_btn.clicked.connect(self._export_png)
        self._export_btn.setEnabled(False)

        self._status = QLabel()
        self._status.setStyleSheet('color: #555; padding: 0 8px;')

        action_row = QHBoxLayout()
        action_row.addWidget(self._run_btn)
        action_row.addWidget(self._status, 1)
        action_row.addWidget(self._export_btn)

        # --- stats cards ---
        self._cards = QWidget()
        cards_layout = QHBoxLayout(self._cards)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)
        self._card_rmse_e = _make_card('RMSE E (meV/atom)')
        self._card_mae_e = _make_card('MAE E (meV/atom)')
        self._card_rmse_f = _make_card('RMSE F (meV/A)')
        self._card_mae_f = _make_card('MAE F (meV/A)')
        self._card_n_structs = _make_card('Structures')
        for c in (
            self._card_rmse_e,
            self._card_mae_e,
            self._card_rmse_f,
            self._card_mae_f,
            self._card_n_structs,
        ):
            cards_layout.addWidget(c)
        cards_layout.addStretch()
        self._cards.setVisible(False)

        # --- charts ---
        self._chart_energy = _Chart()
        self._chart_force = _Chart()

        chart_splitter = QSplitter(Qt.Horizontal)
        chart_splitter.addWidget(self._chart_energy)
        chart_splitter.addWidget(self._chart_force)

        # --- placeholder ---
        self._placeholder = QLabel(
            'Select a MACE model file and a labelled database,\n'
            'then click "Evaluate Model" to generate parity plots.\n\n'
            'The evaluation compares NN predictions against DFT\n'
            'reference values for energy and forces.'
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            'color: #6c757d; padding: 40px; font-size: 13px;'
        )

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.addLayout(model_row)
        layout.addLayout(db_row)
        layout.addLayout(action_row)
        layout.addWidget(self._cards)
        layout.addWidget(chart_splitter, 1)
        layout.addWidget(self._placeholder)

        self._chart_splitter = chart_splitter
        self._chart_splitter.setVisible(False)

        self._try_auto_fill()

    # ---------------------------------------------------------- file pickers

    def _browse_model(self) -> None:
        start = str(self._project.dir)
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            'Select MACE model',
            start,
            'MACE model (*.model);;All files (*)',
        )
        if filepath:
            self._model_edit.setText(filepath)

    def _browse_db(self) -> None:
        start = str(self._project.dir / 'databases')
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            'Select labelled database',
            start,
            'ATLAS database (*.xyz *.xz);;All files (*)',
        )
        if filepath:
            self._db_edit.setText(filepath)

    def _try_auto_fill(self) -> None:
        db_path = self._project.init_db_path()
        if db_path.exists():
            self._db_edit.setText(str(db_path))
        model_candidates = list(self._project.dir.glob('**/*.model'))
        if model_candidates:
            newest = max(model_candidates, key=lambda p: p.stat().st_mtime)
            self._model_edit.setText(str(newest))

    # ---------------------------------------------------------- evaluation

    def _run_evaluation(self) -> None:
        model_path = self._model_edit.text().strip()
        db_path = self._db_edit.text().strip()

        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self, 'Missing file', 'Please select a valid model file.'
            )
            return
        if not db_path or not Path(db_path).exists():
            QMessageBox.warning(
                self, 'Missing file', 'Please select a valid database file.'
            )
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText('Evaluating…')
        self._status.setText('Running MACE evaluation…')

        self._worker = _EvalWorker(model_path, db_path)
        self._worker.finished_signal.connect(self._on_eval_done)
        self._worker.start()

    def _on_eval_done(self, result: dict) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText('Evaluate Model')

        if 'error' in result:
            self._status.setText(f'Error: {result["error"]}')
            return

        self._status.setText(f'Evaluated {result["n_structures"]} structures.')

        _set_card(self._card_rmse_e, f'{result["rmse_e"]:.2f}')
        _set_card(self._card_mae_e, f'{result["mae_e"]:.2f}')
        _set_card(self._card_rmse_f, f'{result["rmse_f"]:.1f}')
        _set_card(self._card_mae_f, f'{result["mae_f"]:.1f}')
        _set_card(self._card_n_structs, f'{result["n_structures"]:,}')
        self._cards.setVisible(True)

        self._draw_energy_parity(result)
        self._draw_force_parity(result)
        self._chart_splitter.setVisible(True)
        self._placeholder.setVisible(False)
        self._export_btn.setEnabled(True)

    def _draw_energy_parity(self, result: dict) -> None:
        fig = self._chart_energy.fig
        fig.clear()
        ax = fig.add_subplot(111)

        e_dft = result['e_dft']
        e_nn = result['e_nn']

        ax.scatter(e_dft, e_nn, s=8, alpha=0.5, color=COLORS[0], edgecolors='none')
        lims = [min(e_dft.min(), e_nn.min()), max(e_dft.max(), e_nn.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='#888', lw=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('DFT Energy [eV/atom]')
        ax.set_ylabel('NN Energy [eV/atom]')
        ax.set_title(
            f'Energy Parity  (RMSE: {result["rmse_e"]:.2f} meV/atom)',
            fontsize=10,
            fontweight='bold',
        )
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._chart_energy.redraw()

    def _draw_force_parity(self, result: dict) -> None:
        fig = self._chart_force.fig
        fig.clear()
        ax = fig.add_subplot(111)

        f_dft = result['f_dft']
        f_nn = result['f_nn']

        ax.scatter(f_dft, f_nn, s=4, alpha=0.3, color=COLORS[1], edgecolors='none')
        lims = [min(f_dft.min(), f_nn.min()), max(f_dft.max(), f_nn.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='#888', lw=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('DFT Force [eV/A]')
        ax.set_ylabel('NN Force [eV/A]')
        ax.set_title(
            f'Force Parity  (RMSE: {result["rmse_f"]:.1f} meV/A)',
            fontsize=10,
            fontweight='bold',
        )
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._chart_force.redraw()

    def _export_png(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            'Export Parity Plots',
            'error_analysis.png',
            'PNG Images (*.png);;PDF (*.pdf);;All files (*)',
        )
        if not filepath:
            return

        fig = Figure(figsize=(14, 6), dpi=150)
        fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])

        for i, src_chart in enumerate([self._chart_energy, self._chart_force]):
            ax = fig.add_subplot(1, 2, i + 1)
            src_ax = src_chart.fig.axes[0] if src_chart.fig.axes else None
            if src_ax is None:
                continue
            for coll in src_ax.collections:
                offsets = coll.get_offsets()
                ax.scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    s=coll.get_sizes()[0] if len(coll.get_sizes()) else 8,
                    alpha=0.5,
                    color=coll.get_facecolor()[0],
                    edgecolors='none',
                )
            for line in src_ax.get_lines():
                ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    color=line.get_color(),
                    ls=line.get_linestyle(),
                )
            ax.set_xlabel(src_ax.get_xlabel())
            ax.set_ylabel(src_ax.get_ylabel())
            ax.set_title(src_ax.get_title())
            ax.set_aspect('equal')

        fig.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')

    # ---------------------------------------------------------- refresh

    def refresh(self) -> None:
        self._try_auto_fill()


class _EvalWorker(QThread):
    """Run MACE evaluation off the main thread."""

    finished_signal = Signal(dict)

    def __init__(self, model_path: str, db_path: str, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._db_path = db_path

    def run(self):
        try:
            result = _evaluate_model(self._model_path, self._db_path)
            self.finished_signal.emit(result)
        except Exception as exc:
            self.finished_signal.emit({'error': str(exc)})


def _evaluate_model(model_path: str, db_path: str) -> dict:
    from ase.io import read as ase_read

    structures = ase_read(db_path, format='extxyz', index=':')
    if not structures:
        return {'error': 'No structures found in database.'}

    labelled = [
        s for s in structures if 'REF_energy' in s.info and 'REF_forces' in s.arrays
    ]
    if not labelled:
        return {
            'error': 'No DFT-labelled structures found (need REF_energy/REF_forces).'
        }

    try:
        from atlas.active_learning.backends import get_backend

        backend = get_backend('mace')
        calc = backend.create_calculator(model_path=model_path, device='cpu')
    except ImportError:
        return {
            'error': 'MACE is not installed. Install mace-torch to use error analysis.'
        }
    except Exception as exc:
        return {'error': f'Failed to load MACE model: {exc}'}

    e_dft_list, e_nn_list = [], []
    f_dft_all, f_nn_all = [], []

    for atoms in labelled:
        n = len(atoms)
        e_dft = atoms.info['REF_energy'] / n
        e_dft_list.append(e_dft)

        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        e_nn = atoms_copy.get_potential_energy() / n
        e_nn_list.append(e_nn)

        f_ref = atoms.arrays['REF_forces'].flatten()
        f_nn = atoms_copy.get_forces().flatten()
        f_dft_all.extend(f_ref.tolist())
        f_nn_all.extend(f_nn.tolist())

    e_dft = np.array(e_dft_list)
    e_nn = np.array(e_nn_list)
    f_dft = np.array(f_dft_all)
    f_nn = np.array(f_nn_all)

    e_diff_meV = (e_nn - e_dft) * 1000
    f_diff_meV = (f_nn - f_dft) * 1000

    return {
        'e_dft': e_dft,
        'e_nn': e_nn,
        'f_dft': f_dft,
        'f_nn': f_nn,
        'rmse_e': float(np.sqrt(np.mean(e_diff_meV**2))),
        'mae_e': float(np.mean(np.abs(e_diff_meV))),
        'rmse_f': float(np.sqrt(np.mean(f_diff_meV**2))),
        'mae_f': float(np.mean(np.abs(f_diff_meV))),
        'n_structures': len(labelled),
    }


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


def _set_card(card: QWidget, text: str) -> None:
    lbl = card.findChild(QLabel, 'card_value')
    if lbl:
        lbl.setText(text)
