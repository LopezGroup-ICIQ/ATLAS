"""Database management panel, delete and export the initial database."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project

_BOUNDARY_PATTERNS = (
    'concave_hull*.png',
    'comp_plot*.png',
    'descriptors_concave_hull*.png',
)

# The latent map is only reproducible together with the autoencoder that built
# it and that model's standardization arrays, which are written beside it.
# Leaving them behind means the next boundary run silently reuses a stale
# mean/std and the new map does not match its own descriptors.
_BOUNDARY_FILES = (
    'latent_space.npz',
    'autoencoder_model.pth',
    'ae_mean_vals.npy',
    'ae_std_vals.npy',
)

_EXPORT_PATTERNS = ('export_db_filename_structures_*.extxyz',)


def _collect_database_files(db_path: Path, db_dir: Path) -> list[Path]:
    """Collect all files that should be deleted when the database is removed.

    Operates only within the databases/ directory to avoid affecting the
    project root.
    """
    files: list[Path] = [db_path]

    for pattern in _EXPORT_PATTERNS:
        for f in db_dir.glob(pattern):
            if f not in files:
                files.append(f)

    # Catch export files named after the database (e.g.
    # <db_name>_structures_<ts>.extxyz).
    db_stem = db_path.stem
    if db_stem.endswith('.xz'):
        db_stem = db_stem[:-3]
    for f in db_dir.glob(f'{db_stem}_structures_*.extxyz'):
        if f not in files:
            files.append(f)

    # Filtered-structures log from structure_filters.
    filtered = db_dir / 'filtered_structures.xyz'
    if filtered.is_file() and filtered not in files:
        files.append(filtered)

    return files


def _collect_boundary_files(directory: Path) -> list[Path]:
    """Return all boundary artifact files in a single directory."""
    files: list[Path] = []
    for name in _BOUNDARY_FILES:
        artifact = directory / name
        if artifact.is_file():
            files.append(artifact)
    for pattern in _BOUNDARY_PATTERNS:
        for f in directory.glob(pattern):
            files.append(f)
    return files


def _remove_boundary_files(directory: Path) -> None:
    """Remove latent-space boundary artifacts from a single directory."""
    for name in _BOUNDARY_FILES:
        artifact = directory / name
        if artifact.is_file():
            artifact.unlink()
    for pattern in _BOUNDARY_PATTERNS:
        for f in directory.glob(pattern):
            f.unlink()


class DbManagePanel(QWidget):
    """Delete or export the project's initial database."""

    database_deleted = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self._project = project

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(16)

        # ── Info ───────────────────────────────────────────────────
        info_group = QGroupBox('Database Info')
        info_layout = QVBoxLayout(info_group)
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_layout.addWidget(self._info_label)
        outer.addWidget(info_group)

        # ── Export ─────────────────────────────────────────────────
        export_group = QGroupBox('Export')
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(
            QLabel('Export the initial database as an extended XYZ file.')
        )
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton('Export as .extxyz')
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch()
        export_layout.addLayout(btn_row)
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 0)
        self._export_progress.setVisible(False)
        export_layout.addWidget(self._export_progress)
        outer.addWidget(export_group)

        # ── Delete ─────────────────────────────────────────────────
        delete_group = QGroupBox('Danger Zone')
        delete_group.setStyleSheet(
            'QGroupBox { border: 1px solid #ef5350; border-radius: 4px; }'
            'QGroupBox::title { color: #ef5350; }'
        )
        delete_layout = QVBoxLayout(delete_group)
        delete_layout.addWidget(
            QLabel('Permanently delete the generated database. This cannot be undone.')
        )
        btn_row2 = QHBoxLayout()
        self._delete_btn = QPushButton('Delete Database')
        self._delete_btn.setStyleSheet(
            'QPushButton { color: white; background-color: #ef5350;'
            ' border: none; padding: 6px 16px; border-radius: 4px; }'
            'QPushButton:hover { background-color: #e53935; }'
        )
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row2.addWidget(self._delete_btn)
        btn_row2.addStretch()
        delete_layout.addLayout(btn_row2)
        outer.addWidget(delete_group)

        outer.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        db_path = self._project.init_db_path()
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            counts = self._project.structure_counts()
            self._info_label.setText(
                f'<b>Path:</b> {db_path}<br>'
                f'<b>Size:</b> {size_mb:.1f} MB<br>'
                f'<b>Structures:</b> {counts["total"]}'
            )
            self._export_btn.setEnabled(True)
            self._delete_btn.setEnabled(True)
        else:
            self._info_label.setText('No database has been generated yet.')
            self._export_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)

    # ── Export ──────────────────────────────────────────────────────

    def _on_export(self) -> None:
        db_path = self._project.init_db_path()
        if not db_path.exists():
            QMessageBox.warning(self, 'Export', 'No database found to export.')
            return

        default_name = db_path.stem.replace('.xz', '') + '.extxyz'
        dest, _ = QFileDialog.getSaveFileName(
            self,
            'Export database as extxyz',
            str(db_path.parent / default_name),
            'Extended XYZ (*.extxyz);;All Files (*)',
        )
        if not dest:
            return

        self._export_btn.setEnabled(False)
        self._export_progress.setVisible(True)
        self._worker = _ExportWorker(db_path, dest)
        self._worker.finished_signal.connect(self._on_export_done)
        self._worker.start()

    def _on_export_done(self, error: str) -> None:
        self._export_btn.setEnabled(True)
        self._export_progress.setVisible(False)
        if error:
            QMessageBox.critical(self, 'Export Failed', error)
        else:
            QMessageBox.information(
                self, 'Export Complete', 'Database exported successfully.'
            )

    # ── Delete ─────────────────────────────────────────────────────

    def _on_delete(self) -> None:
        db_path = self._project.init_db_path()
        if not db_path.exists():
            return

        db_dir = db_path.parent
        project_dir = self._project.dir

        files = _collect_database_files(db_path, db_dir)
        boundary_db = _collect_boundary_files(db_dir)
        boundary_proj = _collect_boundary_files(project_dir)
        log_dir = db_dir / 'logs'

        all_files = (
            [f for f in files if f not in boundary_db] + boundary_db + boundary_proj
        )
        has_logs = log_dir.is_dir()

        rel_paths = []
        for f in all_files:
            try:
                rel_paths.append(str(f.relative_to(project_dir)))
            except ValueError:
                rel_paths.append(f.name)
        if has_logs:
            try:
                rel_paths.append(str(log_dir.relative_to(project_dir)))
            except ValueError:
                rel_paths.append(log_dir.name)

        file_list = '\n'.join(f'  - {p}' for p in rel_paths)
        answer = QMessageBox.warning(
            self,
            'Delete Database',
            f'This will permanently remove the following files'
            f' from the project directory:\n\n'
            f'{file_list}\n\n'
            'This action cannot be undone. Proceed?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            for f in all_files:
                f.unlink()
            self._project.clear_structures_index()
            if has_logs:
                shutil.rmtree(log_dir)
        except OSError as exc:
            QMessageBox.critical(
                self, 'Delete Failed', f'Could not delete database: {exc}'
            )
            return

        self.refresh()
        self.database_deleted.emit()


class _ExportWorker(QThread):
    finished_signal = Signal(str)

    def __init__(self, db_path: Path, dest: str, parent=None):
        super().__init__(parent)
        self._db_path = db_path
        self._dest = dest

    def run(self):
        try:
            from atlas.core.initial_db import InitialDatabase

            db = InitialDatabase.load_database(self._db_path)
            dest_path = Path(self._dest)
            db.export_db(
                out_format='extxyz',
                file_name=dest_path.stem,
                file_path=dest_path.parent,
            )
            self.finished_signal.emit('')
        except Exception as exc:
            self.finished_signal.emit(str(exc))
