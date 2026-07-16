"""Active learning loop page (``atl_active_learning``)."""

from __future__ import annotations

import contextlib
import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.process.runner import (
    DetachedProcessMonitor,
    ProcessRunner,
    find_detached_process,
)
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.al_monitor_panel import AlMonitorPanel
from atlas.core.gui.widgets.al_outputs_panel import AlOutputsPanel
from atlas.core.gui.widgets.preflight_panel import Check, PreflightPanel
from atlas.core.gui.widgets.prereq_banner import PrereqBanner

CANONICAL_CONFIG = 'active_learning_settings.toml'

AL_TABS: list[tuple[str, list[str]]] = [
    ('General', ['active_learning']),
    ('Stopping & Test Set', ['stop_conditions', 'test_db']),
    ('Seed & Selection', ['al_seed', 'data_reduction']),
    ('Domain Checks', ['interpolation', 'extrapolation', 'safeguard']),
    ('MD Simulation', ['md']),
    ('Training & Committee', ['mlip', 'mace_train', 'committee_eval']),
    ('Descriptors & Code', ['descriptors', 'code']),
]


class ActiveLearningPage(WorkflowPage):
    """Configure and launch an AL loop; monitor it once running."""

    DISPLAY_NAME = 'Active Learning'
    NAVIGATION_KEY = 'al'

    def __init__(
        self,
        project: Project,
        schema_data,
        application_font,
        log,
        navigate,
        notification=None,
        parent=None,
    ):
        super().__init__(
            project,
            schema_data,
            application_font,
            log,
            navigate,
            notification,
            parent,
        )

        self.config_panel = self._build_config_panel()
        self.run_button = QPushButton(
            self._themed_icon('play_arrow'), ' Run Active Learning'
        )
        self.run_button.clicked.connect(self.run)
        self.cancel_button = self._make_cancel_button()
        self.config_panel.add_action_button(self.run_button)
        self.config_panel.add_action_button(self.cancel_button)

        self._preflight = PreflightPanel(
            'Pre-flight checks',
            self._build_preflight_checks(),
            focus_field=self.config_panel.focus_field,
        )
        self.config_panel.data_changed.connect(self._preflight.run_checks)

        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self._build_db_picker())
        config_layout.addWidget(self._preflight)
        config_layout.addWidget(self.config_panel, 1)

        self._runs_panel = _AlRunsPanel(project)
        self._monitor_panel = AlMonitorPanel(project)
        self._outputs_panel = AlOutputsPanel(project)

        self.tabs = QTabWidget()
        self.tabs.setObjectName('pageTab')
        self.tabs.addTab(config_tab, 'Config')
        self.tabs.addTab(self._monitor_panel, 'Monitor')
        self.tabs.addTab(self._outputs_panel, 'Outputs')
        self.tabs.addTab(self._runs_panel, 'Runs')
        self._mark_output_tabs(1)

        self._setup_banner = PrereqBanner(
            'ATLAS initial setup is incomplete.',
            'Go to Settings',
            lambda: self._navigate('settings'),
        )
        self._setup_banner.setVisible(False)

        self.prereq_banner = PrereqBanner(
            message='',
            action_label='Go to DFT Labelling',
            on_action=lambda: self._navigate('dft'),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._setup_banner)
        layout.addWidget(self.prereq_banner)
        layout.addWidget(self.tabs)

        self._pending_run_id: int | None = None
        self._update_prereq_banner()
        self._preflight.run_checks()

    def on_shown(self) -> None:
        self._update_prereq_banner()
        self._refresh_setup_banner()
        init_db = self.project.init_db_path()
        if init_db.exists() and not self.db_path_edit.text().strip():
            self.db_path_edit.setText(str(init_db.parent))
        self._try_reconnect_detached()
        self._refresh_suggestions()
        self._preflight.run_checks()
        self._runs_panel.refresh()
        self._monitor_panel.refresh()
        self._outputs_panel.refresh()

    def _try_reconnect_detached(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        logs_dir = self.project.dir / 'logs'
        info = find_detached_process(logs_dir, 'atl_active_learning')
        if info is None:
            return
        self._log(f'🔄 Found running AL process (PID {info["pid"]}), reconnecting…')
        self._set_running(self.run_button, self.cancel_button)
        self.worker = DetachedProcessMonitor(
            pid=info['pid'],
            log_file=info['log_file'],
            pid_file=info['pid_file'],
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

        if info['log_file'].exists():
            self._monitor_panel.set_active_run(
                self._pending_run_id,
                info['log_file'],
            )
            self._monitor_panel.start_auto_refresh()
            self.tabs.setCurrentWidget(self._monitor_panel)

    def _update_prereq_banner(self) -> None:
        state = self.project.workflow_state()
        init_status = state['stages']['init_db']['status']
        dft_status = state['stages']['dft']['status']

        if init_status == 'empty':
            self.prereq_banner.set_message(
                'Active learning needs an initial database first. '
                'Generate one before continuing.'
            )
            self.prereq_banner.action_button.setText('Go to Initial DB')
            with contextlib.suppress(RuntimeError):
                self.prereq_banner.action_button.clicked.disconnect()
            self.prereq_banner.action_button.clicked.connect(
                lambda: self._navigate('init_db')
            )
            self.prereq_banner.setVisible(True)
        elif dft_status == 'empty':
            self.prereq_banner.set_message(
                'Active learning needs DFT-labelled structures. '
                'Run DFT labelling on at least part of the database first.'
            )
            self.prereq_banner.action_button.setText('Go to DFT Labelling')
            with contextlib.suppress(RuntimeError):
                self.prereq_banner.action_button.clicked.disconnect()
            self.prereq_banner.action_button.clicked.connect(
                lambda: self._navigate('dft')
            )
            self.prereq_banner.setVisible(True)
        else:
            self.prereq_banner.setVisible(False)

    def _refresh_setup_banner(self) -> None:
        from atlas.core.gui.widgets.setup_wizard import check_setup_problems

        problems = check_setup_problems()
        if problems:
            self._setup_banner.set_message(
                'ATLAS initial setup is incomplete: '
                + '; '.join(problems)
                + '. Open Settings to run the setup wizard.'
            )
            self._setup_banner.setVisible(True)
        else:
            self._setup_banner.setVisible(False)

    def _build_config_panel(self):
        from atlas.core.gui.widgets.config_panel import ConfigPanel

        return ConfigPanel(
            schema_data=self._schema_data,
            section_key='active_learning',
            sub_section_tabs=AL_TABS,
            project=self.project,
            application_font=self._application_font,
        )

    def _refresh_suggestions(self) -> None:
        from atlas.core.gui.widgets.aiida_suggestions import (
            fetch_aiida_suggestions_async,
        )

        self.config_panel.set_suggestions_loading()
        fetch_aiida_suggestions_async(self.config_panel.populate_suggestions)

    def _build_preflight_checks(self) -> list[Check]:
        def _check_aiida() -> str | None:
            if not _aiida_available():
                return 'No AiiDA profile loaded, set AIIDA_PROFILE or load one first.'
            return None

        def _check_profile() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (
                (parsed or {})
                .get('active_learning', {})
                .get(
                    'aiida_profile',
                    '',
                )
            )
            if not val:
                return 'Missing aiida_profile in General tab.'
            try:
                from aiida.manage.configuration import get_config

                config = get_config()
                if val not in config.profile_names:
                    return f'Profile "{val}" not found in AiiDA config.'
            except Exception:
                pass
            return None

        def _check_run_name() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (
                (parsed or {})
                .get('active_learning', {})
                .get(
                    'run_name',
                    '',
                )
            )
            if not val:
                return 'Missing run_name in General tab.'
            return None

        def _check_results_dir() -> str | None:
            parsed, _ = self.config_panel.parsed_config()
            val = (
                (parsed or {})
                .get('active_learning', {})
                .get(
                    'results_dir',
                    '',
                )
            )
            if not val:
                return 'Missing results_dir in General tab.'
            return None

        def _check_db_folder() -> str | None:
            path = self.db_path_edit.text().strip()
            if not path:
                return 'Select an init DB folder in the picker above the config panel.'
            if not os.path.isdir(path):
                return f'Directory not found: {path}'
            return None

        return [
            Check('AiiDA profile', 'AiiDA is loaded', _check_aiida),
            Check(
                'AL profile name',
                'AiiDA profile name is set',
                _check_profile,
                requires_aiida=True,
                field_key='aiida_profile',
            ),
            Check(
                'Run name', 'AL run name is set', _check_run_name, field_key='run_name'
            ),
            Check(
                'Results directory',
                'Results directory is set',
                _check_results_dir,
                field_key='results_dir',
            ),
            Check(
                'Init DB folder', 'Init DB folder selected and exists', _check_db_folder
            ),
        ]

    def _build_db_picker(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Init DB folder:'))
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText(
            'Path to the folder containing the initial database'
        )
        init_db = self.project.init_db_path()
        if init_db.exists():
            self.db_path_edit.setText(str(init_db.parent))
        layout.addWidget(self.db_path_edit, 1)
        browse = QPushButton('Browse...')
        browse.clicked.connect(self._browse_db)
        layout.addWidget(browse)
        return bar

    def _browse_db(self) -> None:
        start = str(self.project.dir / 'databases')
        dirpath = QFileDialog.getExistingDirectory(
            self,
            'Select initial database folder',
            start,
        )
        if dirpath:
            self.db_path_edit.setText(dirpath)

    # ============================================================ run

    def run(self) -> None:
        db_folder = self.db_path_edit.text().strip()
        if not db_folder or not os.path.isdir(db_folder):
            self._log('❌ Provide an existing database folder before running AL.')
            return

        if not self.config_panel.save_to_project(label='run'):
            self._log('❌ Cannot run: configuration could not be saved.')
            return

        parsed, err = self.config_panel.parsed_config()
        if err or not parsed:
            self._log(f'❌ Cannot run: {err or "empty configuration"}')
            return

        problems = _preflight_check(parsed, db_folder)
        if problems:
            for p in problems:
                self._log(f'❌ {p}')
            return

        self._preflight.run_checks()
        preflight_warn = self._preflight.failing_summary()
        if preflight_warn:
            answer = QMessageBox.warning(
                self,
                'Pre-flight checks failed',
                'Some pre-flight checks are not passing:\n\n'
                f'{preflight_warn}\n\n'
                'Do you want to proceed anyway?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        if (
            QMessageBox.question(
                self,
                'Confirm AL Run',
                'Start the active learning loop with the current configuration?\n'
                'This may run for an extended period.',
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        active = self.project.active_config('active_learning')
        snapshot_id = active[0] if active else None

        try:
            run_id = self.project.record_al_submission(
                base_workchain_pk=None,
                config_snapshot_id=snapshot_id,
            )
            self._pending_run_id = run_id
            self._log(f'📒 Recorded AL run #{run_id} in the project.')
        except Exception as exc:
            self._log(f'⚠ Could not record AL run: {exc}')
            self._pending_run_id = None

        command = ['atl_active_learning', 'run', '-c', CANONICAL_CONFIG]
        self._set_running(self.run_button, self.cancel_button)
        log_file = self.project.log_path('atl_active_learning')
        self.worker = ProcessRunner(
            command,
            cwd=self.project.cwd(),
            log_file=log_file,
            detached=True,
        )
        self.worker.log_message.connect(self._log)
        self.worker.process_finished.connect(self._on_finished)
        self.worker.start()

        if self._pending_run_id is not None:
            self._monitor_panel.set_active_run(self._pending_run_id, log_file)
            self._monitor_panel.start_auto_refresh()
            self.tabs.setCurrentWidget(self._monitor_panel)

        self.workflow_state_changed.emit()

    def _on_finished(self, return_code: int) -> None:
        self._log(
            f'\n✅ atl_active_learning finished with exit code: {return_code}\n',
        )
        self._set_idle(self.run_button, self.cancel_button, 'Run Active Learning')
        self._notification('Active Learning', return_code == 0)
        self._monitor_panel.stop_auto_refresh()

        new_status = 'completed' if return_code == 0 else 'errored'
        if self._pending_run_id is not None:
            try:
                self.project.update_al_run_status(
                    self._pending_run_id,
                    new_status,
                    finished=True,
                )
            except Exception as exc:
                self._log(f'⚠ Could not update AL run status: {exc}')

        if return_code == 0:
            try:
                count = self.project.refresh_structures_index()
                if count:
                    self._log(f'📦 Re-indexed {count} structures.')
            except Exception as exc:
                self._log(f'⚠ Failed to re-index structures: {exc}')

        self._runs_panel.refresh()
        self._monitor_panel.refresh()
        self._outputs_panel.refresh()
        self.workflow_state_changed.emit()


# ================================================================ helpers


def _preflight_check(parsed: dict, db_folder: str) -> list[str]:
    """Return a list of problems; empty if ready to submit."""
    problems: list[str] = []

    al = parsed.get('active_learning', {})
    if not al.get('aiida_profile'):
        problems.append('Missing active_learning.aiida_profile.')
    if not al.get('run_name'):
        problems.append('Missing active_learning.run_name.')
    if not al.get('results_dir'):
        problems.append('Missing active_learning.results_dir.')

    configured_db = al.get('init_db_path', '')
    if not configured_db:
        problems.append(
            'Missing active_learning.init_db_path, set the path to the '
            'folder containing the initial database.'
        )
    elif not os.path.isdir(str(configured_db)):
        problems.append(
            f'active_learning.init_db_path "{configured_db}" does not exist.'
        )

    if not _aiida_available():
        problems.append(
            'AiiDA is not loaded or not available in this environment. '
            'Active learning requires a running AiiDA daemon.'
        )

    return problems


def _aiida_available() -> bool:
    try:
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        return profile is not None
    except Exception:
        return False


# ============================================================= runs panel


class _AlRunsPanel(QWidget):
    """Shows AL run records from the project database."""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

        self._summary = QLabel()
        self._summary.setStyleSheet('padding: 8px; color: #495057;')

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            [
                'Run #',
                'Status',
                'Started',
                'Finished',
                'Config Snapshot',
            ]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)

        self._sync_btn = QPushButton('Sync from AiiDA')
        self._sync_btn.setFixedWidth(130)
        self._sync_btn.setToolTip('Query AiiDA for current AL workchain statuses')
        self._sync_btn.clicked.connect(self._on_sync_aiida)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self._summary, stretch=1)
        top.addWidget(self._sync_btn)
        top.addWidget(refresh_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._table, stretch=1)

        self.refresh()

    def refresh(self) -> None:
        counts = self._project.al_run_counts()
        parts = []
        for status in ('submitted', 'running', 'completed', 'errored', 'stopped'):
            n = counts.get(status, 0)
            if n:
                parts.append(f'{n} {status}')
        total = sum(counts.values())
        if total == 0:
            self._summary.setText('No AL runs recorded yet.')
        else:
            self._summary.setText(f'AL runs: {", ".join(parts)} ({total} total)')

        rows = self._project.list_al_runs()
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._table.setItem(i, 0, _item(str(row.get('id', ''))))
            status = row.get('status', '')
            status_item = _item(status)
            if status == 'completed':
                status_item.setForeground(Qt.darkGreen)
            elif status in ('errored', 'stopped'):
                status_item.setForeground(Qt.red)
            elif status in ('submitted', 'running'):
                status_item.setForeground(Qt.darkYellow)
            self._table.setItem(i, 1, status_item)
            self._table.setItem(i, 2, _item(row.get('started_at', '')))
            self._table.setItem(
                i,
                3,
                _item(row.get('finished_at', '') or '—'),
            )
            snap = row.get('config_snapshot_id')
            self._table.setItem(i, 4, _item(str(snap) if snap else '—'))

    def _on_sync_aiida(self) -> None:
        if hasattr(self, '_sync_worker') and self._sync_worker.isRunning():
            return
        self._sync_btn.setEnabled(False)
        self._sync_btn.setText('Syncing…')
        self._sync_worker = _AlSyncWorker(self._project)
        self._sync_worker.finished_signal.connect(self._on_sync_done)
        self._sync_worker.start()

    def _on_sync_done(self, summary: str, errors: str) -> None:
        self._sync_btn.setEnabled(True)
        self._sync_btn.setText('Sync from AiiDA')
        if errors:
            QMessageBox.warning(self, 'AiiDA sync', errors)
        self.refresh()


class _AlSyncWorker(QThread):
    finished_signal = Signal(str, str)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project

    def run(self):
        try:
            from atlas.core.gui.project.aiida_sync import sync_al_runs

            result = sync_al_runs(self._project)
            errors = '\n'.join(result.errors) if result.errors else ''
            self.finished_signal.emit(result.summary, errors)
        except Exception as exc:
            self.finished_signal.emit('', str(exc))


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item
