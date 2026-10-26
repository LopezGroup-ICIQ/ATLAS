"""Main ATLAS GUI window with sidebar navigation and collapsible log panel.

Layout
------
* Left: ``QListWidget`` sidebar listing each workflow page.
* Centre: ``QStackedWidget`` holding one page per sidebar entry.
* Bottom: collapsible ``CollapsibleLogPanel`` with toggle bar and badge.
* Top: standard menu bar (File / Edit / Format / Help).
"""

from __future__ import annotations

import contextlib
import os
import time
from functools import partial

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QFont,
    QIcon,
    QKeySequence,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QProgressBar,
    QSplitter,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.pages.active_learning import ActiveLearningPage
from atlas.core.gui.pages.dft import DftPage
from atlas.core.gui.pages.dft_benchmark import DftBenchmarkPage
from atlas.core.gui.pages.init_db import InitDbPage
from atlas.core.gui.pages.logs import LogsPage
from atlas.core.gui.pages.overview import OverviewPage
from atlas.core.gui.pages.reports import ReportsPage
from atlas.core.gui.pages.settings import SettingsPage
from atlas.core.gui.project import Project
from atlas.core.gui.widgets.toml_editor import TomlHighlighter

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'config_schema.yaml'
)

PAGE_ICON_NAMES = {
    'overview': 'dashboard',
    'init_db': 'database',
    'dft': 'science',
    'dft_benchmark': 'tune',
    'al': 'sync',
    'reports': 'assessment',
    'settings': 'settings',
    'logs': 'terminal',
}


class MainWindow(QMainWindow):
    """Sidebar-navigated main window for the ATLAS GUI."""

    def __init__(
        self,
        project: Project,
        application_font: QFont | None = None,
        progress_callback=None,
        verbose: bool = False,
    ):
        super().__init__()
        self._progress = progress_callback or (lambda *_: None)
        self._verbose = verbose
        self._t0 = time.perf_counter()

        self.setGeometry(100, 100, 1600, 900)
        self.project = project
        self.setWindowTitle(f'ATLAS, {project.name}')
        self.setWindowIcon(self._app_icon())

        self._application_font = application_font or QFont()
        self._schema_data: dict = {}

        self._log_timing('Loading configuration schema...')
        self._progress(10, 'Loading configuration schema...')
        self._load_schema(SCHEMA_PATH)

        self._force_quit = False
        from atlas.core.gui.themes import saved_global_theme

        self._current_theme = saved_global_theme()
        self._update_matplotlib_style(self._current_theme)

        self._log_timing('Building interface...')
        self._progress(20, 'Building interface...')
        self._build_collapsible_log()
        self._build_central_widget()

        self._log_timing('Applying theme...')
        self._progress(25, 'Applying theme...')
        self._apply_app_theme(self._current_theme)

        self._log_timing('Loading pages...')
        self._progress(30, 'Loading pages...')
        self._build_pages()

        self._log_timing('Creating menus and shortcuts...')
        self._progress(85, 'Creating menus and shortcuts...')
        self._create_menu_bar()
        self._create_shortcuts()
        self._build_status_bar()
        self._build_tray_icon()

        self._sidebar_set_current(0)

        self._log_timing('Ready')
        self._progress(100, 'Ready')
        if self._verbose:
            elapsed = time.perf_counter() - self._t0
            print(f'[ATLAS] Project loaded in {elapsed:.2f}s')

    def _log_timing(self, stage: str) -> None:
        if not self._verbose:
            return
        elapsed = time.perf_counter() - self._t0
        print(f'[ATLAS] {elapsed:6.3f}s - {stage}')

    # ============================================================ schema

    def _load_schema(self, filepath: str) -> None:
        try:
            with open(filepath, encoding='utf-8') as f:
                self._schema_data = yaml.safe_load(f) or {}
        except Exception as exc:
            self._schema_data = {}
            self.statusBar().showMessage(f'Error loading schema: {exc}', 10000)

    # ========================================================== layout

    def _build_collapsible_log(self) -> None:
        from atlas.core.gui.widgets.collapsible_log_panel import CollapsibleLogPanel

        self._log_document = QTextDocument()
        self.log_panel = CollapsibleLogPanel(
            shared_doc=self._log_document,
            parent=self,
        )
        self.log_viewer = self.log_panel.viewer()
        self.log_panel.toggled.connect(self._on_log_panel_toggled)
        self.log_panel.unread_count_changed.connect(self._update_logs_sidebar_badge)

    # Number of pages in the main (top) sidebar list.
    _SIDEBAR_MAIN_COUNT = 6

    def _build_central_widget(self) -> None:
        v_splitter = QSplitter(Qt.Vertical)

        h_splitter = QSplitter(Qt.Horizontal)

        # -- composite sidebar widget --
        sidebar_container = QWidget()
        sidebar_container.setMaximumWidth(200)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        self.sidebar = QListWidget()
        sidebar_layout.addWidget(self.sidebar, 1)

        self._sidebar_sep = QWidget()
        self._sidebar_sep.setFixedHeight(9)
        self._sidebar_sep.setAttribute(Qt.WA_TranslucentBackground)
        self._sidebar_sep.setStyleSheet('background: transparent;')
        sep_layout = QHBoxLayout(self._sidebar_sep)
        sep_layout.setContentsMargins(20, 4, 20, 4)
        self._sidebar_sep_line = QFrame()
        self._sidebar_sep_line.setFrameShape(QFrame.NoFrame)
        self._sidebar_sep_line.setFixedHeight(1)
        self._sidebar_sep_line.setStyleSheet('background: palette(mid);')
        sep_layout.addWidget(self._sidebar_sep_line)
        sidebar_layout.addWidget(self._sidebar_sep)

        self._sidebar_bottom = QListWidget()
        self._sidebar_bottom.setFixedHeight(0)
        sidebar_layout.addWidget(self._sidebar_bottom, 0)

        h_splitter.addWidget(sidebar_container)

        self.stack = QStackedWidget()
        h_splitter.addWidget(self.stack)

        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setSizes([180, 1420])

        self.sidebar.currentRowChanged.connect(self._on_main_sidebar_changed)
        self._sidebar_bottom.currentRowChanged.connect(
            self._on_bottom_sidebar_changed,
        )

        v_splitter.addWidget(h_splitter)
        v_splitter.addWidget(self.log_panel)
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([870, 26])

        self.setCentralWidget(v_splitter)

    def _on_log_panel_toggled(self, expanded: bool) -> None:
        pass

    # -- sidebar selection helpers --

    def _sidebar_set_current(self, idx: int) -> None:
        """Select a page by global index across both sidebar lists."""
        mc = self._SIDEBAR_MAIN_COUNT
        if idx < mc:
            self._sidebar_bottom.blockSignals(True)
            self._sidebar_bottom.setCurrentRow(-1)
            self._sidebar_bottom.blockSignals(False)
            self.sidebar.setCurrentRow(idx)
        else:
            self.sidebar.blockSignals(True)
            self.sidebar.setCurrentRow(-1)
            self.sidebar.blockSignals(False)
            self._sidebar_bottom.setCurrentRow(idx - mc)

    def _sidebar_current_row(self) -> int:
        """Return global page index from whichever list has a selection."""
        row = self.sidebar.currentRow()
        if row >= 0:
            return row
        brow = self._sidebar_bottom.currentRow()
        if brow >= 0:
            return self._SIDEBAR_MAIN_COUNT + brow
        return -1

    def _on_main_sidebar_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self._sidebar_bottom.blockSignals(True)
        self._sidebar_bottom.setCurrentRow(-1)
        self._sidebar_bottom.blockSignals(False)
        self._on_sidebar_changed(idx)

    def _on_bottom_sidebar_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self.sidebar.blockSignals(True)
        self.sidebar.setCurrentRow(-1)
        self.sidebar.blockSignals(False)
        self._on_sidebar_changed(self._SIDEBAR_MAIN_COUNT + idx)

    def _update_logs_sidebar_badge(self, count: int) -> None:
        item = self._logs_sidebar_item
        if count > 0:
            item.setText(f'Logs ({min(count, 999)})')
        else:
            item.setText('Logs')

    def _build_pages(self) -> None:
        page_classes = (
            OverviewPage,
            InitDbPage,
            DftBenchmarkPage,
            DftPage,
            ActiveLearningPage,
            ReportsPage,
            SettingsPage,
            LogsPage,
        )
        page_names = (
            'Overview',
            'Initial DB',
            'DFT Benchmark',
            'DFT Labelling',
            'Active Learning',
            'Reports',
            'Settings',
            'Logs',
        )
        self.pages: list = []
        self._page_index_by_key: dict[str, int] = {}
        for idx, page_cls in enumerate(page_classes):
            pct = 30 + int(50 * idx / len(page_classes))
            self._log_timing(f'  Loading page: {page_names[idx]}...')
            self._progress(pct, f'Loading page: {page_names[idx]}...')
            kwargs = dict(
                project=self.project,
                schema_data=self._schema_data,
                application_font=self._application_font,
                log=self.log,
                navigate=self.navigate_to,
                notification=self._show_tray_notification,
            )
            if page_cls is LogsPage:
                kwargs['shared_doc'] = self._log_document
            page = page_cls(**kwargs)
            self.pages.append(page)
            self._page_index_by_key[page.NAVIGATION_KEY] = idx
            self.stack.addWidget(page)

            item = QListWidgetItem(page.DISPLAY_NAME)
            icon_name = PAGE_ICON_NAMES.get(page.NAVIGATION_KEY)
            if icon_name:
                item.setIcon(self._themed_sidebar_icon(icon_name))
            if idx < self._SIDEBAR_MAIN_COUNT:
                self.sidebar.addItem(item)
            else:
                self._sidebar_bottom.addItem(item)

            # Store a reference to the Logs sidebar item for the badge.
            if page.NAVIGATION_KEY == 'logs':
                self._logs_sidebar_item = item

            page.workflow_state_changed.connect(self._broadcast_workflow_refresh)
            page.process_running_changed.connect(self._on_process_running_changed)
            page.progress_changed.connect(self._on_page_progress)
            if page.config_panel is not None:
                page.config_panel.save_succeeded.connect(
                    lambda: self.statusBar().showMessage(
                        '✓ Configuration saved.',
                        3000,
                    )
                )

        # Size the bottom list to fit its items exactly.
        bcount = self._sidebar_bottom.count()
        if bcount:
            h = sum(self._sidebar_bottom.sizeHintForRow(i) for i in range(bcount))
            h += 2 * self._sidebar_bottom.frameWidth() + 12
            self._sidebar_bottom.setFixedHeight(h)

        # Settings page broadcasts theme changes; route them to every page.
        settings_page = self.pages[len(page_classes) - 2]
        settings_page.theme_changed.connect(self._apply_theme)
        settings_page.app_theme_changed.connect(self._apply_app_theme)

    def _broadcast_workflow_refresh(self) -> None:
        """Ask every page (including the sender) to recompute pipeline UI."""
        current_idx = self.stack.currentIndex()
        for i, page in enumerate(self.pages):
            if i == current_idx:
                page.on_shown()
            else:
                page._workflow_stale = True
        self._update_status_bar()

    def navigate_to(self, key: str) -> None:
        """Switch the active page by ``NAVIGATION_KEY``."""
        idx = self._page_index_by_key.get(key)
        if idx is not None:
            self._sidebar_set_current(idx)

    def _on_sidebar_changed(self, idx: int) -> None:
        prev_idx = self.stack.currentIndex()
        self.stack.setCurrentIndex(idx)
        if 0 <= idx < len(self.pages):
            page = self.pages[idx]
            if getattr(page, '_theme_stale', False):
                page._theme_stale = False
                self._repolish_page(page)
            page._workflow_stale = False
            t0 = time.perf_counter()
            page.on_shown()
            if self._verbose:
                dt = (time.perf_counter() - t0) * 1000
                prev_name = (
                    self.pages[prev_idx].DISPLAY_NAME
                    if 0 <= prev_idx < len(self.pages)
                    else '?'
                )
                print(f'[ATLAS] tab: {prev_name} -> {page.DISPLAY_NAME} ({dt:.0f}ms)')
            if page.NAVIGATION_KEY == 'logs':
                self.log_panel._unread_count = 0
                self.log_panel._badge.hide()
                self._update_logs_sidebar_badge(0)
        self._update_status_bar()

    def _repolish_page(self, page) -> None:
        """Re-polish a page that was hidden during a theme switch."""
        from PySide6.QtWidgets import QWidget

        style = page.style()
        for widget in [page, *page.findChildren(QWidget)]:
            style.unpolish(widget)
            style.polish(widget)
        page.update()

    # ====================================================== status bar

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._status_project = QLabel(f'  {self.project.name}')
        self._status_project.setStyleSheet('font-weight: bold; padding: 0 8px;')
        self._status_stage = QLabel()
        self._status_stage.setStyleSheet('padding: 0 8px;')

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(160)
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            'QProgressBar { border: 1px solid palette(mid); border-radius: 4px; }'
            'QProgressBar::chunk { background: #4a90d9; border-radius: 3px; }'
        )
        self._progress_bar.hide()

        bar.addPermanentWidget(self._progress_bar)
        bar.addPermanentWidget(self._status_project)
        bar.addPermanentWidget(self._status_stage)
        self._update_status_bar()

    def _on_process_running_changed(self, running: bool) -> None:
        if running:
            self._progress_bar.show()
        else:
            any_running = any(
                p.worker is not None and p.worker.isRunning() for p in self.pages
            )
            if not any_running:
                self._progress_bar.hide()

    def _on_page_progress(self, value: int, maximum: int, message: str) -> None:
        """Drive the shared status-bar progress bar from a page.

        ``maximum == 0`` switches it to indeterminate; otherwise it becomes a
        determinate bar for the duration of the task.
        """
        if maximum > 0:
            self._progress_bar.setRange(0, maximum)
            self._progress_bar.setValue(max(0, min(value, maximum)))
        else:
            self._progress_bar.setRange(0, 0)
        self._progress_bar.setToolTip(message)
        self._progress_bar.show()

    def _running_task_name(self) -> str | None:
        for p in self.pages:
            if p.worker is not None and p.worker.isRunning():
                return p.DISPLAY_NAME
        return None

    def _update_status_bar(self) -> None:
        try:
            state = self.project.workflow_state()
            next_key = state.get('next_recommended', '')
            stage_names = {
                'init_db': 'Initial DB',
                'dft': 'DFT Labelling',
                'al': 'Active Learning',
                'reports': 'Reports',
            }
            if next_key:
                name = stage_names.get(next_key, next_key)
                self._status_stage.setText(f'Next: {name}')
            else:
                self._status_stage.setText('All stages complete')
        except Exception:
            self._status_stage.setText('')

    # ======================================================= shortcuts

    def _create_shortcuts(self) -> None:
        save_shortcut = QAction('Save', self)
        save_shortcut.setShortcut(QKeySequence.Save)
        save_shortcut.triggered.connect(self._current_save_snapshot)
        self.addAction(save_shortcut)

        run_shortcut = QAction('Run', self)
        run_shortcut.setShortcut(QKeySequence('Ctrl+R'))
        run_shortcut.triggered.connect(self._current_run)
        self.addAction(run_shortcut)

        for i in range(min(9, len(self.pages))):
            action = QAction(f'Page {i + 1}', self)
            action.setShortcut(QKeySequence(f'Ctrl+{i + 1}'))
            action.triggered.connect(partial(self._sidebar_set_current, i))
            self.addAction(action)

    # ========================================================== menu bar

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('&File')
        load_action = QAction('&Import TOML…', self)
        load_action.triggered.connect(self._current_load_toml)
        file_menu.addAction(load_action)
        save_action = QAction('&Save Snapshot', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._current_save_snapshot)
        file_menu.addAction(save_action)
        export_action = QAction('&Export TOML…', self)
        export_action.triggered.connect(self._current_export_toml)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        close_action = QAction('&Close to Tray', self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        exit_action = QAction('E&xit', self)
        exit_action.triggered.connect(self._tray_quit)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu('&Edit')
        undo_action = QAction('&Undo', self)
        undo_action.setEnabled(False)
        edit_menu.addAction(undo_action)
        redo_action = QAction('&Redo', self)
        redo_action.setEnabled(False)
        edit_menu.addAction(redo_action)

        format_menu = menu_bar.addMenu('F&ormat')
        theme_menu = format_menu.addMenu('Syntax Highlighting Theme')
        theme_group = QActionGroup(self)
        for theme_name in TomlHighlighter.THEMES:
            action = QAction(theme_name, self, checkable=True)
            if theme_name == 'Default':
                action.setChecked(True)
            action.triggered.connect(partial(self._apply_theme, theme_name))
            theme_group.addAction(action)
            theme_menu.addAction(action)

        view_menu = menu_bar.addMenu('&View')
        toggle_log_action = QAction('Show/Hide Logs', self)
        toggle_log_action.setShortcut(QKeySequence('Ctrl+`'))
        toggle_log_action.triggered.connect(self.log_panel.toggle)
        view_menu.addAction(toggle_log_action)

    # =========================================================== helpers

    def _current_page(self) -> QWidget | None:
        idx = self._sidebar_current_row()
        if idx < 0 or idx >= len(self.pages):
            return None
        return self.pages[idx]

    def _current_load_toml(self) -> None:
        page = self._current_page()
        if page is not None and page.config_panel is not None:
            page.config_panel.load_toml_from_dialog()

    def _current_save_snapshot(self) -> None:
        page = self._current_page()
        if (
            page is not None
            and page.config_panel is not None
            and page.config_panel.save_to_project()
        ):
            self.statusBar().showMessage('Configuration saved.', 3000)

    def _current_export_toml(self) -> None:
        page = self._current_page()
        if page is not None and page.config_panel is not None:
            page.config_panel.save_to_file()

    def _current_run(self) -> None:
        page = self._current_page()
        if page is not None and hasattr(page, 'run'):
            page.run()

    def _apply_theme(self, theme_name: str) -> None:
        for page in self.pages:
            page.set_theme(theme_name)

    def _apply_app_theme(self, theme_name: str) -> None:
        from atlas.core.gui.themes import (
            apply_theme_to_app,
            theme_colors,
        )

        def _t(label: str, t0: float) -> float:
            now = time.perf_counter()
            if self._verbose:
                print(f'[ATLAS] theme: {label} {(now - t0) * 1000:.1f}ms')
            return now

        if self._verbose:
            print(
                f'[ATLAS] theme: Changing from '
                f'"{self._current_theme}" to "{theme_name}"'
            )

        t0 = time.perf_counter()
        self._current_theme = theme_name
        self.setUpdatesEnabled(False)
        try:
            apply_theme_to_app(theme_name, verbose=self._verbose)
        finally:
            self.setUpdatesEnabled(True)
        t0 = _t('apply_theme_to_app', t0)

        self._update_matplotlib_style(theme_name)
        t0 = _t('matplotlib rcParams', t0)

        self._refresh_sidebar_icons()
        t0 = _t('sidebar icons', t0)

        colors = theme_colors(theme_name)
        self.log_panel.set_theme(colors['fg'])
        t0 = _t('log panel', t0)

        current_idx = self.stack.currentIndex()
        for i, page in enumerate(getattr(self, 'pages', ())):
            page.set_app_theme(theme_name)
            page._theme_stale = i != current_idx
        _t('page.set_app_theme (all)', t0)

    @staticmethod
    def _update_matplotlib_style(theme_name: str) -> None:
        """Set matplotlib rcParams to match the current app theme."""
        import matplotlib as mpl

        from atlas.core.gui.themes import theme_colors

        c = theme_colors(theme_name)
        bg, fg = c['bg'], c['fg']
        surface = c['surface']
        border = c['border']
        axes_bg = c['axes_bg']

        mpl.rcParams.update(
            {
                'figure.facecolor': bg,
                'axes.facecolor': axes_bg,
                'axes.edgecolor': border,
                'axes.labelcolor': fg,
                'text.color': fg,
                'xtick.color': fg,
                'ytick.color': fg,
                'grid.color': border,
                'legend.facecolor': surface,
                'legend.edgecolor': border,
            }
        )

    def log(self, message: str) -> None:
        self.log_panel.append(message)

    # ================================================== task notification

    def _show_tray_notification(self, task_name: str, success: bool) -> None:
        if self._tray is None or not self._tray.isVisible():
            return

        if success:
            self._tray.showMessage(
                task_name,
                'Completed successfully.',
                QSystemTrayIcon.Information,
                3000,
            )
        else:
            self._tray.showMessage(
                f'{task_name}, Failed',
                'The process did not complete successfully.',
                QSystemTrayIcon.Warning,
                3000,
            )

    # ======================================================= system tray

    def _build_tray_icon(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return

        self._tray = QSystemTrayIcon(self)
        from atlas.core.gui.icons import tray_icon

        self._tray.setIcon(tray_icon())
        self._tray.setToolTip(f'ATLAS, {self.project.name}')

        menu = QMenu()
        show_action = menu.addAction('Show ATLAS')
        show_action.triggered.connect(self._tray_show)
        menu.addSeparator()
        quit_action = menu.addAction('Quit')
        quit_action.triggered.connect(self._tray_quit)
        self._tray.setContextMenu(menu)

        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _themed_sidebar_icon(self, name: str) -> QIcon:
        from atlas.core.gui.icons import themed_icon
        from atlas.core.gui.themes import theme_colors

        color = theme_colors(self._current_theme)['fg']
        return themed_icon(name, color, size=20)

    def _refresh_sidebar_icons(self) -> None:
        mc = self._SIDEBAR_MAIN_COUNT
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            if i < len(self.pages):
                icon_name = PAGE_ICON_NAMES.get(self.pages[i].NAVIGATION_KEY)
                if icon_name:
                    item.setIcon(self._themed_sidebar_icon(icon_name))
        for i in range(self._sidebar_bottom.count()):
            item = self._sidebar_bottom.item(i)
            page_idx = mc + i
            if page_idx < len(self.pages):
                icon_name = PAGE_ICON_NAMES.get(self.pages[page_idx].NAVIGATION_KEY)
                if icon_name:
                    item.setIcon(self._themed_sidebar_icon(icon_name))

    @staticmethod
    def _app_icon() -> QIcon:
        from atlas.core.gui.icons import app_icon

        return app_icon()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._tray_show()

    def _tray_show(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _tray_quit(self) -> None:
        self._force_quit = True
        self.close()
        from PySide6.QtWidgets import QApplication

        QApplication.instance().quit()

    # ================================================== setup wizard

    def _check_first_run(self) -> None:
        from atlas.core.gui.widgets.setup_wizard import (
            SetupWizard,
            needs_first_run_wizard,
        )

        if needs_first_run_wizard():
            wizard = SetupWizard(self)
            wizard.exec()
            self._broadcast_workflow_refresh()

    # ======================================================= close

    def closeEvent(self, event):
        if not self._force_quit and self._tray is not None and self._tray.isVisible():
            self.hide()
            task = self._running_task_name()
            if task:
                detail = f'{task} is running and will continue in the background.'
            else:
                detail = 'Still running in the background.'
            self._tray.showMessage(
                'ATLAS',
                detail + ' Right-click the tray icon to quit.',
                QSystemTrayIcon.Information,
                2000,
            )
            event.ignore()
            return

        if self._tray is not None:
            self._tray.hide()
        with contextlib.suppress(Exception):
            self.project.close()
        super().closeEvent(event)
