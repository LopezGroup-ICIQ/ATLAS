"""Reusable TOML configuration editor for one schema section.

``ConfigPanel`` wraps one or more `SchemaForm` instances, a description
viewer, a live `TomlEditor`, and the standard Load / Save / Validate
buttons.  It is locked to a single top-level schema section (e.g.
``database_generation``) and is the building block of every workflow
page's "Config" tab.

When *sub_section_tabs* is provided, the single long form is split into
multiple inner tabs, each rendering only the specified sub-keys of the
schema section.  Data collection and population transparently span all
tabs so the resulting TOML is identical to the single-form case.
"""

from __future__ import annotations

import base64

import tomli
import tomli_w
from PySide6.QtCore import QBuffer, QIODevice, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from atlas.core.command_line import command_line_utils as cli_utils
from atlas.core.gui.app_params import ApplicationParameters
from atlas.core.gui.widgets.schema_form import SchemaForm
from atlas.core.gui.widgets.side_panel import SidePanel
from atlas.core.gui.widgets.toml_editor import TomlEditor
from atlas.core.gui.widgets.workflow_view import WorkflowStep, WorkflowView


class ConfigPanel(QWidget):
    """Editor for one section of the ATLAS configuration schema.

    Parameters
    ----------
    schema_data
        The full parsed ``config_schema.yaml`` dictionary.
    section_key
        Which top-level schema section this panel edits.
    sub_section_tabs
        Optional list of ``(tab_label, [sub_key, …])`` tuples.  When
        given, the single form is replaced by an inner ``QTabWidget``
        where each tab renders only the listed sub-keys.
    application_font
        Font passed to the embedded schema form.

    Supports drag-and-drop of ``.toml`` files to import configurations.
    """

    data_changed = Signal()
    validated = Signal(bool, list)
    save_succeeded = Signal()
    workflow_step_clicked = Signal(str)

    def __init__(
        self,
        schema_data: dict,
        section_key: str,
        sub_section_tabs: list[tuple[str, list[str]]] | None = None,
        project=None,
        application_font: QFont | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName('atlasPanel')
        self._section_key = section_key
        self._app_font = application_font or QFont()
        self._project = project
        self._sub_forms: list[SchemaForm] = []
        self._sub_section_tabs = sub_section_tabs or []
        self._workflow_active: bool = False

        outer = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # --- Left pane: form(s) ---
        if sub_section_tabs:
            self._form_tabs = QTabWidget()
            for tab_label, _sub_keys in sub_section_tabs:
                form = SchemaForm(field_font=self._app_font)
                form.field_focused.connect(self._on_field_focused)
                self._sub_forms.append(form)
                self._form_tabs.addTab(form, tab_label)
            self._form_tabs.currentChanged.connect(self._on_sub_tab_changed)
            splitter.addWidget(self._form_tabs)
            self.schema_form = self._sub_forms[0]
        else:
            self._form_tabs = None
            self.schema_form = SchemaForm(field_font=self._app_font)
            self.schema_form.field_focused.connect(self._on_field_focused)
            self._sub_forms.append(self.schema_form)
            splitter.addWidget(self.schema_form)

        # --- Middle pane: description ---
        description_group = QGroupBox('Description')
        description_layout = QVBoxLayout(description_group)
        self.description_viewer = QTextEdit()
        self.description_viewer.setReadOnly(True)
        desc_font = QFont()
        desc_font.setPointSize(desc_font.pointSize() + 1)
        self.description_viewer.setFont(desc_font)
        self.description_viewer.setPlaceholderText(
            'Click on any input field to see its description here.'
        )
        description_layout.addWidget(self.description_viewer)
        splitter.addWidget(description_group)

        # --- Right pane: switchable side panel ---
        self.side_panel = SidePanel()

        self._workflow_view = WorkflowView()
        self._workflow_view.step_clicked.connect(self.workflow_step_clicked)
        workflow_group = QGroupBox('Workflow Overview')
        workflow_group_layout = QVBoxLayout(workflow_group)
        workflow_group_layout.addWidget(self._workflow_view)
        self._workflow_view_idx = self.side_panel.add_view(
            'workflow', 'Workflow Overview', workflow_group, visible=False
        )

        toml_group = QGroupBox('Live TOML Preview')
        toml_layout = QVBoxLayout(toml_group)
        self.toml_editor = TomlEditor(
            monospace_families=ApplicationParameters.FONT_FAMILIES_MONOSPACE,
        )
        self.toml_editor.user_edited.connect(self._on_toml_user_edited)
        toml_layout.addWidget(self.toml_editor)
        self._toml_view_idx = self.side_panel.add_view(
            'code', 'Live TOML Preview', toml_group
        )

        self.side_panel.set_current_view(self._toml_view_idx)
        splitter.addWidget(self.side_panel)

        splitter.setSizes([600, 300, 400])

        # --- Buttons ---
        self._button_layout = QHBoxLayout()
        load_button = QPushButton('Import TOML…')
        load_button.setToolTip('Import an external TOML into the form (does not save).')
        load_button.clicked.connect(self.load_toml_from_dialog)
        self._button_layout.addWidget(load_button)

        if self._project is not None:
            save_button = QPushButton('Save Snapshot')
            save_button.setToolTip(
                'Record the current configuration as a snapshot in this project '
                'and mirror it to the canonical TOML on disk.'
            )
            save_button.clicked.connect(self.save_to_project)
            self._button_layout.addWidget(save_button)
            save_as_button = QPushButton('Export TOML…')
            save_as_button.setToolTip(
                'Export the current configuration to an arbitrary TOML file '
                '(does not affect the project).'
            )
            save_as_button.clicked.connect(self.save_to_file)
            self._button_layout.addWidget(save_as_button)
        else:
            save_button = QPushButton('Save TOML…')
            save_button.clicked.connect(self.save_to_file)
            self._button_layout.addWidget(save_button)

        validate_button = QPushButton('Validate')
        validate_button.clicked.connect(self.validate_current)
        self._button_layout.addWidget(validate_button)
        self._button_layout.addStretch()
        outer.addLayout(self._button_layout)

        # --- Initialise schemas ---
        for form in self._sub_forms:
            form.data_changed.connect(self._on_form_changed)
            form.set_schema(schema_data)

        if sub_section_tabs:
            for (_, sub_keys), form in zip(
                sub_section_tabs,
                self._sub_forms,
                strict=True,
            ):
                form.set_current_section(section_key, sub_keys=sub_keys)
        else:
            self.schema_form.set_current_section(section_key)

        if self._project is not None:
            self._load_active_snapshot()

        self.setAcceptDrops(True)

    # =========================================================== drag-drop

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.toml'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith('.toml'):
                try:
                    with open(path, 'rb') as f:
                        data = tomli.load(f)
                    if data:
                        self.load_data(data)
                        self._show_message(
                            'Import Successful',
                            f'Loaded configuration from {path}.',
                            QMessageBox.Information,
                        )
                    else:
                        self._show_message(
                            'Import Failed',
                            'The TOML file is empty.',
                            QMessageBox.Warning,
                        )
                except Exception as exc:
                    self._show_message(
                        'Import Failed',
                        f'Could not load TOML: {exc}',
                        QMessageBox.Critical,
                    )
                event.acceptProposedAction()
                return
        event.ignore()

    # =============================================================== api

    @property
    def section_key(self) -> str:
        return self._section_key

    def add_action_button(self, button: QPushButton) -> None:
        self._button_layout.addWidget(button)

    def focus_field(self, field_key: str) -> bool:
        """Switch to the tab containing *field_key* and scroll/focus it.

        *field_key* is a dotted schema path component (e.g. ``code_string``
        or ``queue.code_string``).  Returns True if the field was found.
        """
        needle = field_key.rsplit('.', 1)[-1]
        for tab_idx, form in enumerate(self._sub_forms):
            for w in form.iter_leaf_widgets():
                key = w.property('schema_key') or ''
                path = '.'.join(w.property('schema_path') or ())
                if key == needle or path.endswith(field_key):
                    if self._form_tabs is not None:
                        self._form_tabs.setCurrentIndex(tab_idx)
                    form.scroll_area.ensureWidgetVisible(w, 50, 100)
                    w.setFocus(Qt.ShortcutFocusReason)
                    return True
        return False

    def collect_data(self) -> dict:
        merged: dict = {}
        for form in self._sub_forms:
            merged.update(form.collect_data())
        return merged

    def current_toml(self) -> str:
        return self.toml_editor.plain_text()

    def parsed_config(self) -> tuple[dict | None, str | None]:
        text = self.toml_editor.plain_text()
        if not text.strip():
            return None, 'TOML preview is empty.'
        try:
            return tomli.loads(text), None
        except tomli.TOMLDecodeError as exc:
            return None, f'TOML parsing error: {exc}'

    def load_data(self, data: dict) -> None:
        for form in self._sub_forms:
            form.populate_from_data(data)
        self.refresh_toml_preview()

    def set_workflow_steps(
        self,
        steps: list[WorkflowStep],
        num_phases: int = 1,
        total_estimate: int | None = None,
    ) -> None:
        """Populate the workflow diagram and make it the default view."""
        self._workflow_view.set_steps(steps, num_phases, total_estimate)
        self.side_panel.show_view(self._workflow_view_idx)
        if not self._workflow_active:
            self._workflow_active = True
            self.side_panel.set_current_view(self._workflow_view_idx)

    def set_theme(self, theme_name: str) -> None:
        self.toml_editor.set_theme(theme_name)

    def set_app_theme(self, theme_name: str) -> None:
        self.side_panel.set_theme(theme_name)
        self._workflow_view.set_theme(theme_name)

    def set_suggestions_loading(self) -> None:
        """Show a loading placeholder on every sub-form's suggestion combos."""
        for form in self._sub_forms:
            form.set_suggestions_loading()

    def populate_suggestions(self, suggestions: dict[str, list[str]]) -> None:
        """Forward AiiDA suggestions to every sub-form's editable combos."""
        for form in self._sub_forms:
            form.populate_suggestions(suggestions)

    # ============================================================ slots

    def _on_form_changed(self) -> None:
        self.refresh_toml_preview()
        self.data_changed.emit()

    def _on_toml_user_edited(self) -> None:
        self.data_changed.emit()

    def _on_sub_tab_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sub_section_tabs):
            return
        _label, sub_keys = self._sub_section_tabs[index]
        if sub_keys:
            self.toml_editor.highlight_path([sub_keys[0]])
            self.description_viewer.clear()

    def _on_field_focused(
        self,
        description: str,
        value_type: str,
        schema_path,
        schema_key: str = '',
        mandatory: str = 'False',
        default_value: str = '',
    ):
        if description:
            is_required = mandatory == 'True'

            req_img = self._pill_img(
                'Required' if is_required else 'Optional',
                QColor('#dc3545') if is_required else QColor('#6c757d'),
            )
            type_img = self._pill_img(value_type, QColor('#1d6fa5'))

            default_line = ''
            if default_value:
                default_line = (
                    f'<p style="font-size:13px;">'
                    f'<b>Default:</b> <code>{default_value}</code></p>'
                )

            key_display = schema_key or (schema_path[-1] if schema_path else '')
            html = (
                f'<div style="font-family: sans-serif;">'
                f'<h3 style="margin:0 0 8px 0;">{key_display}</h3>'
                f'<p style="margin:0 0 10px 0;">'
                f'<img src="data:image/png;base64,{req_img}"> '
                f'<img src="data:image/png;base64,{type_img}"></p>'
                f'<p style="font-size:13px; line-height:1.5;">'
                f'{description}</p>'
                f'{default_line}'
                f'</div>'
            )
            self.description_viewer.setHtml(html)
        if schema_path:
            self.toml_editor.highlight_path(list(schema_path))

    @staticmethod
    def _pill_img(text: str, bg: QColor) -> str:
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance(text)
        pad_x, pad_y = 10, 4
        w = text_w + pad_x * 2
        h = fm.height() + pad_y * 2
        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(bg)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h / 2, h / 2)
        painter.setPen(QColor('#ffffff'))
        painter.setFont(font)
        painter.drawText(pad_x, pad_y + fm.ascent(), text)
        painter.end()
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        pixmap.save(buf, 'PNG')
        return base64.b64encode(buf.data().data()).decode('ascii')

    def refresh_toml_preview(self) -> None:
        data = self.collect_data()
        try:
            toml_string = tomli_w.dumps(data or {})
            self.toml_editor.set_text(toml_string)
        except Exception as exc:
            self.toml_editor.set_text(f'# Error generating TOML preview:\n# {exc}')

    # =========================================================== project

    def reload_active_snapshot(self) -> None:
        """Reload the form from the active snapshot.

        Used when another panel edits the same configuration (e.g. the
        latent-space generation pane updating ``[concave_hull]``), so the form
        never shows stale values that a later save would write back.
        """
        self._load_active_snapshot()

    def _load_active_snapshot(self) -> None:
        if self._project is None:
            return
        snapshot = self._project.active_config(self._section_key)
        if snapshot is None:
            return
        try:
            data = tomli.loads(snapshot[1])
        except tomli.TOMLDecodeError:
            return
        self.load_data(data)

    def save_to_project(self, label: str | None = None) -> bool:
        if self._project is None:
            return self.save_to_file()
        parsed, error = self.parsed_config()
        if error:
            self._show_message('Save Failed', error, QMessageBox.Critical)
            return False
        try:
            snapshot_id = self._project.save_config_snapshot(
                self._section_key,
                tomli_w.dumps(parsed),
                label=label,
            )
        except Exception as exc:
            self._show_message('Save Failed', str(exc), QMessageBox.Critical)
            return False
        active = self._project.active_config(self._section_key)
        if active and active[0] == snapshot_id:
            try:
                data = tomli.loads(active[1])
            except tomli.TOMLDecodeError:
                data = None
            if data is not None:
                for form in self._sub_forms:
                    form.populate_from_data(data)
                self.refresh_toml_preview()
        self.save_succeeded.emit()
        return True

    # ====================================================== file dialogs

    def load_toml_from_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load TOML File', '', 'TOML Files (*.toml)'
        )
        if not filepath:
            return
        try:
            with open(filepath, 'rb') as f:
                data = tomli.load(f)
            if not data:
                raise ValueError('TOML file is empty.')
            self.load_data(data)
        except Exception as exc:
            self._show_message(
                'Load Failed', f'Could not load TOML: {exc}', QMessageBox.Critical
            )

    def save_to_file(self, filepath: str | None = None) -> bool:
        parsed, error = self.parsed_config()
        if error:
            self._show_message('Save Failed', error, QMessageBox.Critical)
            return False

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                self, 'Save TOML Config', '', 'TOML Files (*.toml)'
            )
            if not filepath:
                return False

        with open(filepath, 'wb') as f:
            tomli_w.dump(parsed, f)
        return True

    # ========================================================= validate

    def validate_current(self) -> None:
        parsed, error = self.parsed_config()
        if error:
            self._report_validation(False, [error])
            return

        success, errors, _warnings = self._run_validator(parsed)
        self._report_validation(success, errors)

    def _run_validator(self, parsed: dict) -> tuple[bool, list, list]:
        try:
            any_errors, errors, warnings = cli_utils.validate_config_file(
                config_type=self._section_key,
                config_dict=parsed,
                allow_deprecated=False,
                run_mode='script',
            )
            return (not any_errors), list(errors or []), list(warnings or [])
        except Exception as exc:
            return False, [str(exc)], []

    def _report_validation(self, success: bool, errors: list) -> None:
        self.validated.emit(success, errors)
        if success:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle('Configuration Valid')
            msg.setText(f'Configuration for section "{self._section_key}" is valid.')
            msg.exec()
            return

        from atlas.core.gui.widgets.validation_dialog import ValidationDialog

        dlg = ValidationDialog(
            errors=errors or ['Validation failed for unknown reasons.'],
            section_key=self._section_key,
            focus_field=self.focus_field,
            parent=self,
        )
        dlg.exec()
