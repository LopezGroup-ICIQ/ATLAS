"""Schema-driven Qt form widget for the ATLAS GUI.

`SchemaForm` reads ATLAS' `config_schema.yaml` and builds a hierarchy of
`QFormLayout`-based widgets for one selected top-level section
(`database_generation`, `active_learning`, ...). It owns the section
selector, the scrollable form area, and the per-field schema metadata used
by the coordinator to populate the description and TOML highlight panels.

Signals
-------
schema_loaded(sections)
    Emitted after `load_schema` parses a YAML file successfully.
section_changed(section_key)
    Emitted when the user (or code) switches the active top-level section.
data_changed()
    Emitted whenever any field's value changes, including dynamic add/remove.
field_focused(description, value_type, schema_path, schema_key,
              mandatory, default_value)
    Emitted when a form widget gains focus; consumers can update a description
    panel or highlight the corresponding TOML line.
"""

from __future__ import annotations

from collections.abc import Iterator

import yaml
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.app_params import pretty_label
from atlas.core.gui.widgets.dict_widgets import IncarWidget, KspacingWidget


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores wheel events unless it has focus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores wheel events unless it has focus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class _CheckBoxGroup(QWidget):
    """Group of checkboxes for multi-select from a fixed set of choices."""

    def __init__(self, choices: list[str], parent=None):
        super().__init__(parent)
        self._checkboxes: list[QCheckBox] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for choice in choices:
            cb = QCheckBox(str(choice))
            self._checkboxes.append(cb)
            layout.addWidget(cb)

    def set_checked(self, items: list) -> None:
        checked_set = {str(item) for item in items}
        for cb in self._checkboxes:
            cb.setChecked(cb.text() in checked_set)

    def get_checked(self) -> list[str]:
        return [cb.text() for cb in self._checkboxes if cb.isChecked()]


class SchemaForm(QWidget):
    """Schema-driven form builder for one TOML configuration section."""

    schema_loaded = Signal(list)
    section_changed = Signal(str)
    data_changed = Signal()
    field_focused = Signal(str, str, tuple, str, str, str)

    def __init__(self, parent=None, field_font: QFont | None = None):
        super().__init__(parent)
        self.setObjectName('atlasPanel')
        self._field_font = field_font or QFont()

        self.schema_data: dict | None = None
        self.widgets_map: dict = {}
        self.dynamic_widgets: dict = {}
        self._current_section: str | None = None
        self._search_matches: list[QWidget] = []
        self._search_index: int = -1
        self._last_query: str = ''

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        search_row = QHBoxLayout()
        search_row.setContentsMargins(8, 4, 8, 4)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            'Find parameter (Ctrl+F), name, key, or description'
        )
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_text_changed)
        self.search_input.returnPressed.connect(self._search_step_forward)
        self.search_count_label = QLabel('')
        self.search_count_label.setMinimumWidth(60)
        self.search_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        search_row.addStretch(1)
        search_row.addWidget(self.search_input, 4)
        search_row.addWidget(self.search_count_label)
        search_row.addStretch(1)
        outer.addLayout(search_row)

        QShortcut(QKeySequence.Find, self, activated=self.search_input.setFocus)
        QShortcut(
            QKeySequence('Shift+Return'),
            self.search_input,
            activated=self._search_step_backward,
        )
        QShortcut(
            QKeySequence(Qt.Key_Escape),
            self.search_input,
            activated=self._clear_search,
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setObjectName('atlasPanel')
        self.scroll_area.setWidget(self.container)
        outer.addWidget(self.scroll_area, 1)

    # =============================================================== schema

    def load_schema(self, filepath) -> bool:
        """Parse the YAML schema at ``filepath``. Returns True on success."""
        with open(filepath, encoding='utf-8') as f:
            self.set_schema(yaml.safe_load(f))
        return True

    def set_schema(self, schema_data: dict) -> None:
        """Set the schema dict directly (e.g. when sharing across pages)."""
        self.schema_data = schema_data or {}
        self.schema_loaded.emit(self.available_sections())

    def available_sections(self) -> list[str]:
        return list(self.schema_data.keys()) if self.schema_data else []

    def current_section(self) -> str | None:
        return self._current_section

    def set_current_section(
        self,
        key: str,
        sub_keys: list[str] | None = None,
    ) -> None:
        """Build the form for ``key``.

        If *sub_keys* is given, only those top-level children of the
        section are rendered, useful for splitting a large section
        across multiple tabs.

        Emits ``section_changed`` and ``data_changed``.
        """
        if not self.schema_data or key not in self.schema_data:
            return

        self._current_section = key
        schema_section = self.schema_data[key]

        if sub_keys is not None:
            schema_section = {k: v for k, v in schema_section.items() if k in sub_keys}

        self._hidden_sections: dict[tuple[str, ...], dict] = {}
        self._scan_hidden_sections(schema_section, ())

        self._reset_container()
        self._build_widgets_recursively(
            self.container, schema_section, self.widgets_map, 0, []
        )
        self._wire_checkbox_section_toggles(self.widgets_map)
        self.section_changed.emit(key)
        self.data_changed.emit()

    def _reset_container(self) -> None:
        new_container = QWidget()
        new_container.setObjectName('atlasPanel')
        self.scroll_area.setWidget(new_container)
        self.container = new_container
        self.widgets_map = {}
        self.dynamic_widgets = {}
        self._section_toggle_fields: dict[str, list[str]] = {}

    def _wire_checkbox_section_toggles(self, widgets: dict) -> None:
        """Connect _CheckBoxGroup items to matching sibling QGroupBox sections.

        When a checkbox list (e.g. generate_type = [bulk, surface, cluster])
        has choices that match sibling section keys, checking/unchecking a
        choice enables/disables the corresponding section group box.
        """
        for _key, section in widgets.items():
            if not isinstance(section, dict) or '_group' not in section:
                continue
            checkbox_groups: list[tuple[str, _CheckBoxGroup]] = []
            sub_sections: dict[str, QGroupBox] = {}
            for sub_key, sub_val in section.items():
                if sub_key == '_group':
                    continue
                if isinstance(sub_val, _CheckBoxGroup):
                    checkbox_groups.append((sub_key, sub_val))
                elif isinstance(sub_val, dict) and '_group' in sub_val:
                    sub_sections[sub_key] = sub_val['_group']

            for _cb_key, cb_group in checkbox_groups:
                matched = {}
                for cb in cb_group.findChildren(QCheckBox):
                    label = cb.text()
                    if label in sub_sections:
                        matched[label] = sub_sections[label]
                if not matched:
                    continue
                for cb in cb_group.findChildren(QCheckBox):
                    if cb.text() in matched:
                        gb = matched[cb.text()]
                        gb.setEnabled(cb.isChecked())
                        self._update_optional_style(gb, cb.isChecked())
                        cb.toggled.connect(
                            lambda checked, g=gb: (
                                g.setEnabled(checked),
                                self._update_optional_style(g, checked),
                            )
                        )

    def _scan_hidden_sections(self, schema: dict, prefix: tuple[str, ...]) -> None:
        for k, v in schema.items():
            if not isinstance(v, dict):
                continue
            if v.get('gui_hidden'):
                self._hidden_sections[prefix + (k,)] = v
            elif 'description' in v and 'type' not in v:
                self._scan_hidden_sections(v, prefix + (k,))

    def clear(self) -> None:
        self._reset_container()

    def _clear_dynamic_items(self) -> None:
        for dyn_info in self.dynamic_widgets.values():
            if 'card_panel' in dyn_info:
                dyn_info['card_panel'].populate({})
                continue
            for item_widget in dyn_info['widgets'][:]:
                item_widget.deleteLater()
            dyn_info['widgets'].clear()

    def _build_widgets_recursively(
        self,
        parent_widget,
        config_level,
        widget_storage,
        depth,
        path_prefix=None,
    ):
        path_prefix = list(path_prefix or [])
        layout = QFormLayout(parent_widget)

        for key, item in config_level.items():
            if not isinstance(item, dict):
                continue
            if item.get('gui_hidden'):
                continue

            current_path = path_prefix + [key]

            if 'type' in item:
                label_text = item.get('name', pretty_label(key))
                is_mandatory = item.get('mandatory', False)
                if is_mandatory:
                    label = QLabel(f'{label_text} *')
                else:
                    label = QLabel(label_text)
                if is_mandatory:
                    font = QFont(self._field_font)
                    font.setBold(True)
                    label.setFont(font)
                label.setFont(QFont(self._field_font))

                widget = self._create_widget(
                    item,
                    key,
                    current_path,
                    widget_storage,
                )
                widget.installEventFilter(self)
                layout.addRow(label, widget)
                widget_storage[key] = widget
                widget.setProperty('schema_key', key)
                widget.setProperty('schema_path', tuple(current_path))
                widget.setProperty('label_text', label_text)

                if isinstance(widget, (KspacingWidget, IncarWidget)):
                    for child in widget.findChildren(QWidget):
                        child.setProperty('description', item.get('description', ''))
                        child.setProperty('value_type', item.get('type', 'N/A'))
                        child.setProperty('schema_path', tuple(current_path))
                        child.setProperty('schema_key', key)
                        child.setProperty(
                            'mandatory', str(item.get('mandatory', False))
                        )
                        child.setProperty('default_value', '')
                        child.installEventFilter(self)

                if isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self._emit_data_changed)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.connect(self._emit_data_changed)
                elif isinstance(widget, QComboBox):
                    if widget.isEditable():
                        widget.currentTextChanged.connect(self._emit_data_changed)
                    else:
                        widget.currentIndexChanged.connect(self._emit_data_changed)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self._emit_data_changed)
                elif isinstance(widget, _CheckBoxGroup):
                    for cb in widget.findChildren(QCheckBox):
                        cb.stateChanged.connect(self._emit_data_changed)

            elif 'description' in item:
                if item.get('dynamic_keys'):
                    self._handle_dynamic_section(
                        layout, key, item, widget_storage, depth, current_path
                    )
                else:
                    group_box = QGroupBox(item.get('name_pretty', pretty_label(key)))
                    font = group_box.font()
                    base_size = self._field_font.pointSize()
                    if depth == 0:
                        font.setPointSize(base_size + 2)
                        font.setBold(True)
                    else:
                        font.setPointSize(base_size)
                        font.setBold(False)
                    group_box.setFont(font)
                    group_box.setStyleSheet(
                        'QGroupBox { border-radius: 4px; margin-top: 8px;'
                        ' padding-top: 12px; }'
                        'QGroupBox::title { subcontrol-origin: margin;'
                        ' left: 8px; padding: 0 4px; }'
                    )

                    is_optional = not item.get('mandatory', True)
                    if is_optional:
                        group_box.setCheckable(True)
                        group_box.setChecked(item.get('enabled', True))
                        group_box.toggled.connect(self._emit_data_changed)
                        group_box.toggled.connect(
                            lambda checked, gb=group_box: self._update_optional_style(
                                gb, checked
                            )
                        )

                    toggle_fields = [
                        k
                        for k, v in item.items()
                        if isinstance(v, dict) and v.get('gui_section_toggle')
                    ]
                    if toggle_fields:
                        self._section_toggle_fields[key] = toggle_fields

                    widget_storage[key] = {'_group': group_box}
                    self._build_widgets_recursively(
                        group_box,
                        item,
                        widget_storage[key],
                        depth + 1,
                        current_path,
                    )
                    if is_optional:
                        self._update_optional_style(group_box, group_box.isChecked())
                    layout.addRow(group_box)

    def _handle_dynamic_section(
        self, parent_layout, key, item, widget_storage, depth, path_prefix
    ):
        template = item.get('schema_under_dynamic_keys', {})

        if key == 'phase':
            from atlas.core.gui.widgets.phase_card_panel import PhaseCardPanel

            panel = PhaseCardPanel(
                template,
                tuple(path_prefix),
                self._field_font,
                sibling_widgets=widget_storage,
                parent=self,
            )
            panel.data_changed.connect(self._emit_data_changed)
            self.dynamic_widgets[key] = {
                'card_panel': panel,
                'template': template,
                'widgets': [],
                'path': tuple(path_prefix),
            }
            widget_storage[key] = {'__dyn_ref__': key}

            container_group = QGroupBox(item.get('name_pretty', pretty_label(key)))
            font = container_group.font()
            base_size = self._field_font.pointSize()
            if depth == 0:
                font.setPointSize(base_size + 2)
                font.setBold(True)
            else:
                font.setPointSize(base_size)
                font.setBold(False)
            container_group.setFont(font)
            container_group.setStyleSheet(
                'QGroupBox { border-radius: 4px; margin-top: 8px;'
                ' padding-top: 12px; }'
                'QGroupBox::title { subcontrol-origin: margin;'
                ' left: 8px; padding: 0 4px; }'
            )
            group_layout = QVBoxLayout(container_group)
            group_layout.addWidget(panel)
            parent_layout.addRow(container_group)
            return

        container_group = QGroupBox(pretty_label(key))
        container_layout = QVBoxLayout(container_group)
        self.dynamic_widgets[key] = {
            'layout': container_layout,
            'template': template,
            'widgets': [],
            'path': tuple(path_prefix),
        }
        widget_storage[key] = {'__dyn_ref__': key}

        font = container_group.font()
        base_size = self._field_font.pointSize()
        if depth == 0:
            font.setPointSize(base_size + 2)
            font.setBold(True)
        else:
            font.setPointSize(base_size)
            font.setBold(False)
        container_group.setFont(font)
        container_group.setStyleSheet(
            'QGroupBox { border: 1px solid #d0d4db;'
            ' border-radius: 4px; margin-top: 8px;'
            ' padding-top: 12px; }'
            'QGroupBox::title { subcontrol-origin: margin;'
            ' left: 8px; padding: 0 4px; }'
        )

        add_button = QPushButton(f'Add New {pretty_label(key)}')
        add_button.setFont(self._field_font)
        add_button.clicked.connect(lambda: self.add_dynamic_item(key))
        container_layout.addWidget(add_button)

        parent_layout.addRow(container_group)

    def add_dynamic_item(self, key, name=None, data=None):
        if key not in self.dynamic_widgets:
            return

        dyn_info = self.dynamic_widgets[key]

        if name is None:
            existing = {
                w.findChild(QGroupBox).title()
                for w in dyn_info['widgets']
                if w.findChild(QGroupBox)
            }
            pretty = key.replace('_', ' ')
            idx = 1
            while f'{pretty}_{idx}' in existing:
                idx += 1
            name = f'{pretty}_{idx}'

        item_widget_container = QWidget()
        h_layout = QHBoxLayout(item_widget_container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        item_group = QGroupBox(name)
        item_group.setCheckable(True)
        item_group.setChecked(True)
        item_group.toggled.connect(self._emit_data_changed)
        item_group.toggled.connect(
            lambda checked, gb=item_group: self._update_optional_style(gb, checked)
        )

        widget_storage = {}
        self._build_widgets_recursively(
            item_group,
            dyn_info['template'],
            widget_storage,
            0,
            list(dyn_info['path']) + [name],
        )

        form_layout = item_group.layout()
        if isinstance(form_layout, QFormLayout):
            name_label = QLabel('Key (used in TOML) *')
            name_label.setStyleSheet('font-weight: bold;')
            name_label.setFont(QFont(self._field_font))
            name_edit = QLineEdit(name)
            name_edit.setFont(self._field_font)
            name_edit.setPlaceholderText(
                f'Enter a unique identifier for this {key.replace("_", " ")}'
            )
            name_edit.textChanged.connect(lambda text: item_group.setTitle(text))
            name_edit.textChanged.connect(self._emit_data_changed)
            form_layout.insertRow(0, name_label, name_edit)
            if data is None:
                name_edit.selectAll()
                name_edit.setFocus()

        # Populate sub-widgets from loaded data
        if data is not None:
            self._populate_widgets_recursively(widget_storage, data)

        remove_button = QPushButton('Remove')
        remove_button.setFont(self._field_font)
        remove_button.clicked.connect(
            lambda: (
                item_widget_container.deleteLater(),
                dyn_info['widgets'].remove(item_widget_container),
                self._emit_data_changed(),
            )
        )

        h_layout.addWidget(item_group, 1)
        h_layout.addWidget(remove_button)

        dyn_info['layout'].insertWidget(
            dyn_info['layout'].count() - 1, item_widget_container
        )
        dyn_info['widgets'].append(item_widget_container)
        item_group.setProperty('widget_storage', widget_storage)
        self._emit_data_changed()

    # ========================================================== data I/O

    _GUI_HIDDEN_OVERRIDES: dict[tuple[str, ...], dict] = {
        ('database', 'plot_db'): {'show': False},
        ('database', 'show_db_ase'): {'show': False},
        ('database', 'export'): {'file_path': 'databases'},
    }

    def collect_data(self) -> dict:
        """Walk the form and return a nested dict of current values."""
        if not self.widgets_map:
            return {}
        data = self._collect_data_recursively(self.widgets_map)
        for path, section in getattr(self, '_hidden_sections', {}).items():
            target = data
            for part in path[:-1]:
                target = target.setdefault(part, {})
            defaults = self._extract_defaults(section)
            defaults.update(self._GUI_HIDDEN_OVERRIDES.get(path, {}))
            target.setdefault(path[-1], defaults)
        return data

    @staticmethod
    def _extract_defaults(section: dict) -> dict:
        """Extract default values from a schema section."""
        defaults = {}
        for k, v in section.items():
            if not isinstance(v, dict):
                continue
            if 'type' in v and 'default' in v:
                defaults[k] = v['default']
            elif 'description' in v:
                sub = SchemaForm._extract_defaults(v)
                if sub:
                    defaults[k] = sub
        return defaults

    def populate_from_data(self, data: dict) -> None:
        if not self.widgets_map:
            return
        self._clear_dynamic_items()
        self._populate_widgets_recursively(self.widgets_map, data)

    def _collect_data_recursively(self, widget_level):
        data = {}
        for key, item in widget_level.items():
            if isinstance(item, dict) and '_group' in item:
                group_box = item['_group']
                if group_box.isCheckable() and not group_box.isChecked():
                    continue
                sub_data_widgets = {k: v for k, v in item.items() if k != '_group'}
                sub_data = self._collect_data_recursively(sub_data_widgets)
                for toggle_field in self._section_toggle_fields.get(key, []):
                    sub_data[toggle_field] = True
                data[key] = sub_data
            elif isinstance(item, dict) and '__dyn_ref__' in item:
                dyn_key = item['__dyn_ref__']
                if dyn_key in self.dynamic_widgets:
                    dyn_data = self._collect_dynamic_section_data(dyn_key)
                    if dyn_data:
                        data[key] = dyn_data
            elif isinstance(item, QWidget):
                value = self._get_widget_value(item)
                if value is not None and value != '' and value != {}:
                    data[key] = value

        return data

    def _collect_dynamic_section_data(self, key):
        dyn_info = self.dynamic_widgets[key]

        if 'card_panel' in dyn_info:
            return dyn_info['card_panel'].collect_data()

        dyn_data = {}
        for item_widget in dyn_info['widgets']:
            group_box = item_widget.findChild(QGroupBox)
            if group_box and group_box.isChecked():
                item_name = group_box.title()
                widget_storage = group_box.property('widget_storage')
                if widget_storage:
                    dyn_data[item_name] = self._collect_data_recursively(widget_storage)
        return dyn_data

    def _populate_widgets_recursively(self, widget_level, data_level):
        if not isinstance(data_level, dict):
            return

        for key, item in widget_level.items():
            if key not in data_level:
                continue

            value = data_level[key]
            if isinstance(item, dict) and '_group' in item:
                if not isinstance(value, dict):
                    continue
                group_box = item['_group']
                if group_box.isCheckable():
                    group_box.setChecked(True)
                sub_widgets = {k: v for k, v in item.items() if k != '_group'}
                self._populate_widgets_recursively(sub_widgets, value)
            elif isinstance(item, dict) and '__dyn_ref__' in item:
                dyn_key = item['__dyn_ref__']
                if dyn_key in self.dynamic_widgets and isinstance(value, dict):
                    dyn_info = self.dynamic_widgets[dyn_key]
                    if 'card_panel' in dyn_info:
                        dyn_info['card_panel'].populate(value)
                    else:
                        for item_name, item_data in value.items():
                            self.add_dynamic_item(
                                dyn_key, name=item_name, data=item_data
                            )
            elif isinstance(item, QWidget):
                self._set_widget_value(item, value)

    @staticmethod
    def _set_widget_value(widget, value):
        from atlas.core.gui.widgets.periodic_table_widget import (
            ElementPickerField,
        )

        if isinstance(widget, ElementPickerField):
            widget.set_elements(value if isinstance(value, list) else [])
        elif isinstance(widget, (KspacingWidget, IncarWidget)):
            widget.set_value(value)
        elif isinstance(widget, _CheckBoxGroup):
            widget.set_checked(value if isinstance(value, list) else [])
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QLineEdit):
            if isinstance(value, list):
                widget.setText(', '.join(map(str, value)))
            elif value is None:
                vtype = widget.property('value_type') or ''
                if 'float' in vtype or 'int' in vtype:
                    widget.setText('')
                else:
                    widget.setText(str(value))
            else:
                widget.setText(str(value))

    # ===================================================== widget factory

    def _create_widget(self, item_def, key='', path=None, siblings=None):
        widget_type_str = item_def.get('type', 'str')
        default_value = item_def.get('default')
        choices = item_def.get('choices') or item_def.get('allowed')
        suggestions = item_def.get('suggestions')
        full_path = '.'.join(path) if path else key
        siblings = siblings or {}

        default_font = QFont()

        if default_value == 'None':
            default_value = None

        if key == 'base_element' and 'element_list' in siblings:
            from atlas.core.gui.widgets.periodic_table_widget import (
                ElementPickerField,
            )

            picker = siblings['element_list']
            widget = QComboBox()
            widget.setEditable(True)
            widget.setInsertPolicy(QComboBox.NoInsert)
            widget.lineEdit().setPlaceholderText(item_def.get('example', 'e.g. Cu'))
            if isinstance(picker, ElementPickerField):
                for el in picker.elements():
                    widget.addItem(el)
                picker.elements_changed.connect(
                    lambda elems, cb=widget: (
                        cb.clear(),
                        [cb.addItem(e) for e in elems],
                    )
                )
            if default_value is not None:
                widget.setCurrentText(str(default_value))
        elif widget_type_str == 'dict' and 'kspacing' in key:
            widget = KspacingWidget()
            if isinstance(default_value, dict):
                widget.set_value(default_value)
            widget.value_changed.connect(self._emit_data_changed)
        elif widget_type_str == 'dict' and ('incar' in key or 'incar' in full_path):
            widget = IncarWidget()
            if isinstance(default_value, dict):
                widget.set_value(default_value)
            widget.value_changed.connect(self._emit_data_changed)
        elif choices and 'list' in widget_type_str:
            widget = _CheckBoxGroup(choices)
            if isinstance(default_value, list):
                widget.set_checked(default_value)
        elif choices:
            widget = QComboBox()
            widget.addItems(map(str, choices))
            if default_value is not None:
                widget.setCurrentText(str(default_value))
        elif suggestions:
            widget = QComboBox()
            widget.setEditable(True)
            widget.setInsertPolicy(QComboBox.NoInsert)
            placeholder = item_def.get('example', '')
            widget.lineEdit().setPlaceholderText(placeholder)
            widget.lineEdit().setProperty('_original_placeholder', placeholder)
            if default_value is not None:
                widget.setCurrentText(str(default_value))
            widget.setProperty('suggestions_key', suggestions)
        elif 'list' in widget_type_str and key == 'element_list':
            from atlas.core.gui.widgets.periodic_table_widget import (
                ElementPickerField,
            )

            widget = ElementPickerField()
            if default_value:
                widget.set_elements(
                    default_value if isinstance(default_value, list) else []
                )
            widget.elements_changed.connect(self._emit_data_changed)
        elif 'list' in widget_type_str:
            widget = QLineEdit(', '.join(map(str, default_value or [])))
        elif 'bool' in widget_type_str:
            widget = QCheckBox()
            if default_value is not None:
                widget.setChecked(bool(default_value))
        elif 'int' in widget_type_str:
            if default_value is None:
                widget = QLineEdit()
                widget.setPlaceholderText(
                    str(item_def.get('example', f'Optional {widget_type_str}'))
                )
            else:
                widget = _NoScrollSpinBox()
                lo = int(item_def['min']) if 'min' in item_def else -1_000_000
                hi = int(item_def['max']) if 'max' in item_def else 1_000_000_000
                widget.setRange(lo, hi)
                if default_value is not None:
                    widget.setValue(int(default_value))
        elif 'float' in widget_type_str:
            if default_value is None:
                widget = QLineEdit()
                widget.setPlaceholderText(
                    str(item_def.get('example', f'Optional {widget_type_str}'))
                )
            else:
                widget = _NoScrollDoubleSpinBox()
                lo = float(item_def['min']) if 'min' in item_def else -1e9
                hi = float(item_def['max']) if 'max' in item_def else 1e9
                widget.setRange(lo, hi)
                widget.setDecimals(5)
                if default_value is not None:
                    widget.setValue(float(default_value))
        else:
            widget = QLineEdit(str(default_value or ''))

        widget.setFont(default_font)
        widget.setProperty(
            'description', item_def.get('description', 'No description available.')
        )
        widget.setProperty('value_type', item_def.get('type', 'N/A'))
        widget.setProperty('mandatory', str(item_def.get('mandatory', False)))
        raw_default = item_def.get('default')
        widget.setProperty(
            'default_value', str(raw_default) if raw_default is not None else ''
        )
        desc = item_def.get('description', '')
        if desc:
            widget.setToolTip(f'<p>{desc}</p>')
        return widget

    @staticmethod
    def _get_widget_value(widget):
        from atlas.core.gui.widgets.periodic_table_widget import (
            ElementPickerField,
        )

        value_type = widget.property('value_type') or ''

        if isinstance(widget, ElementPickerField):
            return widget.elements()
        if isinstance(widget, KspacingWidget):
            return widget.get_value()
        if isinstance(widget, IncarWidget):
            return widget.get_value()
        if isinstance(widget, _CheckBoxGroup):
            return widget.get_checked()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            text = widget.currentText()
            try:
                if 'int' in value_type:
                    return int(text)
                if 'float' in value_type:
                    return float(text)
            except (ValueError, TypeError):
                pass
            return text
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if not text:
                return None
            if text == 'None' and ('float' in value_type or 'int' in value_type):
                return None

            if 'list' in value_type:
                items = [item.strip() for item in text.split(',')]
                typed_items = []
                for item in items:
                    try:
                        typed_items.append(int(item))
                    except ValueError:
                        try:
                            typed_items.append(float(item))
                        except ValueError:
                            typed_items.append(item)
                return typed_items

            try:
                if 'int' in value_type:
                    return int(text)
                if 'float' in value_type:
                    return float(text)
            except (ValueError, TypeError):
                pass
            return text
        return None

    # ======================================================= suggestions

    def set_suggestions_loading(self) -> None:
        """Show a 'Loading...' placeholder on suggestion combo boxes."""
        for widget in self._iter_all_widgets(self.widgets_map):
            if not isinstance(widget, QComboBox):
                continue
            skey = widget.property('suggestions_key')
            if skey and widget.count() == 0:
                widget.lineEdit().setPlaceholderText('Loading suggestions…')

    def populate_suggestions(self, suggestions: dict[str, list[str]]) -> None:
        """Fill editable combo boxes tagged with a ``suggestions`` key.

        *suggestions* maps suggestion-key names (e.g. ``"aiida_codes"``)
        to a list of strings.  Every ``QComboBox`` whose
        ``suggestions_key`` property matches a key gets its items
        replaced while preserving the current user text.
        """
        for widget in self._iter_all_widgets(self.widgets_map):
            if not isinstance(widget, QComboBox):
                continue
            skey = widget.property('suggestions_key')
            if skey and skey in suggestions:
                current = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItems(suggestions[skey])
                widget.setCurrentText(current)
                le = widget.lineEdit()
                if le is not None:
                    example = le.property('_original_placeholder') or ''
                    le.setPlaceholderText(example)
                widget.blockSignals(False)

    @classmethod
    def _iter_all_widgets(cls, widget_level) -> Iterator[QWidget]:
        """Yield every leaf widget, including those inside groups."""
        for item in widget_level.values():
            if isinstance(item, dict) and '_group' in item:
                sub = {k: v for k, v in item.items() if k != '_group'}
                yield from cls._iter_all_widgets(sub)
            elif isinstance(item, QWidget):
                yield item

    # ============================================================== events

    def eventFilter(self, source, event):
        if event.type() == QEvent.FocusIn:
            description = source.property('description') or ''
            value_type = source.property('value_type') or ''
            schema_path = source.property('schema_path') or ()
            schema_key = source.property('schema_key') or ''
            mandatory = source.property('mandatory') or 'False'
            default_value = source.property('default_value') or ''
            if description or schema_path:
                self.field_focused.emit(
                    description,
                    value_type,
                    tuple(schema_path),
                    schema_key,
                    mandatory,
                    default_value,
                )
        return super().eventFilter(source, event)

    # ============================================================ helpers

    @staticmethod
    def _update_optional_style(group_box, checked: bool) -> None:
        group_box.setProperty('section_off', 'false' if checked else 'true')
        style = group_box.style()
        style.unpolish(group_box)
        style.polish(group_box)
        for child in group_box.findChildren(QWidget):
            style.unpolish(child)
            style.polish(child)

    def _emit_data_changed(self, *_args, **_kwargs):
        self.data_changed.emit()

    # =============================================================== search

    def iter_leaf_widgets(self) -> Iterator[QWidget]:
        """Yield every leaf input widget in the active section, in form order."""
        yield from self._iter_leaves(self.widgets_map)
        for dyn_info in self.dynamic_widgets.values():
            for item_container in dyn_info['widgets']:
                group_box = item_container.findChild(QGroupBox)
                if group_box is None:
                    continue
                storage = group_box.property('widget_storage')
                if storage:
                    yield from self._iter_leaves(storage)

    @classmethod
    def _iter_leaves(cls, widget_level) -> Iterator[QWidget]:
        for item in widget_level.values():
            if isinstance(item, dict) and '_group' in item:
                sub = {k: v for k, v in item.items() if k != '_group'}
                yield from cls._iter_leaves(sub)
            elif isinstance(item, QWidget):
                yield item

    @staticmethod
    def _widget_matches(widget: QWidget, needle: str) -> bool:
        haystacks = (
            widget.property('label_text') or '',
            widget.property('schema_key') or '',
            widget.property('description') or '',
        )
        return any(needle in (h or '').lower() for h in haystacks)

    def _on_search_text_changed(self, query: str) -> None:
        query = (query or '').strip()
        self._last_query = query
        if not query:
            self._search_matches = []
            self._search_index = -1
            self.search_count_label.setText('')
            return

        needle = query.lower()
        self._search_matches = [
            w for w in self.iter_leaf_widgets() if self._widget_matches(w, needle)
        ]
        if not self._search_matches:
            self._search_index = -1
        else:
            self._search_index = 0
        self._update_search_count_label()

    def _search_step_forward(self) -> None:
        self._step_search(+1)

    def _search_step_backward(self) -> None:
        self._step_search(-1)

    def _step_search(self, delta: int) -> None:
        if not self._search_matches:
            return
        self._search_index = (self._search_index + delta) % len(self._search_matches)
        self._focus_current_match()
        self._update_search_count_label()

    def _focus_current_match(self) -> None:
        if not (0 <= self._search_index < len(self._search_matches)):
            return
        widget = self._search_matches[self._search_index]
        self.scroll_area.ensureWidgetVisible(widget, 50, 100)
        widget.setFocus(Qt.ShortcutFocusReason)

    def _update_search_count_label(self) -> None:
        if not self._last_query:
            self.search_count_label.setText('')
            return
        if not self._search_matches:
            self.search_count_label.setText('0 / 0')
            return
        self.search_count_label.setText(
            f'{self._search_index + 1} / {len(self._search_matches)}'
        )

    def _clear_search(self) -> None:
        self.search_input.clear()
        self._search_matches = []
        self._search_index = -1
        self._last_query = ''
        self.search_count_label.setText('')
