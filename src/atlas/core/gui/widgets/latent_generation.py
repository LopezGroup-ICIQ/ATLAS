"""Latent-space generation panel.

The latent map drawn by :mod:`~atlas.core.gui.widgets.boundary_panel` is built
by ``InitialDatabase.descriptors_concave_hull``: descriptors, an autoencoder,
and a boundary. This pane is the GUI surface for rebuilding it, which is what
turns a non-invertible map (SOAP, or one predating provenance tracking) into
one the traversal can decode.

Regeneration itself runs as the ``atl_gen_init_db boundary`` subprocess, so
heavy torch/dscribe work never blocks the GUI process and the same command is
available from the terminal. The panel's selections are written into the
project's ``database_generation`` configuration before the run, keeping the
Config tab's Hull section as the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.process.runner import ProcessRunner
from atlas.core.gui.project import Project

#: Config file the project keeps its active database-generation settings in;
#: the same file ``InitDbPage.run()`` passes to ``atl_gen_init_db generate``.
CANONICAL_CONFIG = 'database_generation_settings.toml'

DESCRIPTOR_SOAP = 'SOAP'

BOUNDARY_METHODS: tuple[tuple[str, str], ...] = (
    ('Concave hull', 'concave_hull'),
    ('Morphological closing', 'morphological_closing'),
)

DEVICES = ('cpu', 'cuda')


def descriptor_options() -> list[tuple[str, str]]:
    """Return ``(label, model_spec)`` pairs, SOAP first with an empty spec.

    MACE foundation models come from the backend registry so the list cannot
    drift from what the backends actually accept; a missing MACE install falls
    back to the two most common variants rather than hiding the option.
    """
    options = [(DESCRIPTOR_SOAP, '')]
    try:
        from atlas.active_learning.backends import get_backend

        models = get_backend('mace').available_pretrained_models
        options.extend((f'MACE {spec.removeprefix("mace:")}', spec) for spec in models)
    except Exception:
        options.extend(
            [
                ('MACE mp-small', 'mace:mp-small'),
                ('MACE mp-medium', 'mace:mp-medium'),
            ]
        )
    return options


def merge_concave_config(toml_text: str, values: dict) -> str:
    """Return *toml_text* with its ``[concave_hull]`` block updated.

    Pure so the merge is testable without a project. A missing or empty
    snapshot yields a minimal document containing just the new block.
    """
    import tomli
    import tomli_w

    data = tomli.loads(toml_text) if toml_text and toml_text.strip() else {}
    concave = data.setdefault('concave_hull', {})
    concave.update(values)
    # An explicit boundary run is exactly what this flag gates in the full
    # pipeline, so keep the config honest about what was generated.
    concave['gen_concave_hull'] = True
    return tomli_w.dumps(data)


def build_boundary_command(
    config_path: Path | str,
    db_path: Path | str,
) -> list[str]:
    """Command that recomputes the map of an existing database."""
    return [
        'atl_gen_init_db',
        'boundary',
        '-c',
        str(config_path),
        '-d',
        str(db_path),
    ]


class LatentGenerationPane(QWidget):
    """Descriptor/model selection and the regeneration run for one project."""

    #: A new latent map exists on disk and the panel should reload it.
    generated = Signal()
    #: The project's database-generation snapshot was rewritten.
    config_updated = Signal()
    #: A line of subprocess output, for the page's log.
    log_message = Signal(str)
    #: Whether a regeneration is in flight.
    busy_changed = Signal(bool)
    #: No database to regenerate from; the page should point at the Config tab.
    database_missing = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self._project = project
        self._runner: ProcessRunner | None = None
        self._busy = False
        self._traversal_running = False
        self._map_summary = 'No latent-space map yet.'

        pane = QVBoxLayout(self)
        pane.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        hint = QLabel(
            'Rebuild the latent-space map from this database. A MACE descriptor '
            'makes the map invertible, which is what latent-space traversal '
            'needs; SOAP maps can be plotted but not decoded.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: palette(mid);')
        layout.addWidget(hint)

        map_group = QGroupBox('Current map')
        map_layout = QVBoxLayout(map_group)
        self._map_label = QLabel(self._map_summary)
        self._map_label.setWordWrap(True)
        map_layout.addWidget(self._map_label)
        layout.addWidget(map_group)

        options_group = QGroupBox('Generation settings')
        options_layout = QVBoxLayout(options_group)

        descriptor_row = QHBoxLayout()
        descriptor_row.addWidget(QLabel('Descriptor'))
        self._descriptor_combo = QComboBox()
        for label, spec in descriptor_options():
            self._descriptor_combo.addItem(
                label, {'type': 'MACE' if spec else 'SOAP', 'model': spec}
            )
        self._descriptor_combo.currentIndexChanged.connect(self._on_descriptor_changed)
        descriptor_row.addWidget(self._descriptor_combo, 1)
        options_layout.addLayout(descriptor_row)

        self._soap_warning = QLabel(
            'SOAP maps cannot be inverted; traversal will stay unavailable. '
            'Choose a MACE model if you need to decode structures.'
        )
        self._soap_warning.setWordWrap(True)
        self._soap_warning.setStyleSheet('color: #b8860b;')
        self._soap_warning.setVisible(False)
        options_layout.addWidget(self._soap_warning)

        boundary_row = QHBoxLayout()
        boundary_row.addWidget(QLabel('Boundary'))
        self._boundary_combo = QComboBox()
        for label, value in BOUNDARY_METHODS:
            self._boundary_combo.addItem(label, value)
        boundary_row.addWidget(self._boundary_combo, 1)
        options_layout.addLayout(boundary_row)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel('Device'))
        self._device_combo = QComboBox()
        self._device_combo.addItems(DEVICES)
        device_row.addWidget(self._device_combo, 1)
        options_layout.addLayout(device_row)

        saved_note = QLabel('Settings are saved to the project configuration.')
        saved_note.setWordWrap(True)
        saved_note.setStyleSheet('color: palette(mid);')
        options_layout.addWidget(saved_note)
        layout.addWidget(options_group)

        self._db_warning = QLabel('')
        self._db_warning.setWordWrap(True)
        self._db_warning.setStyleSheet('color: palette(mid);')
        self._db_warning.setVisible(False)
        layout.addWidget(self._db_warning)

        self._go_config_btn = QPushButton('Go to Config to generate a database')
        self._go_config_btn.clicked.connect(lambda: self.database_missing.emit())
        self._go_config_btn.setVisible(False)
        layout.addWidget(self._go_config_btn)

        self._generate_btn = QPushButton('Generate latent space')
        self._generate_btn.clicked.connect(self.run)
        layout.addWidget(self._generate_btn)

        self._cancel_btn = QPushButton('Cancel')
        self._cancel_btn.clicked.connect(self.cancel)
        self._cancel_btn.setVisible(False)
        layout.addWidget(self._cancel_btn)

        self._status = QLabel('')
        self._status.setWordWrap(True)
        self._status.setStyleSheet('color: palette(mid); font-size: 11px;')
        layout.addWidget(self._status)

        layout.addStretch()
        scroll.setWidget(container)
        pane.addWidget(scroll)

        self.load_from_config()
        self._on_descriptor_changed()
        self.refresh()

    # ------------------------------------------------------------- state

    def load_from_config(self) -> None:
        """Mirror the active ``[concave_hull]`` config into the form.

        Called at construction and whenever the Config tab saves, so the two
        surfaces cannot disagree about what the next run will use. A model
        that is not in the registry (a trained file, typically) is added to the
        combo rather than silently dropping back to SOAP.
        """
        import tomli

        snapshot = self._project.active_config('database_generation')
        if snapshot is None:
            return
        try:
            concave = tomli.loads(snapshot[1]).get('concave_hull') or {}
        except tomli.TOMLDecodeError:
            return

        descriptor = str(concave.get('descriptor', DESCRIPTOR_SOAP)).upper()
        model = str(concave.get('descriptor_model') or '')
        if descriptor == 'MACE':
            index = self._find_model_index(model)
            if index is None and model:
                self._descriptor_combo.addItem(
                    f'MACE {Path(model).name}', {'type': 'MACE', 'model': model}
                )
                index = self._descriptor_combo.count() - 1
            self._descriptor_combo.setCurrentIndex(index or 0)
        else:
            self._descriptor_combo.setCurrentIndex(0)

        method = concave.get('boundary_method', 'concave_hull')
        for i in range(self._boundary_combo.count()):
            if self._boundary_combo.itemData(i) == method:
                self._boundary_combo.setCurrentIndex(i)
                break

        self._device_combo.setCurrentText(str(concave.get('device', 'cpu')))

    def _find_model_index(self, model: str) -> int | None:
        for i in range(self._descriptor_combo.count()):
            data = self._descriptor_combo.itemData(i) or {}
            if data.get('type') == 'MACE' and data.get('model') == model:
                return i
        return None

    def refresh(self) -> None:
        """Re-evaluate what can be generated right now."""
        db_path = self._project.init_db_path()
        db_ok = db_path.exists()

        if db_ok:
            self._db_warning.setVisible(False)
        else:
            self._db_warning.setText(
                'No initial database was found for this project. Generate one '
                'from the Config tab before rebuilding the latent space.'
            )
            self._db_warning.setVisible(True)
        self._go_config_btn.setVisible(not db_ok)
        self._update_gate(db_ok)

    def set_current_map(self, data) -> None:
        """Describe the map currently on screen, from its npz provenance."""
        from atlas.core.gui.widgets.latent_inversion import npz_text

        if data is None:
            self._map_label.setText('No latent-space map yet.')
            return

        descriptor = npz_text(data, 'descriptor_type')
        if descriptor is None:
            self._map_label.setText(
                'The current map predates provenance tracking, so how it was '
                'built is not recorded. Regenerate it to enable traversal.'
            )
            return

        model = npz_text(data, 'descriptor_model') or ''
        method = npz_text(data, 'boundary_method') or ''
        parts = [descriptor.upper()]
        if model:
            parts.append(model)
        if method:
            parts.append(method)
        self._map_label.setText(' / '.join(parts))

    def set_traversal_running(self, running: bool) -> None:
        """Called by the panel so a traversal and a generation cannot overlap."""
        self._traversal_running = bool(running)
        self.refresh()

    def is_running(self) -> bool:
        return self._busy

    def _update_gate(self, db_ok: bool) -> None:
        busy = self._busy
        self._generate_btn.setEnabled(
            db_ok and not busy and not self._traversal_running
        )
        if self._traversal_running:
            self._generate_btn.setToolTip('A traversal is running.')
        elif not db_ok:
            self._generate_btn.setToolTip('No database to regenerate from.')
        else:
            self._generate_btn.setToolTip('')

    def _on_descriptor_changed(self) -> None:
        data = self._descriptor_combo.currentData() or {'type': DESCRIPTOR_SOAP}
        self._soap_warning.setVisible(data.get('type') == DESCRIPTOR_SOAP)

    # -------------------------------------------------------------- run

    def _selected_config_values(self) -> dict:
        data = self._descriptor_combo.currentData() or {'type': DESCRIPTOR_SOAP}
        values = {
            'descriptor': data.get('type', DESCRIPTOR_SOAP),
            'boundary_method': self._boundary_combo.currentData(),
            'device': self._device_combo.currentText(),
        }
        # SOAP ignores the model field; writing it anyway would mislead the
        # Config tab about which model is in play.
        if data.get('type') == 'MACE' and data.get('model'):
            values['descriptor_model'] = data['model']
        return values

    def run(self) -> None:
        """Save the settings and launch the boundary subprocess."""
        if self.is_running():
            return
        db_path = self._project.init_db_path()
        if not db_path.exists():
            self.database_missing.emit()
            return

        snapshot = self._project.active_config('database_generation')
        toml_text = snapshot[1] if snapshot else ''
        try:
            merged = merge_concave_config(toml_text, self._selected_config_values())
            self._project.save_config_snapshot(
                'database_generation', merged, label='latent-space generation'
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Could not save settings', str(exc))
            return
        self.config_updated.emit()

        self._status.setText('Generating descriptors, autoencoder and boundary...')
        self._set_busy(True)
        self._runner = ProcessRunner(
            build_boundary_command(
                self._project.config_path('database_generation'), db_path
            ),
            cwd=self._project.cwd(),
            log_file=self._project.log_path('atl_gen_init_db_boundary'),
        )
        self._runner.log_message.connect(self._on_log_line)
        self._runner.process_finished.connect(self._on_finished)
        self._runner.start()

    def cancel(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            self._runner.stop()

    def stop(self) -> None:
        """Terminate a running regeneration when the window goes away."""
        if self._runner is not None and self._runner.isRunning():
            self._runner.stop()
            self._runner.wait(5000)

    def _on_log_line(self, line: str) -> None:
        self.log_message.emit(line)
        stripped = line.strip()
        if stripped:
            self._status.setText(stripped)

    def _on_finished(self, return_code: int) -> None:
        self._runner = None
        self._set_busy(False)
        if return_code == 0:
            self._status.setText('Latent space regenerated.')
            self.generated.emit()
            return
        self._status.setText(f'Regeneration failed (exit code {return_code}).')
        QMessageBox.critical(
            self,
            'Latent-space generation failed',
            'atl_gen_init_db boundary did not complete.\n\n'
            'Check the log dock for details.',
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._cancel_btn.setVisible(busy)
        self._descriptor_combo.setEnabled(not busy)
        self._boundary_combo.setEnabled(not busy)
        self._device_combo.setEnabled(not busy)
        self.refresh()
        self.busy_changed.emit(busy)
