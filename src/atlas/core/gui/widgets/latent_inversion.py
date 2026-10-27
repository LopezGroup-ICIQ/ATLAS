"""Dialogs and workers for latent-space inversion and traversal.

The latent map drawn by :mod:`~atlas.core.gui.widgets.boundary_panel` is a
differentiable function of the atomic positions when it was built with an MLIP
descriptor, which means a point on the map can be decoded back into a
structure. This module holds the user-facing half of that feature:

* :class:`LatentTraversalSettingsDialog` -- every knob of
  :class:`~atlas.active_learning.extrapolation.latent_inversion.LatentInversionSettings`,
  persisted per project so a traversal is configured once;
* :class:`TraversalWorker` -- runs the optimization off the UI thread and
  reports each decoded structure as it lands;
* :class:`LatentTraversalResultsDialog` -- the decoded structures, their
  diagnostics, and the two things one can do with them (add to the initial
  database, or export).

Everything torch-related is imported inside :meth:`TraversalWorker.run`, so
importing this module costs nothing on a machine without torch and the panel
can still gate the feature with a readable reason.
"""

from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from atlas.active_learning.extrapolation.latent_inversion import LatentInversionSettings

#: Key under which a project stores its traversal configuration. Presence of
#: the key is what tells the panel whether to show the settings dialog before
#: the first run.
SETTINGS_META_KEY = 'latent_traversal_settings'


# ---------------------------------------------------------------------------
# Per-project persistence
# ---------------------------------------------------------------------------


def has_saved_settings(project) -> bool:
    """Whether this project has ever had its traversal configured."""
    return project.meta(SETTINGS_META_KEY) is not None


def load_settings(project) -> LatentInversionSettings:
    """Return the project's traversal settings, falling back to defaults.

    A stored blob written by a different version is filtered field by field
    (see ``LatentInversionSettings.from_dict``), and unreadable JSON is
    discarded, because a settings file must never be able to break the panel.
    """
    raw = project.meta(SETTINGS_META_KEY)
    if not raw:
        return LatentInversionSettings()
    try:
        return LatentInversionSettings.from_dict(json.loads(raw))
    except (ValueError, TypeError):
        return LatentInversionSettings()


def save_settings(project, settings: LatentInversionSettings) -> None:
    """Persist *settings* on the project."""
    project.set_meta(SETTINGS_META_KEY, json.dumps(settings.to_dict()))


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

#: Filenames the autoencoder's standardization arrays are written under.
STANDARDIZATION_FILES = ('ae_mean_vals.npy', 'ae_std_vals.npy')


class TraversalGate(NamedTuple):
    """Outcome of the availability check, with a reason and an action hint.

    ``recompute_fixable`` says whether regenerating the latent space can
    remove the obstacle, which is what decides between offering the user a
    "Regenerate latent space" button or just explaining the problem.
    """

    available: bool
    #: Empty when available; a user-facing explanation otherwise.
    reason: str
    #: Whether latent-space regeneration can lift this obstacle.
    recompute_fixable: bool


def npz_text(data, key: str) -> str | None:
    """Read a 1-element string array out of a loaded npz mapping."""
    if data is None or key not in data:
        return None
    value = data[key]
    try:
        return str(value[0])
    except (IndexError, TypeError):
        return str(value)


def resolve_autoencoder_path(npz_path: Path, data) -> Path:
    """Locate the autoencoder that built this map.

    The recorded path is authoritative, but a project directory that was moved
    or copied invalidates it, so fall back to the conventional filename beside
    the map itself.
    """
    npz_path = Path(npz_path)
    recorded = npz_text(data, 'autoencoder_path')
    if recorded:
        candidate = Path(recorded)
        if candidate.is_file():
            return candidate
        beside = npz_path.parent / candidate.name
        if beside.is_file():
            return beside
    return npz_path.parent / 'autoencoder_model.pth'


def traversal_availability(
    npz_path: Path | None,
    data,
    db_available: bool,
) -> TraversalGate:
    """Whether a traversal can run here, and why not when it cannot.

    Ordered from the most fundamental obstacle to the most incidental, so the
    reason shown is the one the user has to fix first. The descriptor width
    against the encoder's input width is deliberately *not* checked here: it
    needs both models loaded, and a mismatch raises a precise error at run
    time rather than silently disabling the button.
    """
    if npz_path is None or data is None:
        return TraversalGate(
            False,
            'No latent-space data has been generated for this project.',
            True,
        )

    import importlib.util

    if importlib.util.find_spec('torch') is None:
        return TraversalGate(
            False,
            'PyTorch is not installed in this environment, so the encoder '
            'cannot be evaluated.',
            False,
        )

    descriptor_type = npz_text(data, 'descriptor_type')
    if descriptor_type is None:
        return TraversalGate(
            False,
            'This latent space predates descriptor provenance tracking, so '
            'there is no record of how it was built. Re-run the boundary '
            'generation to enable inversion.',
            True,
        )

    descriptor_type = descriptor_type.lower()
    if descriptor_type == 'soap':
        return TraversalGate(
            False,
            'SOAP provides no analytic gradients for periodic, averaged '
            'descriptors, so this map cannot be inverted. Re-run the boundary '
            'generation with the MACE descriptor.',
            True,
        )

    try:
        from atlas.active_learning.backends import supports_descriptor_gradients
    except ImportError as exc:
        return TraversalGate(
            False,
            f'The MLIP backend registry could not be imported: {exc}',
            False,
        )

    if not supports_descriptor_gradients(descriptor_type):
        return TraversalGate(
            False,
            f"The '{descriptor_type}' descriptor backend does not expose "
            'differentiable descriptors, either because it cannot or because '
            'its dependencies are missing in this environment.',
            True,
        )

    autoencoder_path = resolve_autoencoder_path(npz_path, data)
    if not autoencoder_path.is_file():
        return TraversalGate(
            False,
            f'The autoencoder model ({autoencoder_path.name}) was not found '
            f'in {autoencoder_path.parent}. Re-run the boundary generation.',
            True,
        )

    for name in STANDARDIZATION_FILES:
        if not (autoencoder_path.parent / name).is_file():
            return TraversalGate(
                False,
                f'The autoencoder standardization array {name} is missing from '
                f'{autoencoder_path.parent}, so descriptors cannot be scaled '
                'the way the encoder was trained. Re-run the boundary '
                'generation.',
                True,
            )

    if not db_available:
        return TraversalGate(
            False,
            'The initial database could not be loaded, and inversion needs it '
            'to seed the optimization with real structures.',
            False,
        )

    return TraversalGate(True, '', False)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

# (group title, (field spec, ...)). Each spec is (attribute, label, kind, opts)
# where *kind* selects the widget. Kept declarative so the dialog and the
# dataclass cannot drift apart silently: every attribute named here is checked
# against the dataclass in the tests.
_FIELD_GROUPS = (
    (
        'Path',
        (
            (
                'n_structures',
                'Structures to decode',
                'int',
                {
                    'min': 1,
                    'max': 500,
                    'tip': 'How many structures to decode along the path through the '
                    'marked points. The path is sampled evenly in latent '
                    'distance, so more structures means a denser traversal and '
                    'one exactly per marked point is the minimum. With snapping '
                    'on, each marked point is forced onto its nearest sample.',
                },
            ),
            (
                'snap_waypoints',
                'Snap to waypoints',
                'bool',
                {
                    'tip': 'Force one sample onto each marked point, so the path '
                    'passes exactly through what was clicked.'
                },
            ),
            (
                'chain_from_previous',
                'Chain from previous',
                'bool',
                {
                    'tip': 'Offer the previous decoded structure as a seed for the '
                    'next point. Faster, and gives a visibly smoother path.'
                },
            ),
        ),
    ),
    (
        'Seeding',
        (
            (
                'seed_strategy',
                'Strategy',
                'choice',
                {
                    'choices': ('nearest', 'by_size', 'fixed_size'),
                    'tip': "'nearest' picks the closest structures in latent space; "
                    'the size strategies additionally restrict the atom count.',
                },
            ),
            (
                'target_n_atoms',
                'Target atoms',
                'optional_int',
                {
                    'min': 0,
                    'max': 2000,
                    'special': 'any size',
                    'tip': 'Decode structures with this many atoms. Required by the '
                    'size-based strategies.',
                },
            ),
            (
                'size_tolerance',
                'Size tolerance',
                'int',
                {
                    'min': 0,
                    'max': 100,
                    'tip': 'How many atoms a seed may differ from the target size.',
                },
            ),
            (
                'n_starts',
                'Restarts',
                'int',
                {
                    'min': 1,
                    'max': 8,
                    'tip': 'Seeds tried per point. The best result wins; feasible '
                    'and accurate results stop the remaining restarts early.',
                },
            ),
            (
                'require_compact',
                'Require compact seeds',
                'bool',
                {
                    'tip': 'Reject seeds that are already fragmented, which no '
                    'amount of optimization reliably repairs.'
                },
            ),
        ),
    ),
    (
        'Optimizer',
        (
            (
                'optimizer',
                'Algorithm',
                'choice',
                {
                    'choices': ('lbfgs',),
                    'tip': 'L-BFGS with a strong-Wolfe line search: the objective is '
                    'smooth and stiff, which first-order methods handle poorly.',
                },
            ),
            (
                'n_rounds',
                'Rounds',
                'int',
                {
                    'min': 1,
                    'max': 30,
                    'tip': 'Neighbour lists are rebuilt between rounds, since the '
                    'geometry moves far enough to invalidate them.',
                },
            ),
            (
                'max_iter',
                'Iterations per round',
                'int',
                {'min': 10, 'max': 1000, 'tip': 'L-BFGS iterations within one round.'},
            ),
            (
                'lr',
                'Learning rate',
                'float',
                {'min': 1e-4, 'max': 10.0, 'decimals': 4, 'step': 0.01},
            ),
            (
                'tolerance',
                'Latent tolerance',
                'sci',
                {
                    'tip': 'Stop once the latent residual falls below this, but only '
                    'at full penalty strength.'
                },
            ),
            (
                'device',
                'Device',
                'choice',
                {
                    'choices': ('cpu', 'cuda'),
                    'editable': True,
                    'tip': 'CPU by default: a CUDA out-of-memory inside a GUI '
                    'thread is not recoverable.',
                },
            ),
            (
                'dtype',
                'Precision',
                'choice',
                {
                    'choices': ('float64', 'float32'),
                    'tip': 'Double by default: a strong-Wolfe line search on a stiff '
                    'inverse barrier is far better conditioned in double.',
                },
            ),
        ),
    ),
    (
        'Objective',
        (
            (
                'w_lat',
                'Latent weight',
                'float',
                {'min': 0.0, 'max': 1e6, 'decimals': 3, 'step': 0.5},
            ),
            (
                'w_rep',
                'Repulsion weight',
                'float',
                {
                    'min': 0.0,
                    'max': 1e6,
                    'decimals': 3,
                    'step': 1.0,
                    'tip': 'Barrier keeping pairs above the covalent floor.',
                },
            ),
            (
                'w_coh',
                'Cohesion weight',
                'float',
                {
                    'min': 0.0,
                    'max': 1e6,
                    'decimals': 3,
                    'step': 10.0,
                    'tip': 'Penalty on atoms that lose their neighbours, which '
                    'stops the optimizer from dissociating the structure.',
                },
            ),
            (
                'w_disp',
                'Displacement weight',
                'float',
                {
                    'min': 0.0,
                    'max': 1e6,
                    'decimals': 3,
                    'step': 0.5,
                    'tip': 'Search regulariser, not a physical constraint. Set to 0 '
                    'for exploratory decoding far from the seed.',
                },
            ),
            (
                'rep_form',
                'Repulsion form',
                'choice',
                {
                    'choices': ('inverse', 'quadratic'),
                    'tip': "'inverse' diverges at contact; 'quadratic' is softer and "
                    'lets the optimizer cross the floor.',
                },
            ),
            (
                'rep_ramp',
                'Penalty ramp',
                'ramp',
                {
                    'tip': 'Multipliers applied to the repulsion and cohesion weights '
                    'over the first rounds, comma separated. Starting soft lets the '
                    'structure rearrange before the barrier is enforced.'
                },
            ),
        ),
    ),
    (
        'Geometry',
        (
            (
                'covalent_fraction',
                'Covalent fraction f',
                'float',
                {
                    'min': 0.5,
                    'max': 1.0,
                    'decimals': 2,
                    'step': 0.05,
                    'tip': 'A pair is rejected below f * (r_i + r_j).',
                },
            ),
            (
                'bond_factor',
                'Bond factor',
                'float',
                {
                    'min': 1.0,
                    'max': 3.0,
                    'decimals': 2,
                    'step': 0.05,
                    'tip': 'Two atoms count as bonded within this multiple of their '
                    'summed covalent radii.',
                },
            ),
            (
                'margin',
                'Barrier margin',
                'float',
                {
                    'min': 0.0,
                    'max': 1.0,
                    'decimals': 3,
                    'step': 0.01,
                    'tip': 'Extra clearance the barrier aims for above the floor, so '
                    'the accepted geometry is not flush against it.',
                },
            ),
            (
                'min_coord',
                'Minimum coordination',
                'float',
                {'min': 0.0, 'max': 12.0, 'decimals': 2, 'step': 0.5},
            ),
            (
                'd_free',
                'Free displacement (A)',
                'float',
                {
                    'min': 0.0,
                    'max': 20.0,
                    'decimals': 2,
                    'step': 0.25,
                    'tip': 'Atoms may move this far from the seed before the '
                    'displacement restraint acts.',
                },
            ),
            (
                'max_fragments',
                'Maximum fragments',
                'int',
                {
                    'min': 1,
                    'max': 20,
                    'tip': 'A decode with more disconnected pieces than this is '
                    'reported as non-physical.',
                },
            ),
        ),
    ),
    (
        'Cell',
        (
            (
                'relax_cell',
                'Relax cell',
                'bool',
                {'tip': 'Optimize the lattice alongside the positions.'},
            ),
            (
                'max_strain',
                'Maximum strain',
                'float',
                {'min': 0.0, 'max': 1.0, 'decimals': 3, 'step': 0.01},
            ),
            (
                'w_strain',
                'Strain weight',
                'float',
                {'min': 0.0, 'max': 1e6, 'decimals': 3, 'step': 1.0},
            ),
        ),
    ),
    (
        'Reporting',
        (
            (
                'neighbourhood_radius',
                'Neighbourhood radius',
                'float',
                {
                    'min': 0.0,
                    'max': 100.0,
                    'decimals': 3,
                    'step': 0.1,
                    'tip': 'Training points within this latent distance of a target '
                    'are counted, which tells a coverage gap apart from an '
                    'optimizer failure.',
                },
            ),
            (
                'good_enough_latent_err',
                'Good-enough residual',
                'sci',
                {
                    'tip': 'A feasible result below this residual ends the restarts '
                    'for that point.'
                },
            ),
        ),
    ),
)


def _parse_ramp(text: str) -> tuple[float, ...]:
    """Parse a comma- or space-separated ramp, raising ``ValueError`` if bad."""
    parts = [p for p in text.replace(',', ' ').split() if p]
    if not parts:
        raise ValueError('the penalty ramp needs at least one value')
    return tuple(float(p) for p in parts)


class LatentTraversalSettingsDialog(QDialog):
    """Modal editor for every latent-inversion setting."""

    def __init__(
        self,
        settings: LatentInversionSettings | None = None,
        parent=None,
        allow_cell_relaxation: bool = False,
        cell_disabled_reason: str = '',
        min_structures: int = 1,
    ):
        super().__init__(parent)
        self.setWindowTitle('Configure latent traversal')
        self.setMinimumWidth(560)
        self.setMinimumHeight(560)

        self._widgets: dict[str, QWidget] = {}
        self._allow_cell_relaxation = allow_cell_relaxation
        #: One structure per marked point is the floor: a path of N waypoints
        #: cannot be traversed with fewer samples.
        self._min_structures = max(1, int(min_structures))

        outer = QVBoxLayout(self)

        intro = QLabel(
            'The traversal decodes points along the marked path back into '
            'structures by optimizing atomic positions so their latent '
            'coordinate matches the target.'
        )
        intro.setWordWrap(True)
        intro.setStyleSheet('color: palette(mid);')
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(12)

        for title, specs in _FIELD_GROUPS:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setSpacing(8)
            for name, label, kind, opts in specs:
                widget = self._make_widget(kind, opts)
                tip = opts.get('tip')
                if tip:
                    widget.setToolTip(tip)
                self._widgets[name] = widget
                label_widget = QLabel(label)
                if tip:
                    label_widget.setToolTip(tip)
                form.addRow(label_widget, widget)
            container_layout.addWidget(group)

        # One structure per marked point is the floor, so the spin box cannot
        # offer a value that could not cover the path.
        structures_spin = self._widgets['n_structures']
        structures_spin.setMinimum(min(structures_spin.maximum(), self._min_structures))

        if not allow_cell_relaxation:
            relax = self._widgets['relax_cell']
            relax.setEnabled(False)
            reason = cell_disabled_reason or 'Cell relaxation is not available.'
            relax.setToolTip(reason)
            for name in ('max_strain', 'w_strain'):
                self._widgets[name].setEnabled(False)
                self._widgets[name].setToolTip(reason)

        container_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: self.set_settings(LatentInversionSettings())
        )
        outer.addWidget(buttons)

        self.set_settings(settings or LatentInversionSettings())
        self._widgets['seed_strategy'].currentTextChanged.connect(
            self._on_strategy_changed
        )
        self._on_strategy_changed(self._widgets['seed_strategy'].currentText())

    # ------------------------------------------------------------- widgets

    @staticmethod
    def _make_widget(kind: str, opts: dict) -> QWidget:
        if kind == 'bool':
            return QCheckBox()
        if kind in ('int', 'optional_int'):
            widget = QSpinBox()
            widget.setRange(int(opts.get('min', 0)), int(opts.get('max', 1000)))
            if kind == 'optional_int':
                # The minimum doubles as "unset"; a separate checkbox for a
                # single optional integer is more chrome than it is worth.
                widget.setSpecialValueText(opts.get('special', 'unset'))
            return widget
        if kind == 'float':
            widget = QDoubleSpinBox()
            widget.setDecimals(int(opts.get('decimals', 3)))
            widget.setRange(float(opts.get('min', 0.0)), float(opts.get('max', 1e6)))
            widget.setSingleStep(float(opts.get('step', 0.1)))
            return widget
        if kind == 'sci':
            # Values like 1e-4 are unreadable in a spin box with enough
            # decimals to represent them, so these stay free text.
            from PySide6.QtGui import QDoubleValidator

            widget = QLineEdit()
            validator = QDoubleValidator(0.0, 1e9, 12, widget)
            validator.setNotation(QDoubleValidator.ScientificNotation)
            widget.setValidator(validator)
            return widget
        if kind == 'choice':
            widget = QComboBox()
            widget.addItems(list(opts.get('choices', ())))
            widget.setEditable(bool(opts.get('editable', False)))
            return widget
        if kind == 'ramp':
            return QLineEdit()
        raise ValueError(f'unknown field kind: {kind}')

    def _on_strategy_changed(self, strategy: str) -> None:
        # 'nearest' ignores the size window entirely; showing live fields that
        # do nothing is what makes a settings dialog untrustworthy.
        size_based = strategy != 'nearest'
        self._widgets['target_n_atoms'].setEnabled(size_based)
        self._widgets['size_tolerance'].setEnabled(strategy == 'by_size')

    # ------------------------------------------------------------- values

    def set_settings(self, settings: LatentInversionSettings) -> None:
        """Load *settings* into the form."""
        for _, specs in _FIELD_GROUPS:
            for name, _label, kind, _opts in specs:
                widget = self._widgets[name]
                value = getattr(settings, name)
                if kind == 'bool':
                    widget.setChecked(bool(value))
                elif kind == 'optional_int':
                    widget.setValue(widget.minimum() if value is None else int(value))
                elif kind == 'int':
                    widget.setValue(int(value))
                elif kind == 'float':
                    widget.setValue(float(value))
                elif kind == 'sci':
                    widget.setText(f'{float(value):g}')
                elif kind == 'choice':
                    widget.setCurrentText(str(value))
                elif kind == 'ramp':
                    widget.setText(', '.join(f'{v:g}' for v in value))

    def settings(self) -> LatentInversionSettings:
        """Build a settings object from the form.

        Raises ``ValueError`` on a field the widgets cannot constrain
        themselves (the ramp, and the free-text scientific values).
        """
        values: dict = {}
        for _, specs in _FIELD_GROUPS:
            for name, label, kind, _opts in specs:
                widget = self._widgets[name]
                if kind == 'bool':
                    values[name] = widget.isChecked()
                elif kind == 'optional_int':
                    value = widget.value()
                    values[name] = None if value == widget.minimum() else int(value)
                elif kind == 'int':
                    values[name] = int(widget.value())
                elif kind == 'float':
                    values[name] = float(widget.value())
                elif kind == 'sci':
                    text = widget.text().strip()
                    try:
                        values[name] = float(text)
                    except ValueError:
                        raise ValueError(f'{label}: {text!r} is not a number') from None
                elif kind == 'choice':
                    values[name] = widget.currentText().strip()
                elif kind == 'ramp':
                    try:
                        values[name] = _parse_ramp(widget.text())
                    except ValueError as exc:
                        raise ValueError(f'{label}: {exc}') from None

        settings = LatentInversionSettings(**values)
        settings.validate()
        return settings

    def _on_accept(self) -> None:
        try:
            self.settings()
        except ValueError as exc:
            QMessageBox.warning(self, 'Invalid settings', str(exc))
            return
        self.accept()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

#: Fields copied out of a result dict and handed to the UI thread. Deliberately
#: explicit: a result also holds torch-adjacent objects and the full history,
#: and neither belongs on a Qt signal.
_SUMMARY_FIELDS = (
    'point_index',
    'latent_err',
    'min_ratio',
    'min_dist',
    'n_fragments',
    'n_isolated',
    'physical',
    'max_disp',
    'n_rounds_used',
    'time_s',
    'seed_unique_id',
    'seed_distance',
    'target_nn_dist',
    'target_n_within',
)


def summarize_result(result: dict) -> dict:
    """Reduce an inversion result to plain data plus the decoded structure."""
    summary = {key: result.get(key) for key in _SUMMARY_FIELDS}
    summary['target_latent'] = tuple(float(v) for v in result['target_latent'])
    summary['latent'] = tuple(float(v) for v in result['latent'])
    summary['atoms'] = result.get('annotated_atoms', result['atoms']).copy()
    return summary


class TraversalWorker(QThread):
    """Run a latent traversal off the UI thread.

    Emits :attr:`point_ready` for every decoded structure so the panel can draw
    it as it lands, rather than only at the end.
    """

    #: (completed, total, message)
    progress = Signal(int, int, str)
    #: one summarized result dict
    point_ready = Signal(object)
    #: (list of summaries, error string -- empty when the run succeeded)
    finished_signal = Signal(object, str)

    def __init__(
        self,
        *,
        waypoints,
        settings: LatentInversionSettings,
        descriptor_backend: str,
        descriptor_model: str,
        autoencoder_path: Path,
        latent_space,
        unique_ids,
        dataframe,
        parent=None,
    ):
        super().__init__(parent)
        self._waypoints = waypoints
        self._settings = settings
        self._backend = descriptor_backend
        self._model = descriptor_model
        self._autoencoder_path = Path(autoencoder_path)
        self._latent = latent_space
        self._unique_ids = unique_ids
        # Read-only contract: the panel's cached dataframe is shared rather
        # than re-loading a multi-hundred-megabyte .xz, so this must never be
        # mutated here.
        self._df = dataframe
        self._stop = threading.Event()
        self._results: list[dict] = []

    # --------------------------------------------------------------- control

    def cancel(self) -> None:
        """Ask the run to stop at the next check point."""
        self._stop.set()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def results(self) -> list[dict]:
        return list(self._results)

    # ------------------------------------------------------------------ run

    def run(self) -> None:  # noqa: C901 - one linear pipeline, guarded once
        try:
            import os

            import numpy as np
            import torch

            from atlas.active_learning.backends import get_backend
            from atlas.active_learning.extrapolation.latent_inversion import (
                SeedPool,
                sample_path,
                traverse_latent_path,
            )

            # Leave the UI thread some cores; L-BFGS will otherwise take every
            # one of them and the progress bar stops repainting.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

            self.progress.emit(0, self._settings.n_structures, 'Loading model...')

            # map_location so a model trained on a GPU still loads on the CPU
            # default (and vice versa); without it torch refuses to
            # deserialize CUDA tensors when no GPU is present.
            ae_model = torch.load(
                self._autoencoder_path,
                map_location=self._settings.device,
                weights_only=False,
            )
            mean_vals, std_vals = _load_standardization(self._autoencoder_path)

            handle = get_backend(self._backend).differentiable_descriptors(
                self._model,
                {'device': self._settings.device, 'dtype': self._settings.dtype},
            )

            if self._stop.is_set():
                self.finished_signal.emit([], '')
                return

            pool = SeedPool.from_dataframe(self._latent, self._unique_ids, self._df)
            targets = sample_path(
                self._waypoints,
                self._settings.n_structures,
                snap_waypoints=self._settings.snap_waypoints,
            )

            total = len(targets)
            self.progress.emit(0, total, f'Decoding {total} structures...')

            for result in traverse_latent_path(
                handle=handle,
                targets=np.asarray(targets, dtype=float),
                pool=pool,
                ae_model=ae_model,
                mean_vals=mean_vals,
                std_vals=std_vals,
                settings=self._settings,
                should_stop=self._stop.is_set,
            ):
                summary = summarize_result(result)
                self._results.append(summary)
                self.point_ready.emit(summary)
                done = len(self._results)
                self.progress.emit(done, total, f'Decoded {done} of {total}')

            self.finished_signal.emit(self.results(), '')
        except Exception:
            self.finished_signal.emit(self.results(), traceback.format_exc())


def _load_standardization(autoencoder_path: Path):
    """Load the mean/std arrays written next to *autoencoder_path*."""
    import numpy as np

    directory = Path(autoencoder_path).parent
    mean_path = directory / 'ae_mean_vals.npy'
    std_path = directory / 'ae_std_vals.npy'
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(
            'The autoencoder standardization arrays (ae_mean_vals.npy, '
            f'ae_std_vals.npy) were not found in {directory}. Re-run the '
            'boundary generation to produce them.'
        )
    return np.load(mean_path), np.load(std_path)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

_RESULT_COLUMNS = (
    ('#', lambda r: str(r['point_index'] + 1)),
    ('Target', lambda r: _fmt_point(r['target_latent'])),
    ('Achieved', lambda r: _fmt_point(r['latent'])),
    ('Latent err', lambda r: _fmt_float(r['latent_err'], '.3e')),
    ('Min ratio', lambda r: _fmt_float(r['min_ratio'], '.3f')),
    ('Fragments', lambda r: _fmt_int(r['n_fragments'])),
    ('Physical', lambda r: 'yes' if r['physical'] else 'no'),
    ('NN dist', lambda r: _fmt_float(r.get('target_nn_dist'), '.3f')),
)


def _fmt_point(point) -> str:
    return f'({point[0]:.3f}, {point[1]:.3f})'


def _fmt_float(value, spec: str) -> str:
    return '-' if value is None else format(float(value), spec)


def _fmt_int(value) -> str:
    return '-' if value is None else str(int(value))


class LatentTraversalResultsDialog(QDialog):
    """Decoded structures with their diagnostics, plus export and import."""

    structures_added = Signal(int)

    def __init__(self, results: list[dict], db_path: Path | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Latent traversal results')
        self.resize(1000, 620)

        self._results = list(results)
        self._db_path = Path(db_path) if db_path else None
        self._add_worker = None

        outer = QVBoxLayout(self)

        n_physical = sum(1 for r in self._results if r['physical'])
        header = QLabel(
            f'{len(self._results)} structures decoded, '
            f'{n_physical} physical. Non-physical results are kept and flagged '
            'rather than dropped, so a failed decode can be inspected.'
        )
        header.setWordWrap(True)
        outer.addWidget(header)

        self._table = QTableWidget(len(self._results), len(_RESULT_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _RESULT_COLUMNS])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        for row, result in enumerate(self._results):
            for col, (_name, getter) in enumerate(_RESULT_COLUMNS):
                item = QTableWidgetItem(getter(result))
                if not result['physical']:
                    item.setForeground(Qt.red)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        self._table.currentCellChanged.connect(self._on_row_changed)

        from atlas.core.gui.widgets.structure_viewer import StructureViewer

        self._viewer = StructureViewer()
        self._viewer.setMinimumWidth(300)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._table)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        self._status = QLabel('')
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        buttons = QHBoxLayout()
        self._add_btn = QPushButton('Add to initial DB')
        self._add_btn.clicked.connect(self._on_add_to_db)
        if self._db_path is None or not self._db_path.exists():
            self._add_btn.setEnabled(False)
            self._add_btn.setToolTip('The initial database could not be located.')
        export_btn = QPushButton('Export to file...')
        export_btn.clicked.connect(self._on_export)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(self._add_btn)
        buttons.addWidget(export_btn)
        buttons.addStretch()
        buttons.addWidget(close_btn)
        outer.addLayout(buttons)

        if self._results:
            self._table.setCurrentCell(0, 0)

    # ------------------------------------------------------------- viewing

    def _on_row_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int):
        if row < 0 or row >= len(self._results):
            return
        result = self._results[row]
        info = {
            'point': result['point_index'] + 1,
            'target': _fmt_point(result['target_latent']),
            'achieved': _fmt_point(result['latent']),
            'latent_err': _fmt_float(result['latent_err'], '.3e'),
            'min_ratio': _fmt_float(result['min_ratio'], '.3f'),
            'fragments': _fmt_int(result['n_fragments']),
            'physical': result['physical'],
            'seed': result.get('seed_unique_id') or 'previous decode',
        }
        self._viewer.set_atoms(result['atoms'], info)

    # ------------------------------------------------------------- actions

    def _on_export(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Export decoded structures',
            'decoded_structures.xyz',
            'Extended XYZ (*.xyz *.extxyz);;All files (*)',
        )
        if not path:
            return
        try:
            from ase.io import write

            # extxyz so the per-structure diagnostics in atoms.info survive the
            # round trip; a plain xyz would silently drop all of them.
            write(path, [r['atoms'] for r in self._results], format='extxyz')
        except Exception as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))
            return
        self._status.setText(f'Exported {len(self._results)} structures to {path}')

    def _on_add_to_db(self) -> None:
        if self._db_path is None or self._add_worker is not None:
            return
        reply = QMessageBox.question(
            self,
            'Add to initial database',
            f'Add {len(self._results)} decoded structures to the initial '
            'database as uncalculated entries?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._add_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText('Adding structures to the initial database...')

        self._add_worker = _AddToDbWorker(
            db_path=self._db_path,
            structures=[r['atoms'] for r in self._results],
            seed_ids=[r.get('seed_unique_id') for r in self._results],
            parent=self,
        )
        self._add_worker.finished_signal.connect(self._on_add_finished)
        self._add_worker.start()

    def _on_add_finished(self, added: int, error: str) -> None:
        self._progress.setVisible(False)
        self._add_worker = None
        if error:
            self._add_btn.setEnabled(True)
            self._status.setText('')
            QMessageBox.critical(self, 'Could not add structures', error)
            return
        self._status.setText(f'Added {added} structures to the initial database.')
        self.structures_added.emit(added)


class _AddToDbWorker(QThread):
    """Append decoded structures to the initial database."""

    #: (rows added, error string)
    finished_signal = Signal(int, str)

    def __init__(self, db_path: Path, structures, seed_ids, parent=None):
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._structures = list(structures)
        self._seed_ids = list(seed_ids)

    def run(self) -> None:
        try:
            import uuid

            import pandas as pd
            from pymatgen.io.ase import AseAtomsAdaptor

            from atlas.core.initial_db import InitialDatabase

            db = InitialDatabase.load_database(self._db_path)
            df = db.df
            if df is None or df.empty:
                self.finished_signal.emit(0, 'The initial database is empty.')
                return

            attrs = dict(df.attrs)
            id_col = 'atl_id' if 'atl_id' in df.columns else 'unique_id'
            by_id = {str(v): i for i, v in enumerate(df[id_col])}

            rows = []
            for atoms, seed_id in zip(self._structures, self._seed_ids, strict=True):
                # Copy a real row as the template rather than building one from
                # a hard-coded column list: the schema changes between db
                # versions, and a missing column breaks save/load much later.
                template_index = by_id.get(str(seed_id), 0)
                row = df.iloc[template_index].copy()
                row['structure'] = AseAtomsAdaptor.get_structure(atoms)
                row[id_col] = str(uuid.uuid4())
                if id_col != 'unique_id' and 'unique_id' in row.index:
                    row['unique_id'] = str(uuid.uuid4())
                _reset_calculation_fields(row)
                row['latent_decoded'] = True
                rows.append(row)

            new_df = pd.DataFrame(rows).reset_index(drop=True)
            db.df = pd.concat([df, new_df], ignore_index=True)
            # The marker column is new for a database that has never had a
            # traversal added, so backfill it rather than leaving NaN behind.
            db.df['latent_decoded'] = db.df['latent_decoded'].fillna(False)
            # pandas drops .attrs across concat, and db_version is read back on
            # load; losing it silently downgrades the database.
            db.df.attrs.update(attrs)
            db.database_name = self._db_path.stem
            db.save_database(path=str(self._db_path.parent))
            self.finished_signal.emit(len(rows), '')
        except Exception:
            self.finished_signal.emit(0, traceback.format_exc())


#: Columns that describe a completed calculation. A decoded structure has none
#: of them, and inheriting the seed's energy would make it look calculated.
_CALC_FIELDS = (
    'calc_performed',
    'calc_energy',
    'calc_energy_per_atom',
    'calc_energy_toten',
    'calc_output',
    'calc_type',
)


def _reset_calculation_fields(row) -> None:
    """Clear inherited calculation results on a copied row, in place."""
    for field_name in _CALC_FIELDS:
        if field_name not in row.index:
            continue
        row[field_name] = False if field_name == 'calc_performed' else None
