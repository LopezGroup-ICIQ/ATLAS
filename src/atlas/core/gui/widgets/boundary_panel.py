"""Interactive latent-space boundary panel.

Loads ``latent_space.npz`` (produced by ``descriptors_concave_hull``)
and renders an interactive scatter plot with the boundary overlay
(concave hull or morphological closing). The map is drawn by
:class:`~atlas.core.gui.widgets.latent_map_view.LatentMapView`, which puts
the points on the GPU so databases with millions of structures still pan
and zoom smoothly; drag pans, the wheel zooms, and hovering shows the
nearest structure.

The right-hand side is a switchable panel (see ``SidePanel``) with two
views: a 3D structure viewer for the clicked point, and the latent-space
traversal controls. Maps built with a differentiable descriptor can be
*inverted*: points marked on the plot are decoded back into structures by
latent-space traversal (see :mod:`~atlas.core.gui.widgets.latent_inversion`).

Falls back to a static PNG viewer for projects generated before the
``.npz`` format was introduced.

``ATL_GUI_LATENT_PLOT`` overrides the renderer: ``opengl`` (default),
``pyqtgraph`` (same view, points drawn on the CPU) or ``matplotlib`` (the
original canvas, also used when ``pyqtgraph`` is not installed).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.project import Project
from atlas.core.gui.widgets.latent_inversion import TraversalGate

LATENT_SPACE_FILE = 'latent_space.npz'
HULL_GLOBS = ['concave_hull*.png', 'comp_plot*.png', 'descriptors_concave_hull*.png']

_HOVER_RADIUS = 0.02
_CLICK_RADIUS = 0.03

#: Waypoint markers, the dashed path between them, and decoded structures.
#: Colours are chosen to stay distinct from the scatter (steelblue), the
#: boundary (red), and the hover/selection rings (orange/teal).
_WAYPOINT_COLOR = '#d81b60'
_DECODED_OK_COLOR = '#2e7d32'
_DECODED_BAD_COLOR = '#c62828'


def _latent_plot_backend() -> str:
    """``opengl``, ``pyqtgraph`` or ``matplotlib``, from ``ATL_GUI_LATENT_PLOT``.

    Unknown values select the default. A missing OpenGL driver is handled
    later by the view itself, which falls back to drawing on the CPU.
    """
    backend = os.environ.get('ATL_GUI_LATENT_PLOT', '').strip().lower()
    if backend == 'matplotlib':
        return backend
    try:
        import pyqtgraph  # noqa: F401
    except ImportError:
        return 'matplotlib'
    return 'pyqtgraph' if backend == 'pyqtgraph' else 'opengl'


class BoundaryPanel(QWidget):
    """Interactive latent-space scatter + hull boundary plot with structure detail."""

    #: The project's database-generation config was rewritten by the
    #: latent-space generation pane, so the Config tab should reload it.
    config_updated = Signal()
    #: A line of traversal or generation output for the page's log.
    log_message = Signal(str)
    #: No database to regenerate from; the page should point at the Config tab.
    database_missing = Signal()
    #: A traversal or regeneration is running, so the shared progress bar in
    #: the status bar should be visible.
    busy_changed = Signal(bool)
    #: ``(value, maximum, message)`` for the shared progress bar.
    progress_changed = Signal(int, int, str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project
        self._init_db_cache = None

        # Hover state
        self._hover_cid = None
        self._press_cid = None
        self._release_cid = None
        self._press_xy: tuple[float, float] | None = None
        self._hover_idx: int | None = None
        self._hover_highlight = None
        self._hover_annot = None
        self._hover_tree = None
        self._hover_ls = None
        self._hover_ids = None
        self._hover_ax = None

        # Selection state (click)
        self._sel_idx: int | None = None
        self._sel_highlight = None
        self._sel_annot = None

        # Traversal state. Kept strictly separate from the artists that draw
        # it: every redraw destroys the artists, and losing the marked path
        # with them would make the feature unusable next to pan and zoom.
        self._waypoints: list[tuple[float, float]] = []
        self._inverted: list[dict] = []
        self._mark_mode = False
        self._npz_path: Path | None = None
        self._npz_data: dict | None = None
        self._availability = TraversalGate(False, '', False)
        self._worker = None
        self._overlay_ax = None
        self._wp_line = None
        self._wp_labels: list = []
        self._path_line = None
        self._inv_ok_line = None
        self._inv_bad_line = None

        # --- Interactive plot widgets ---
        self._plot_container = QWidget()
        plot_layout = QVBoxLayout(self._plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._map_view = None
        self._fig = self._canvas = self._toolbar = None
        backend = _latent_plot_backend()
        if backend != 'matplotlib':
            from atlas.core.gui.widgets.latent_map_view import (
                LatentMapToolbar,
                LatentMapView,
            )

            self._map_view = LatentMapView(
                hover_radius=_HOVER_RADIUS, use_opengl=backend == 'opengl'
            )
            self._map_view.map_clicked.connect(self._on_map_clicked)
            self._toolbar = LatentMapToolbar(self._map_view, self)
            self._toolbar.clear_selection_requested.connect(self._clear_selection)
            plot_layout.addWidget(self._toolbar)
            plot_layout.addWidget(self._map_view, 1)
        else:
            self._fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
            self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            plot_layout.addWidget(self._toolbar)
            plot_layout.addWidget(self._canvas, 1)

        # --- Structure detail pane (shown in the side panel) ---
        from atlas.core.gui.widgets.structure_viewer import StructureViewer

        self._viewer = StructureViewer()
        self._viewer.setMinimumWidth(260)

        self._detail_placeholder = QLabel(
            'Click a point in the latent space\nto view its structure.'
        )
        self._detail_placeholder.setAlignment(Qt.AlignCenter)
        self._detail_placeholder.setWordWrap(True)
        self._detail_placeholder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._detail_placeholder.setStyleSheet('padding: 30px; font-size: 13px;')

        self._structure_pane = QWidget()
        self._structure_pane.setMinimumWidth(300)
        structure_layout = QVBoxLayout(self._structure_pane)
        structure_layout.setContentsMargins(0, 0, 0, 0)
        structure_layout.addWidget(self._viewer, 1)
        structure_layout.addWidget(self._detail_placeholder)
        self._viewer.setVisible(False)

        # --- Switchable side panel: Structure / Traversal / Generation ---
        from atlas.core.gui.widgets.latent_generation import LatentGenerationPane
        from atlas.core.gui.widgets.side_panel import SidePanel

        self._side_panel = SidePanel()
        self._structure_view_idx = self._side_panel.add_view(
            'science', 'Structure viewer', self._structure_pane
        )
        self._traversal_pane = self._build_traversal_pane()
        self._traversal_view_idx = self._side_panel.add_view(
            'route', 'Latent-space traversal', self._traversal_pane
        )
        self._generation_pane = LatentGenerationPane(project)
        self._generation_pane.generated.connect(self._on_latent_space_generated)
        self._generation_pane.config_updated.connect(self.config_updated.emit)
        self._generation_pane.log_message.connect(self.log_message.emit)
        self._generation_pane.database_missing.connect(self.database_missing.emit)
        self._generation_pane.busy_changed.connect(self._on_generation_busy)
        self._generation_view_idx = self._side_panel.add_view(
            'tune', 'Latent-space generation', self._generation_pane
        )
        self._side_panel.view_changed.connect(self._on_side_view_changed)
        self._side_panel.set_current_view(self._structure_view_idx)

        # Splitter: plot (left) + side panel (right). The explicit sizes give
        # the pane room from the start; minimum widths keep it usable when the
        # divider is dragged, since a scroll area would otherwise collapse.
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.addWidget(self._plot_container)
        self._main_splitter.addWidget(self._side_panel)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([900, 420])

        # --- Legacy image viewer (for old projects without .npz) ---
        self._legacy_container = QWidget()
        legacy_layout = QVBoxLayout(self._legacy_container)
        legacy_layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show_legacy_image)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet('')
        scroll = QScrollArea()
        scroll.setWidget(self._image_label)
        scroll.setWidgetResizable(True)

        legacy_splitter = QSplitter(Qt.Horizontal)
        legacy_splitter.addWidget(self._list)
        legacy_splitter.addWidget(scroll)
        legacy_splitter.setStretchFactor(0, 1)
        legacy_splitter.setStretchFactor(1, 4)
        legacy_layout.addWidget(legacy_splitter, 1)

        # --- Empty state ---
        self._empty_label = QLabel(
            'No boundary data found.\n'
            'Run database generation with boundary determination enabled\n'
            'to see the latent-space plot here.'
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet('padding: 40px;')

        # --- Main layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._main_splitter, 1)
        layout.addWidget(self._legacy_container, 1)
        layout.addWidget(self._empty_label)

        self._legacy_images: list[Path] = []

        # A running torch optimization would otherwise keep the process alive
        # after the window is gone.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_worker)

        self.refresh()

    # ---------------------------------------------------------- public API

    def refresh(self) -> None:
        # A reload mid-traversal would drop the results the running worker is
        # still producing, and one mid-regeneration could read a half-written
        # npz, so leave the panel alone until the work finishes.
        if self._worker is not None and self._worker.isRunning():
            return
        if self._generation_pane.is_running():
            return

        self._init_db_cache = None
        # Waypoints survive: they are latent coordinates and stay meaningful
        # across a redraw. Decoded structures do not, since a regenerated map
        # may place them somewhere else entirely.
        self._inverted = []

        npz_path = self._find_npz()
        if npz_path is not None:
            self._load_and_draw(npz_path)
            self._main_splitter.setVisible(True)
            self._legacy_container.setVisible(False)
            self._empty_label.setVisible(False)
            self._clear_selection()
            self._generation_pane.set_current_map(self._npz_data)
            self._generation_pane.refresh()
            self._update_traversal_controls()
            return

        self._npz_path = None
        self._npz_data = None
        self._availability = self._traversal_availability()
        self._generation_pane.set_current_map(None)
        self._generation_pane.refresh()
        self._update_traversal_controls()

        legacy = self._find_legacy_images()
        if legacy:
            self._show_legacy_list(legacy)
            self._main_splitter.setVisible(False)
            self._legacy_container.setVisible(True)
            self._empty_label.setVisible(False)
            return

        self._main_splitter.setVisible(False)
        self._legacy_container.setVisible(False)
        self._empty_label.setVisible(True)

    def has_content(self) -> bool:
        if self._find_npz() is not None:
            return True
        return bool(self._find_legacy_images())

    def set_theme(self, theme_name: str) -> None:
        """Recolour the side-panel icon strip for *theme_name*."""
        self._side_panel.set_theme(theme_name)
        if self._map_view is not None:
            from atlas.core.gui.themes import theme_colors

            colors = theme_colors(theme_name)
            self._map_view.apply_theme(colors)
            self._toolbar.apply_theme(colors)

    def reload_generation_settings(self) -> None:
        """Re-read the active config after the Config tab saved it."""
        self._generation_pane.load_from_config()

    # --------------------------------------------------------- npz loading

    def _find_npz(self) -> Path | None:
        for parent in (self._project.dir / 'databases', self._project.dir):
            p = parent / LATENT_SPACE_FILE
            if p.is_file():
                return p
        return None

    def _load_and_draw(self, npz_path: Path) -> None:
        with np.load(npz_path, allow_pickle=False) as loaded:
            # Materialised into a plain dict so the file handle does not stay
            # open for the lifetime of the panel; the arrays are small.
            data = dict(loaded)
        self._npz_path = npz_path
        self._npz_data = data
        ls = data['latent_space']
        unique_ids = data['unique_ids']

        method = 'concave_hull'
        if 'boundary_method' in data:
            method = str(data['boundary_method'][0])

        if method == 'morphological_closing':
            coords = data['boundary_coords']
            counts = data['boundary_counts']
            boundaries = np.split(coords, np.cumsum(counts)[:-1])
            disk_size = (
                int(data['morph_disk_size'][0]) if 'morph_disk_size' in data else 10
            )
            self._draw_morph(ls, boundaries, unique_ids, disk_size)
        else:
            hull = data['hull_vertices']
            alpha = float(data['alpha'][0])
            self._draw(ls, hull, unique_ids, alpha)

    # ------------------------------------------------------- info box placement

    @staticmethod
    def _best_corner(ax, ls: np.ndarray) -> tuple[float, float, str, str]:
        xmid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        ymid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2

        corners = [
            (ls[:, 0] < xmid) & (ls[:, 1] >= ymid),
            (ls[:, 0] >= xmid) & (ls[:, 1] >= ymid),
            (ls[:, 0] < xmid) & (ls[:, 1] < ymid),
            (ls[:, 0] >= xmid) & (ls[:, 1] < ymid),
        ]
        positions = [
            (0.03, 0.97, 'top', 'left'),
            (0.97, 0.97, 'top', 'right'),
            (0.03, 0.03, 'bottom', 'left'),
            (0.97, 0.03, 'bottom', 'right'),
        ]
        counts = [int(np.sum(mask)) for mask in corners]
        best = int(np.argmin(counts))
        x, y, va, ha = positions[best]
        return x, y, va, ha

    # --------------------------------------------------------- hover + click

    def _setup_interaction(self, ax, ls, unique_ids) -> None:
        from scipy.spatial import cKDTree

        self._hover_tree = cKDTree(ls)
        self._hover_ls = ls
        self._hover_ids = unique_ids
        self._hover_ax = ax
        self._hover_idx = None
        self._sel_idx = None

        # Hover ring (orange, no fill)
        (self._hover_highlight,) = ax.plot(
            [],
            [],
            'o',
            ms=10,
            mec='#e85d04',
            mfc='none',
            mew=2.0,
            zorder=5,
            visible=False,
        )

        # Hover tooltip
        self._hover_annot = ax.annotate(
            '',
            xy=(0, 0),
            xytext=(14, 14),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc=mpl.rcParams['figure.facecolor'],
                ec='#adb5bd',
                alpha=0.95,
                lw=0.8,
            ),
            arrowprops=dict(arrowstyle='->', color='#6c757d', lw=0.8),
            zorder=6,
            visible=False,
        )

        # Selection marker (teal, persistent until next click)
        (self._sel_highlight,) = ax.plot(
            [],
            [],
            'o',
            ms=12,
            mec='#2a9d8f',
            mfc='#2a9d8f44',
            mew=2.0,
            zorder=7,
            visible=False,
        )

        # Selection tooltip (persistent, same style but teal border)
        self._sel_annot = ax.annotate(
            '',
            xy=(0, 0),
            xytext=(14, 14),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc=mpl.rcParams['figure.facecolor'],
                ec='#2a9d8f',
                alpha=0.95,
                lw=1.0,
            ),
            arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=1.0),
            zorder=8,
            visible=False,
        )

        # Disconnect old handlers
        for cid in (self._hover_cid, self._press_cid, self._release_cid):
            if cid is not None:
                self._canvas.mpl_disconnect(cid)

        self._hover_cid = self._canvas.mpl_connect(
            'motion_notify_event',
            self._on_mouse_move,
        )
        self._press_cid = self._canvas.mpl_connect(
            'button_press_event',
            self._on_mouse_press,
        )
        self._release_cid = self._canvas.mpl_connect(
            'button_release_event',
            self._on_mouse_release,
        )

        def format_coord(x, y):
            dist, idx = self._hover_tree.query([x, y])
            uid = unique_ids[idx] if idx < len(unique_ids) else '?'
            return f'x={x:.4f}  y={y:.4f}  [{uid}]'

        ax.format_coord = format_coord

    def _annot_offset(self, x: float, y: float) -> tuple[float, float]:
        """Return xytext offset that keeps the tooltip inside the axes."""
        ax = self._hover_ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xfrac = (x - xlim[0]) / (xlim[1] - xlim[0])
        yfrac = (y - ylim[0]) / (ylim[1] - ylim[0])
        ox = -14 if xfrac > 0.75 else 14
        oy = -14 if yfrac > 0.80 else 14
        return ox, oy

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is not self._hover_ax or self._hover_tree is None:
            if self._hover_idx is not None:
                self._hover_highlight.set_visible(False)
                self._hover_annot.set_visible(False)
                self._hover_idx = None
                self._canvas.draw_idle()
            return

        dist, idx = self._hover_tree.query([event.xdata, event.ydata])

        xlim = self._hover_ax.get_xlim()
        ylim = self._hover_ax.get_ylim()
        diag = np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if dist > diag * _HOVER_RADIUS:
            if self._hover_idx is not None:
                self._hover_highlight.set_visible(False)
                self._hover_annot.set_visible(False)
                self._hover_idx = None
                self._canvas.draw_idle()
            return

        if idx == self._hover_idx:
            return

        self._hover_idx = idx
        x, y = self._hover_ls[idx]
        uid = self._hover_ids[idx] if idx < len(self._hover_ids) else '?'

        self._hover_highlight.set_data([x], [y])
        self._hover_highlight.set_visible(True)

        ox, oy = self._annot_offset(x, y)
        self._hover_annot.set_position((ox, oy))
        self._hover_annot.xy = (x, y)
        self._hover_annot.set_text(f'{uid}\nx={x:.4f}  y={y:.4f}')
        self._hover_annot.set_visible(True)

        self._canvas.draw_idle()

    def _on_mouse_press(self, event) -> None:
        if event.button == 1 and event.xdata is not None:
            self._press_xy = (event.x, event.y)
        else:
            self._press_xy = None

    def _on_mouse_release(self, event) -> None:
        if event.inaxes is not self._hover_ax or self._hover_tree is None:
            self._press_xy = None
            return

        # Right-click removes a waypoint, and never reaches the press/drag
        # bookkeeping below, which only tracks the left button.
        if self._mark_mode and event.button == 3:
            self._press_xy = None
            self._remove_nearest_waypoint(event.xdata, event.ydata)
            return

        if event.button != 1 or self._press_xy is None:
            return

        dx = event.x - self._press_xy[0]
        dy = event.y - self._press_xy[1]
        self._press_xy = None
        if np.hypot(dx, dy) > 5:
            return

        # A pan or zoom that happens to end within the drag tolerance is still
        # a navigation gesture; acting on it selected a structure every time
        # the user finished zooming.
        if self._toolbar.mode:
            return

        if self._mark_mode:
            self._add_waypoint(event.xdata, event.ydata)
            return

        dist, idx = self._hover_tree.query([event.xdata, event.ydata])

        xlim = self._hover_ax.get_xlim()
        ylim = self._hover_ax.get_ylim()
        diag = np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if dist > diag * _CLICK_RADIUS:
            return

        self._select_point(idx)

    def _on_map_clicked(self, x: float, y: float, button: int) -> None:
        """Click handler for the pyqtgraph view; mirrors ``_on_mouse_release``."""
        if self._mark_mode and button == Qt.RightButton.value:
            self._remove_nearest_waypoint(x, y)
            return
        if button != Qt.LeftButton.value:
            return
        if self._mark_mode:
            self._add_waypoint(x, y)
            return
        idx = self._map_view.nearest(x, y, _CLICK_RADIUS)
        if idx is not None:
            self._select_point(idx)

    def _select_point(self, idx: int) -> None:
        self._sel_idx = idx
        x, y = self._hover_ls[idx]
        uid = str(self._hover_ids[idx]) if idx < len(self._hover_ids) else '?'

        if self._map_view is not None:
            self._map_view.set_selected(idx)
        else:
            self._sel_highlight.set_data([x], [y])
            self._sel_highlight.set_visible(True)

            ox, oy = self._annot_offset(x, y)
            self._sel_annot.set_position((ox, oy))
            self._sel_annot.xy = (x, y)
            self._sel_annot.set_text(f'{uid}\nx={x:.4f}  y={y:.4f}')
            self._sel_annot.set_visible(True)

            self._canvas.draw_idle()

        self._load_structure_for_id(uid, x, y)
        # Selecting a structure is a viewing action, so bring the viewer to
        # the front rather than leaving the traversal pane on top of it.
        self._side_panel.set_current_view(self._structure_view_idx)

    def _clear_selection(self) -> None:
        self._sel_idx = None
        if self._sel_highlight is not None:
            self._sel_highlight.set_visible(False)
        if self._sel_annot is not None:
            self._sel_annot.set_visible(False)
        if self._map_view is not None:
            self._map_view.set_selected(None)
        self._viewer.clear()
        self._viewer.setVisible(False)
        self._detail_placeholder.setVisible(True)

    # ----------------------------------------------------- structure loading

    def _load_structure_for_id(self, atl_id: str, lx: float, ly: float) -> None:
        try:
            df = self._get_init_db()
            if df is None:
                self._show_detail_fallback(atl_id, lx, ly)
                return

            id_col = 'atl_id' if 'atl_id' in df.columns else 'unique_id'
            match = df[df[id_col] == atl_id]
            if match.empty:
                self._show_detail_fallback(atl_id, lx, ly)
                return

            row = match.iloc[0]
            struct = row.get('structure')
            if struct is None:
                self._show_detail_fallback(atl_id, lx, ly)
                return

            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = AseAtomsAdaptor.get_atoms(struct)
            info = {
                'atl_id': atl_id,
                'phase': row.get('phase', ''),
                'struct_type': row.get('struct_type', ''),
                'calc_energy': row.get('calc_energy'),
            }
            self._detail_placeholder.setVisible(False)
            self._viewer.setVisible(True)
            self._viewer.set_atoms(atoms, info)

        except Exception:
            self._show_detail_fallback(atl_id, lx, ly)

    def _show_detail_fallback(self, atl_id: str, lx: float, ly: float) -> None:
        self._viewer.setVisible(False)
        self._detail_placeholder.setVisible(True)
        self._detail_placeholder.setText(
            f'Structure: {atl_id}\n'
            f'Latent coords: ({lx:.4f}, {ly:.4f})\n\n'
            f'Could not load 3D structure.\n'
            f'The database may not be available.'
        )

    def _get_init_db(self):
        if self._init_db_cache is not None:
            return self._init_db_cache
        path = self._project.init_db_path()
        if not path.exists():
            return None
        from atlas.core.initial_db import InitialDatabase

        loaded = InitialDatabase.load_database(path)
        df = getattr(loaded, 'df', loaded)
        if df is None or getattr(df, 'empty', True):
            return None
        self._init_db_cache = df
        return df

    # ------------------------------------------------------ traversal UI

    def _build_traversal_pane(self) -> QWidget:
        """Build the traversal view shown in the right-hand side panel."""
        pane = QWidget()
        pane.setMinimumWidth(320)
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        hint = QLabel(
            'Mark points on the map to lay out a path, then decode the '
            'structures along it. Right-click removes the nearest marked '
            'point.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: palette(mid);')
        layout.addWidget(hint)

        self._mark_btn = QPushButton('Mark points')
        self._mark_btn.setCheckable(True)
        self._mark_btn.setToolTip(
            'While active, clicking the map marks a path instead of selecting '
            'a structure. Right-click removes the nearest marked point.'
        )
        self._mark_btn.toggled.connect(self._on_mark_toggled)
        layout.addWidget(self._mark_btn)

        self._clear_btn = QPushButton('Clear points')
        self._clear_btn.setToolTip('Remove every marked point from the path.')
        self._clear_btn.clicked.connect(self._on_clear_points)
        layout.addWidget(self._clear_btn)

        points_group = QGroupBox('Marked points')
        points_layout = QVBoxLayout(points_group)
        self._waypoint_list = QListWidget()
        self._waypoint_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._waypoint_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._waypoint_list.setMaximumHeight(140)
        self._waypoint_list.itemDoubleClicked.connect(self._on_waypoint_double_clicked)
        points_layout.addWidget(self._waypoint_list)
        self._remove_btn = QPushButton('Remove selected')
        self._remove_btn.setToolTip(
            'Remove the point selected in the list (a double click does the same).'
        )
        self._remove_btn.clicked.connect(self._on_remove_selected_waypoint)
        points_layout.addWidget(self._remove_btn)
        layout.addWidget(points_group)

        self._configure_btn = QPushButton('Configure traversal...')
        self._configure_btn.clicked.connect(self._on_configure)
        layout.addWidget(self._configure_btn)

        # A disabled QPushButton receives no hover events, so its own tooltip
        # never appears. The container stays enabled and carries the reason.
        self._run_container = QWidget()
        run_layout = QHBoxLayout(self._run_container)
        run_layout.setContentsMargins(0, 0, 0, 0)
        self._run_btn = QPushButton('Run traversal')
        self._run_btn.clicked.connect(self._on_run_traversal)
        run_layout.addWidget(self._run_btn)
        layout.addWidget(self._run_container)

        self._cancel_btn = QPushButton('Cancel')
        self._cancel_btn.clicked.connect(self._on_cancel_traversal)
        self._cancel_btn.setVisible(False)
        layout.addWidget(self._cancel_btn)

        self._traversal_reason = QLabel('')
        self._traversal_reason.setStyleSheet('color: palette(mid); font-size: 11px;')
        self._traversal_reason.setWordWrap(True)
        layout.addWidget(self._traversal_reason)

        self._recompute_btn = QPushButton('Recompute latent space...')
        self._recompute_btn.setToolTip(
            'Open the latent-space panel to rebuild this map with a '
            'differentiable descriptor.'
        )
        self._recompute_btn.clicked.connect(self._on_recompute_requested)
        self._recompute_btn.setVisible(False)
        layout.addWidget(self._recompute_btn)

        layout.addStretch()
        scroll.setWidget(container)
        pane_layout.addWidget(scroll)
        self._sync_waypoint_list()
        return pane

    def _traversal_availability(self):
        """Whether a traversal can run on the current map, and why not."""
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        db_available = False
        try:
            db_available = self._get_init_db() is not None
        except Exception:
            db_available = False
        return traversal_availability(self._npz_path, self._npz_data, db_available)

    def _update_traversal_controls(self) -> None:
        running = self._worker is not None and self._worker.isRunning()
        if not running:
            self._availability = self._traversal_availability()
        available, reason, fixable = self._availability
        generation_running = self._generation_pane.is_running()
        # Regeneration and traversal must not overlap: the map is rewritten
        # underneath the optimizer, and both are heavy torch jobs.
        self._generation_pane.set_traversal_running(running)
        idle = not running and not generation_running

        if self._npz_data is not None:
            self._side_panel.show_view(self._traversal_view_idx)
        else:
            # Without a map there is nothing to traverse, so do not leave a
            # pane whose Run button can never become enabled on display.
            if self._side_panel.current_view() == self._traversal_view_idx:
                self._side_panel.set_current_view(self._structure_view_idx)
            self._side_panel.hide_view(self._traversal_view_idx)

        self._mark_btn.setEnabled(available and idle)
        self._clear_btn.setEnabled(bool(self._waypoints) and idle)
        self._configure_btn.setEnabled(available and idle)
        self._run_btn.setEnabled(available and bool(self._waypoints) and idle)
        self._cancel_btn.setVisible(running)
        self._recompute_btn.setVisible(not available and fixable and idle)

        if not available:
            self._run_container.setToolTip(reason)
            self._traversal_reason.setText(reason)
            self._traversal_reason.setToolTip(reason)
        elif not self._waypoints:
            hint = 'Mark at least one point on the map to run a traversal.'
            self._run_container.setToolTip(hint)
            self._traversal_reason.setText(hint)
            self._traversal_reason.setToolTip('')
        else:
            count = len(self._waypoints)
            self._run_container.setToolTip('')
            self._traversal_reason.setText(
                f'{count} point{"s" if count != 1 else ""} marked'
            )
            self._traversal_reason.setToolTip('')

    def _on_side_view_changed(self, index: int) -> None:
        """Leaving the traversal view disarms marking; opening it warns.

        Marking changes what a click on the map does, and the toggle that
        explains it lives in the traversal pane. Leaving it armed while the
        structure viewer is on top would turn a viewing click into a silent
        waypoint.
        """
        if index != self._traversal_view_idx and self._mark_btn.isChecked():
            self._mark_btn.setChecked(False)

        if index == self._traversal_view_idx:
            # Re-evaluate so the warning describes the map on screen now, not
            # whatever the gate said when the panel was last refreshed.
            if not (self._worker is not None and self._worker.isRunning()):
                self._availability = self._traversal_availability()
            self._warn_traversal_unavailable()

    def _warn_traversal_unavailable(self) -> None:
        """Explain why the traversal cannot run, every time it is opened."""
        if self._availability.available:
            return
        if self._ask_recompute_or_close(self._availability.recompute_fixable):
            self._on_recompute_requested()

    def _ask_recompute_or_close(self, fixable: bool) -> bool:
        """Show the unavailable dialog; True when the user asks to regenerate."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle('Latent-space traversal unavailable')
        box.setText(self._availability.reason)
        recompute_btn = None
        if fixable:
            box.setInformativeText(
                'Regenerate the latent space with a differentiable descriptor '
                '(for example MACE) to enable inversion.'
            )
            recompute_btn = box.addButton(
                'Regenerate latent space', QMessageBox.ActionRole
            )
        close_btn = box.addButton('Close', QMessageBox.RejectRole)
        box.setDefaultButton(recompute_btn or close_btn)
        box.exec()
        return recompute_btn is not None and box.clickedButton() is recompute_btn

    def _on_recompute_requested(self) -> None:
        """Bring the latent-space generation view forward."""
        self._side_panel.set_current_view(self._generation_view_idx)

    # -------------------------------------------------------- overlays

    def _draw_overlays(self, ax) -> None:
        """Create the traversal artists on a freshly drawn axes.

        Called after every full redraw. The panel's waypoint and result state
        is the source of truth; these artists are rebuilt from it, which is
        what lets pan, zoom and refresh leave a marked path intact.
        """
        self._overlay_ax = ax
        self._wp_labels = []

        (self._path_line,) = ax.plot(
            [],
            [],
            '--',
            color=_WAYPOINT_COLOR,
            lw=1.2,
            alpha=0.7,
            zorder=3,
        )
        (self._wp_line,) = ax.plot(
            [],
            [],
            'X',
            ls='none',
            ms=11,
            color=_WAYPOINT_COLOR,
            mec='white',
            mew=1.0,
            zorder=9,
        )
        (self._inv_ok_line,) = ax.plot(
            [],
            [],
            '*',
            ls='none',
            ms=13,
            color=_DECODED_OK_COLOR,
            mec='white',
            mew=0.8,
            zorder=10,
        )
        (self._inv_bad_line,) = ax.plot(
            [],
            [],
            '*',
            ls='none',
            ms=13,
            color=_DECODED_BAD_COLOR,
            mec='white',
            mew=0.8,
            zorder=10,
        )

        self._sync_overlays(draw=False)

    def _sync_overlays(self, draw: bool = True) -> None:
        """Push the current waypoints and results onto the existing artists."""
        if self._map_view is not None:
            good, bad = self._decoded_latents()
            self._map_view.set_overlays(self._waypoints, good, bad)
            return
        if self._overlay_ax is None or self._wp_line is None:
            return

        waypoints = np.asarray(self._waypoints, dtype=float).reshape(-1, 2)
        self._wp_line.set_data(waypoints[:, 0], waypoints[:, 1])
        if len(waypoints) >= 2:
            self._path_line.set_data(waypoints[:, 0], waypoints[:, 1])
        else:
            self._path_line.set_data([], [])

        for label in self._wp_labels:
            # The axes may have been cleared underneath us, in which case the
            # artist is gone already and dropping the reference is all that is
            # left to do.
            with contextlib.suppress(ValueError, NotImplementedError):
                label.remove()
        self._wp_labels = [
            self._overlay_ax.annotate(
                str(i + 1),
                (x, y),
                xytext=(7, 7),
                textcoords='offset points',
                fontsize=8,
                color=_WAYPOINT_COLOR,
                zorder=9,
            )
            for i, (x, y) in enumerate(waypoints)
        ]

        good, bad = self._decoded_latents()
        self._inv_ok_line.set_data(good[:, 0], good[:, 1])
        self._inv_bad_line.set_data(bad[:, 0], bad[:, 1])

        if draw:
            self._canvas.draw_idle()

    def _decoded_latents(self) -> tuple[np.ndarray, np.ndarray]:
        """Latent coordinates of the physical and non-physical decodes."""
        good = np.array(
            [r['latent'] for r in self._inverted if r['physical']], dtype=float
        ).reshape(-1, 2)
        bad = np.array(
            [r['latent'] for r in self._inverted if not r['physical']], dtype=float
        ).reshape(-1, 2)
        return good, bad

    # -------------------------------------------------------- waypoints

    def _on_mark_toggled(self, checked: bool) -> None:
        self._mark_mode = bool(checked)
        if self._map_view is not None:
            self._map_view.set_mark_mode(self._mark_mode)
        elif checked:
            self._canvas.setCursor(Qt.CrossCursor)
        else:
            self._canvas.unsetCursor()

    def _add_waypoint(self, x, y) -> None:
        if x is None or y is None:
            return
        self._waypoints.append((float(x), float(y)))
        self._sync_overlays()
        self._sync_waypoint_list(scroll_to_bottom=True)
        self._update_traversal_controls()

    def _remove_nearest_waypoint(self, x, y) -> None:
        if not self._waypoints or x is None or y is None:
            return
        waypoints = np.asarray(self._waypoints, dtype=float)
        distances = np.hypot(waypoints[:, 0] - x, waypoints[:, 1] - y)
        index = int(np.argmin(distances))

        if self._map_view is not None:
            diag = self._map_view.view_diagonal()
        else:
            xlim = self._hover_ax.get_xlim()
            ylim = self._hover_ax.get_ylim()
            diag = np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if distances[index] > diag * _CLICK_RADIUS:
            return

        self._remove_waypoint_at(index)

    def _on_clear_points(self) -> None:
        self._waypoints = []
        self._inverted = []
        self._sync_overlays()
        self._sync_waypoint_list()
        self._update_traversal_controls()

    def _sync_waypoint_list(self, scroll_to_bottom: bool = False) -> None:
        """Mirror the marked path into the list widget."""
        self._waypoint_list.clear()
        for i, (x, y) in enumerate(self._waypoints):
            self._waypoint_list.addItem(f'{i + 1}   ({x:.4f}, {y:.4f})')
        has_points = bool(self._waypoints)
        self._remove_btn.setEnabled(has_points)
        self._waypoint_list.setEnabled(has_points)
        if has_points and scroll_to_bottom:
            self._waypoint_list.scrollToBottom()

    def _remove_waypoint_at(self, index: int) -> None:
        if index < 0 or index >= len(self._waypoints):
            return
        self._waypoints.pop(index)
        self._sync_overlays()
        self._sync_waypoint_list()
        self._update_traversal_controls()

    def _on_remove_selected_waypoint(self) -> None:
        self._remove_waypoint_at(self._waypoint_list.currentRow())

    def _on_waypoint_double_clicked(self, item) -> None:
        self._remove_waypoint_at(self._waypoint_list.row(item))

    # -------------------------------------------------------- traversal

    def _on_configure(self) -> bool:
        """Open the settings dialog; returns True when settings were saved."""
        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalSettingsDialog,
            load_settings,
            save_settings,
        )

        dialog = LatentTraversalSettingsDialog(
            load_settings(self._project),
            parent=self,
            allow_cell_relaxation=False,
            cell_disabled_reason=(
                'Cell relaxation is not implemented yet; inversion optimizes '
                'the atomic positions at a fixed cell.'
            ),
            min_structures=max(1, len(self._waypoints)),
        )
        # QDialog.Accepted lives on the metaclass in PySide6, so it has to be
        # reached through the class; ``dialog.Accepted`` raises AttributeError.
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        save_settings(self._project, dialog.settings())
        return True

    def _on_run_traversal(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._waypoints:
            return

        from atlas.core.gui.widgets.latent_inversion import (
            TraversalWorker,
            has_saved_settings,
            load_settings,
            npz_text,
            resolve_autoencoder_path,
        )

        # First run on this project: show what is about to happen rather than
        # silently applying defaults to a long optimization.
        if not has_saved_settings(self._project) and not self._on_configure():
            return

        settings = load_settings(self._project)
        # A traversal decodes at least one structure per marked point. Saved
        # settings can lag behind the path (more points marked since they were
        # configured), so reconfigure instead of failing inside the worker.
        if settings.n_structures < len(self._waypoints):
            if not self._on_configure():
                return
            settings = load_settings(self._project)
            if settings.n_structures < len(self._waypoints):
                QMessageBox.warning(
                    self,
                    'Traversal configuration',
                    f'{len(self._waypoints)} points are marked, so at least that '
                    'many structures are needed to cover the path.',
                )
                return

        df = self._get_init_db()
        if df is None:
            QMessageBox.warning(
                self,
                'Traversal unavailable',
                'The initial database could not be loaded, and inversion needs '
                'it to seed the optimization.',
            )
            return

        data = self._npz_data
        self._inverted = []
        self._sync_overlays()

        backend = npz_text(data, 'descriptor_type') or 'mace'
        model = npz_text(data, 'descriptor_model') or 'mace:mp-small'
        self.log_message.emit(
            f'Latent traversal started: decoding {settings.n_structures} '
            f'structure(s) along {len(self._waypoints)} marked point(s) with '
            f"the '{backend}' descriptor ({model}) on {settings.device}."
        )

        self._worker = TraversalWorker(
            waypoints=list(self._waypoints),
            settings=settings,
            descriptor_backend=backend,
            descriptor_model=model,
            autoencoder_path=resolve_autoencoder_path(self._npz_path, data),
            latent_space=data['latent_space'],
            unique_ids=data['unique_ids'],
            dataframe=df,
            parent=self,
        )
        self._worker.progress.connect(self._on_traversal_progress)
        self._worker.point_ready.connect(self._on_point_ready)
        self._worker.finished_signal.connect(self._on_traversal_finished)

        self.busy_changed.emit(True)
        self.progress_changed.emit(
            0, max(1, settings.n_structures), 'Starting traversal...'
        )
        self._worker.start()
        self._update_traversal_controls()

    def _on_traversal_progress(self, done: int, total: int, message: str) -> None:
        self.progress_changed.emit(done, max(1, total), message)
        self._traversal_reason.setText(message)

    def _on_point_ready(self, summary: dict) -> None:
        self._inverted.append(summary)
        self._sync_overlays()
        index = len(self._inverted)
        latent_err = summary.get('latent_err')
        err_text = f'{float(latent_err):.3e}' if latent_err is not None else 'n/a'
        state = 'physical' if summary.get('physical') else 'non-physical'
        self.log_message.emit(
            f'Decoded point {index}: latent error {err_text} ({state}, '
            f'{summary.get("n_fragments", "?")} fragment(s)).'
        )

    def _on_cancel_traversal(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._traversal_reason.setText('Cancelling...')
            self.log_message.emit('Latent traversal cancellation requested.')

    def _on_traversal_finished(self, results: list, error: str) -> None:
        cancelled = self._worker is not None and self._worker.cancelled
        self._worker = None
        self._cancel_btn.setEnabled(True)
        self.busy_changed.emit(False)
        self._update_traversal_controls()

        if error:
            last_line = error.strip().splitlines()[-1] if error.strip() else ''
            self.log_message.emit(f'Latent traversal failed: {last_line}')
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle('Traversal failed')
            box.setText('The latent traversal did not complete.')
            box.setDetailedText(error)
            box.exec()
            return

        if not results:
            message = 'Traversal cancelled.' if cancelled else 'No structures decoded.'
            self.log_message.emit(message)
            self._traversal_reason.setText(message)
            return

        self.log_message.emit(
            f'Latent traversal finished: {len(results)} structure(s) decoded.'
        )

        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalResultsDialog,
        )

        dialog = LatentTraversalResultsDialog(
            results,
            db_path=self._project.init_db_path(),
            parent=self,
        )
        dialog.structures_added.connect(self._on_structures_added)
        dialog.exec()

    def _on_structures_added(self, _count: int) -> None:
        # The cached dataframe is now stale, and it is what seeds the next run.
        self._init_db_cache = None

    def _on_generation_busy(self, busy: bool) -> None:
        if busy:
            # Regeneration reports progress through the log, so the shared bar
            # stays indeterminate for its duration.
            self.progress_changed.emit(0, 0, 'Generating latent space...')
        self.busy_changed.emit(busy)
        self._update_traversal_controls()

    def _on_latent_space_generated(self) -> None:
        # Reload the new map and re-evaluate the traversal gate. Waypoints are
        # kept; decoded markers are dropped by refresh().
        self.refresh()

    # ------------------------------------------------------------ lifecycle

    def _stop_worker(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        self._generation_pane.stop()

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self._stop_worker()
        super().closeEvent(event)

    def hideEvent(self, event):  # noqa: N802 - Qt naming
        # Switching to another tab hides the panel while its window stays up,
        # and killing a long optimization for that would be surprising. A
        # closing window hides it too, and that one does have to stop.
        window = self.window()
        if window is None or not window.isVisible():
            self._stop_worker()
        super().hideEvent(event)

    # ----------------------------------------------------------- plotting

    def _draw(
        self,
        ls: np.ndarray,
        hull: np.ndarray,
        unique_ids: np.ndarray,
        alpha: float,
    ) -> None:
        try:
            from shapely.geometry import Polygon

            hull_area = Polygon(hull).area
            area_str = f'\nHull area: {hull_area:.2e}'
        except Exception:
            area_str = ''

        info_text = f'Alpha: {alpha:.2f}{area_str}\nStructures: {len(ls):,}'

        if self._map_view is not None:
            self._show_on_map_view(ls, unique_ids, [hull], 'Concave hull', info_text)
            return

        self._fig.clear()
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        ax = self._fig.add_subplot(111)

        ax.scatter(
            ls[:, 0],
            ls[:, 1],
            s=4,
            c='steelblue',
            alpha=0.6,
            edgecolors='none',
            label=f'Structures ({len(ls):,})',
            rasterized=True,
            zorder=1,
        )

        ax.plot(
            hull[:, 0],
            hull[:, 1],
            '-',
            color='#cc241d',
            lw=2,
            label='Concave hull',
            zorder=2,
        )

        bx, by, va, ha = self._best_corner(ax, ls)
        ax.text(
            bx,
            by,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment=va,
            horizontalalignment=ha,
            bbox=dict(
                boxstyle='round', facecolor=mpl.rcParams['figure.facecolor'], alpha=0.85
            ),
            zorder=4,
        )

        ax.set_xlabel('Embedded dimension 1')
        ax.set_ylabel('Embedded dimension 2')
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._setup_interaction(ax, ls, unique_ids)
        self._draw_overlays(ax)

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def _draw_morph(
        self,
        ls: np.ndarray,
        boundaries: list[np.ndarray],
        unique_ids: np.ndarray,
        disk_size: int,
    ) -> None:
        area_str = ''
        try:
            from shapely.geometry import Polygon

            total_area = sum(Polygon(c).area for c in boundaries if len(c) >= 3)
            area_str = f'\nTotal area: {total_area:.2e}'
        except Exception:
            pass

        info_text = (
            f'Method: morphological closing\n'
            f'Disk size: {disk_size}{area_str}\n'
            f'Regions: {len(boundaries)}\n'
            f'Structures: {len(ls):,}'
        )

        if self._map_view is not None:
            closed = [np.vstack([c, c[:1]]) for c in boundaries if len(c) >= 2]
            self._show_on_map_view(ls, unique_ids, closed, 'Boundary', info_text)
            return

        self._fig.clear()
        self._fig.patch.set_facecolor(mpl.rcParams['figure.facecolor'])
        ax = self._fig.add_subplot(111)

        ax.scatter(
            ls[:, 0],
            ls[:, 1],
            s=4,
            c='steelblue',
            alpha=0.6,
            edgecolors='none',
            label=f'Structures ({len(ls):,})',
            rasterized=True,
            zorder=1,
        )

        for i, contour in enumerate(boundaries):
            if len(contour) < 2:
                continue
            closed = np.vstack([contour, contour[:1]])
            label = 'Boundary' if i == 0 else None
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                '-',
                color='#cc241d',
                lw=2,
                label=label,
                zorder=2,
            )

        bx, by, va, ha = self._best_corner(ax, ls)
        ax.text(
            bx,
            by,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment=va,
            horizontalalignment=ha,
            bbox=dict(
                boxstyle='round', facecolor=mpl.rcParams['figure.facecolor'], alpha=0.85
            ),
            zorder=4,
        )

        ax.set_xlabel('Embedded dimension 1')
        ax.set_ylabel('Embedded dimension 2')
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        self._setup_interaction(ax, ls, unique_ids)
        self._draw_overlays(ax)

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def _show_on_map_view(
        self,
        ls: np.ndarray,
        unique_ids: np.ndarray,
        boundaries: list[np.ndarray],
        boundary_label: str,
        info_text: str,
    ) -> None:
        """Draw the map with the pyqtgraph view instead of matplotlib."""
        self._hover_ls = ls
        self._hover_ids = unique_ids
        self._sel_idx = None
        self._map_view.set_map(ls, unique_ids, boundaries, boundary_label, info_text)
        self._sync_overlays(draw=False)

    # ------------------------------------------------ legacy image viewer

    def _find_legacy_images(self) -> list[Path]:
        images: list[Path] = []
        for parent in (self._project.dir / 'databases', self._project.dir):
            if parent.is_dir():
                for pattern in HULL_GLOBS:
                    for p in sorted(parent.glob(pattern)):
                        if p not in images:
                            images.append(p)
        return images

    def _show_legacy_list(self, images: list[Path]) -> None:
        self._list.clear()
        self._legacy_images = images
        self._image_label.clear()
        for img in images:
            self._list.addItem(img.name)
        self._list.setCurrentRow(0)

    def _show_legacy_image(self, row: int) -> None:
        if row < 0 or row >= len(self._legacy_images):
            return
        path = self._legacy_images[row]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._image_label.setText(f'Cannot load: {path.name}')
            return
        available = self._image_label.parent().size()
        scaled = pixmap.scaled(
            available,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
