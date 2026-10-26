"""GPU-friendly latent-space map drawn with pyqtgraph.

The map shown by :class:`~atlas.core.gui.widgets.boundary_panel.BoundaryPanel`,
replacing its original matplotlib canvas. matplotlib re-rasterizes the whole
figure through Agg on every pan, zoom and hover, which is what makes large
maps feel sluggish; pyqtgraph keeps every layer as a ``QGraphicsItem`` and
only repaints what changed, so a hover ring or a new waypoint costs a few
items instead of the full scatter.

The view owns drawing and hover only. What a click *means* (select a
structure, mark a waypoint, remove one) stays in the panel, which receives
:attr:`LatentMapView.map_clicked` and calls back into :meth:`nearest`,
:meth:`set_selected` and :meth:`set_overlays`.

With ``use_opengl`` the scatter is drawn by
:class:`~atlas.core.gui.widgets.latent_map_gl.GLScatterItem`, whose cost per
frame does not grow with the number of points; without it, or if OpenGL turns
out to be unavailable, a ``ScatterPlotItem`` draws them on the CPU.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QToolBar,
    QWidget,
)

#: Same palette as the matplotlib view, so switching backends changes the
#: feel and not the meaning of the colours.
_SCATTER_COLOR = (70, 130, 180, 153)  # steelblue, alpha 0.6
_BOUNDARY_COLOR = '#cc241d'
_HOVER_COLOR = '#e85d04'
_SELECT_COLOR = '#2a9d8f'
_WAYPOINT_COLOR = '#d81b60'
_DECODED_OK_COLOR = '#2e7d32'
_DECODED_BAD_COLOR = '#c62828'

#: Pixel gap between a marker and its tooltip.
_TIP_OFFSET_PX = 12

#: Default point diameter (logical pixels) and the toolbar's allowed range.
_POINT_SIZE = 4
_POINT_SIZE_RANGE = (1, 20)

#: View scale per zoom-in step; zoom out uses its inverse.
_ZOOM_STEP = 0.8


class LatentMapView(pg.PlotWidget):
    """Interactive scatter + boundary + traversal overlays."""

    #: ``(x, y, button)`` for a click that was not a drag. ``button`` is the
    #: integer value of the ``Qt.MouseButton``.
    map_clicked = Signal(float, float, int)

    #: Data coordinates under the cursor; NaNs once it leaves the plot area.
    cursor_moved = Signal(float, float)

    #: Whether a structure is selected, after every :meth:`set_selected`.
    selection_changed = Signal(bool)

    def __init__(
        self, parent=None, hover_radius: float = 0.02, use_opengl: bool = False
    ):
        super().__init__(parent)
        self._hover_radius = hover_radius
        self._ls: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._tree = None
        self._hover_idx: int | None = None
        self._sel_idx: int | None = None
        self._colors: dict[str, str] = {}
        self._point_size = _POINT_SIZE

        plot = self.getPlotItem()
        self._vb = plot.getViewBox()
        # Right-click belongs to the panel (removing waypoints); the context
        # menu would otherwise swallow it.
        plot.setMenuEnabled(False)
        self._vb.setMenuEnabled(False)
        # ``LatentMapToolbar`` replaces the small auto-range "A" button.
        plot.hideButtons()
        # Keyboard shortcuts of the toolbar need the map to take focus.
        self.setFocusPolicy(Qt.StrongFocus)
        plot.setLabel('bottom', 'Embedded dimension 1')
        plot.setLabel('left', 'Embedded dimension 2')
        plot.showGrid(x=False, y=False)
        self._legend = plot.addLegend(offset=(-10, -10))

        self._scatter = pg.ScatterPlotItem(
            size=_POINT_SIZE,
            pen=None,
            brush=pg.mkBrush(*_SCATTER_COLOR),
            pxMode=True,
        )
        self._boundary = pg.PlotDataItem(
            pen=pg.mkPen(_BOUNDARY_COLOR, width=2), connect='finite'
        )
        plot.addItem(self._scatter)
        plot.addItem(self._boundary)

        # With OpenGL the points live in a vertex buffer; ``_scatter`` stays
        # empty and only provides the legend sample.
        self._gl_points = None
        if use_opengl:
            from atlas.core.gui.widgets.latent_map_gl import GLScatterItem

            self.useOpenGL(True)
            self._gl_points = GLScatterItem(
                size=_POINT_SIZE, color=_SCATTER_COLOR, on_failure=self._on_gl_failure
            )
            plot.addItem(self._gl_points)

        # Traversal overlays: dashed path, waypoints, decoded structures.
        self._path = pg.PlotDataItem(
            pen=pg.mkPen(_WAYPOINT_COLOR, width=1.2, style=Qt.DashLine)
        )
        self._waypoints = pg.ScatterPlotItem(
            symbol='x',
            size=14,
            pen=pg.mkPen('w', width=1),
            brush=pg.mkBrush(_WAYPOINT_COLOR),
        )
        self._decoded_ok = pg.ScatterPlotItem(
            symbol='star',
            size=16,
            pen=pg.mkPen('w', width=0.8),
            brush=pg.mkBrush(_DECODED_OK_COLOR),
        )
        self._decoded_bad = pg.ScatterPlotItem(
            symbol='star',
            size=16,
            pen=pg.mkPen('w', width=0.8),
            brush=pg.mkBrush(_DECODED_BAD_COLOR),
        )
        self._wp_labels: list[pg.TextItem] = []

        # Hover and selection rings with their tooltips.
        self._hover_ring = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen(_HOVER_COLOR, width=2), brush=None
        )
        self._sel_ring = pg.ScatterPlotItem(
            size=14,
            pen=pg.mkPen(_SELECT_COLOR, width=2),
            brush=pg.mkBrush(42, 157, 143, 68),
        )
        self._hover_tip = pg.TextItem(border=pg.mkPen('#adb5bd'))
        self._sel_tip = pg.TextItem(border=pg.mkPen(_SELECT_COLOR, width=1))

        for z, item in enumerate(
            (
                self._path,
                self._waypoints,
                self._decoded_ok,
                self._decoded_bad,
                self._hover_ring,
                self._sel_ring,
                self._hover_tip,
                self._sel_tip,
            ),
            start=3,
        ):
            item.setZValue(z)
            # Overlays must not stretch the auto-range: a waypoint marked far
            # outside the data would otherwise zoom the map out.
            plot.addItem(item, ignoreBounds=True)
        for item in (self._hover_ring, self._sel_ring, self._hover_tip, self._sel_tip):
            item.setVisible(False)

        # The info box lives in the ViewBox's own coordinates, so it stays in
        # its corner while the data underneath pans and zooms.
        self._info = pg.TextItem(border=pg.mkPen('#adb5bd'), anchor=(0, 0))
        self._info.setParentItem(self._vb)
        self._info.setZValue(20)
        self._info_corner = (0, 0)
        self._vb.sigResized.connect(self._place_info)

        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        self.apply_theme({})

    # ------------------------------------------------------------ data

    def set_map(
        self,
        ls: np.ndarray,
        unique_ids: np.ndarray,
        boundaries: list[np.ndarray],
        boundary_label: str,
        info_text: str,
    ) -> None:
        """Replace the scatter, the boundary and the info box."""
        from scipy.spatial import cKDTree

        self._ls = np.asarray(ls, dtype=float)
        self._ids = unique_ids
        self._tree = cKDTree(self._ls)
        self._hover_idx = None

        self._legend.clear()
        if self._gl_points is not None:
            self._gl_points.setData(self._ls)
            self._scatter.setData(pos=np.empty((0, 2)))
        else:
            self._scatter.setData(pos=self._ls)
        self._legend.addItem(self._scatter, f'Structures ({len(self._ls):,})')

        # One item for every contour, separated by NaNs, keeps a map with many
        # regions as cheap to repaint as a map with one.
        segments = []
        for contour in boundaries:
            if len(contour) >= 2:
                segments.extend([contour, np.full((1, 2), np.nan)])
        joined = np.vstack(segments) if segments else np.empty((0, 2))
        self._boundary.setData(joined[:, 0], joined[:, 1])
        if len(joined):
            self._legend.addItem(self._boundary, boundary_label)

        self._info.setText(info_text)
        self._info_corner = self._best_corner()
        self.set_selected(None)
        self._hide_hover()
        self._vb.autoRange(padding=0.05)
        self._place_info()

    def set_overlays(
        self,
        waypoints: np.ndarray,
        decoded_ok: np.ndarray,
        decoded_bad: np.ndarray,
    ) -> None:
        """Show the marked path and the decoded structures."""
        waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 2)
        self._waypoints.setData(pos=waypoints)
        if len(waypoints) >= 2:
            self._path.setData(waypoints[:, 0], waypoints[:, 1])
        else:
            self._path.setData([], [])

        for label in self._wp_labels:
            self.getPlotItem().removeItem(label)
        self._wp_labels = []
        for i, (x, y) in enumerate(waypoints):
            # Anchors past [0, 1] push the label off the marker by a fraction
            # of its own size, which needs no update on zoom.
            label = pg.TextItem(str(i + 1), color=_WAYPOINT_COLOR, anchor=(-0.4, 1.2))
            label.setPos(x, y)
            label.setZValue(9)
            self.getPlotItem().addItem(label, ignoreBounds=True)
            self._wp_labels.append(label)

        self._decoded_ok.setData(pos=np.asarray(decoded_ok, dtype=float).reshape(-1, 2))
        self._decoded_bad.setData(
            pos=np.asarray(decoded_bad, dtype=float).reshape(-1, 2)
        )

    def set_selected(self, idx: int | None) -> None:
        """Ring and label the structure at *idx*, or clear the selection."""
        self._sel_idx = idx
        self.selection_changed.emit(idx is not None and self._ls is not None)
        if idx is None or self._ls is None:
            self._sel_ring.setVisible(False)
            self._sel_tip.setVisible(False)
            return
        x, y = self._ls[idx]
        self._sel_ring.setData(pos=[(x, y)])
        self._sel_ring.setVisible(True)
        self._place_tip(self._sel_tip, x, y, self._tip_text(idx))

    def set_point_size(self, size: float) -> None:
        """Diameter of the structure points in logical pixels."""
        self._point_size = size
        self._scatter.setSize(size)
        if self._gl_points is not None:
            self._gl_points.setSize(size)

    # -------------------------------------------------------- navigation

    def reset_view(self) -> None:
        """Fit the view to the structures and the boundary."""
        self._vb.autoRange(padding=0.05)

    def zoom(self, factor: float) -> None:
        """Scale the visible range by *factor* (< 1 zooms in) about its centre."""
        self._vb.scaleBy((factor, factor))

    def set_box_zoom(self, enabled: bool) -> None:
        """Left drag draws a zoom box instead of panning.

        Middle drag still pans and the wheel still zooms in either mode, and
        a click that is not a drag still selects or marks.
        """
        self._vb.setMouseMode(pg.ViewBox.RectMode if enabled else pg.ViewBox.PanMode)

    def box_zoom(self) -> bool:
        return self._vb.state['mouseMode'] == pg.ViewBox.RectMode

    def set_aspect_locked(self, locked: bool) -> None:
        """Keep one unit the same length on both axes."""
        self._vb.setAspectLocked(locked, ratio=1.0)

    def center_on_selection(self) -> bool:
        """Pan, without zooming, so the selected structure is centred."""
        if self._sel_idx is None or self._ls is None:
            return False
        x, y = self._ls[self._sel_idx]
        (x0, x1), (y0, y1) = self._vb.viewRange()
        self._vb.translateBy(x=x - (x0 + x1) / 2, y=y - (y0 + y1) / 2)
        return True

    def save_image(self, path: str) -> bool:
        """Write the map as shown to *path*; the format follows the suffix.

        Grabs the widget rather than going through a pyqtgraph exporter: an
        exporter paints into a raster image, where the OpenGL points cannot
        draw and would switch the map to the CPU for good.
        """
        return self.grab().save(path)

    def set_mark_mode(self, enabled: bool) -> None:
        if enabled:
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()

    # --------------------------------------------------------- queries

    def view_diagonal(self) -> float:
        (x0, x1), (y0, y1) = self._vb.viewRange()
        return float(np.hypot(x1 - x0, y1 - y0))

    def nearest(self, x: float, y: float, radius: float) -> int | None:
        """Index of the closest structure within *radius* of the view diagonal."""
        if self._tree is None:
            return None
        dist, idx = self._tree.query([x, y])
        if dist > self.view_diagonal() * radius:
            return None
        return int(idx)

    # ----------------------------------------------------------- theme

    def apply_theme(self, colors: dict[str, str]) -> None:
        """Recolour background, axes and boxes from ``theme_colors()``."""
        self._colors = colors
        bg = colors.get('axes_bg', '#ffffff')
        fg = colors.get('fg', '#1a1a2e')
        border = colors.get('border', '#adb5bd')
        self.setBackground(bg)
        plot = self.getPlotItem()
        for name in ('bottom', 'left'):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(colors.get('muted', '#888888')))
            axis.setTextPen(pg.mkPen(fg))
            axis.label.setDefaultTextColor(pg.mkColor(fg))
        self._legend.setBrush(pg.mkBrush(bg))
        self._legend.setPen(pg.mkPen(border))
        self._legend.setLabelTextColor(fg)
        for tip in (self._hover_tip, self._sel_tip, self._info):
            tip.fill = pg.mkBrush(bg)
            tip.setColor(fg)

    @property
    def uses_opengl(self) -> bool:
        return self._gl_points is not None

    @property
    def point_size(self) -> float:
        return self._point_size

    @property
    def selected(self) -> int | None:
        return self._sel_idx

    # ------------------------------------------------------- internals

    def _on_gl_failure(self, reason: str) -> None:
        logging.getLogger('atl').warning(
            'OpenGL latent map unavailable (%s); drawing points on the CPU.', reason
        )
        # Called from inside a paint; changing the scene there is not allowed.
        QTimer.singleShot(0, self._fall_back_to_cpu)

    def _fall_back_to_cpu(self) -> None:
        if self._gl_points is None:
            return
        self.getPlotItem().removeItem(self._gl_points)
        self._gl_points = None
        if self._ls is not None:
            self._scatter.setData(pos=self._ls)

    def _tip_text(self, idx: int) -> str:
        x, y = self._ls[idx]
        uid = self._ids[idx] if self._ids is not None and idx < len(self._ids) else '?'
        return f'{uid}\nx={x:.4f}  y={y:.4f}'

    def _place_tip(self, tip: pg.TextItem, x: float, y: float, text: str) -> None:
        """Put *tip* next to ``(x, y)``, flipping sides near the view edges."""
        tip.setText(text)
        tip.setPos(x, y)
        rect = tip.boundingRect()
        w = max(rect.width(), 1.0)
        h = max(rect.height(), 1.0)
        (x0, x1), (y0, y1) = self._vb.viewRange()
        xfrac = (x - x0) / (x1 - x0) if x1 != x0 else 0.5
        yfrac = (y - y0) / (y1 - y0) if y1 != y0 else 0.5
        ax = 1 + _TIP_OFFSET_PX / w if xfrac > 0.75 else -_TIP_OFFSET_PX / w
        ay = -_TIP_OFFSET_PX / h if yfrac > 0.80 else 1 + _TIP_OFFSET_PX / h
        tip.setAnchor((ax, ay))
        tip.setVisible(True)

    def _hide_hover(self) -> None:
        self._hover_idx = None
        self._hover_ring.setVisible(False)
        self._hover_tip.setVisible(False)

    def _best_corner(self) -> tuple[int, int]:
        """The ViewBox corner with the fewest points under it."""
        if self._ls is None or not len(self._ls):
            return (0, 0)
        mid = (self._ls.min(axis=0) + self._ls.max(axis=0)) / 2
        left = self._ls[:, 0] < mid[0]
        top = self._ls[:, 1] >= mid[1]
        counts = {
            (0, 0): np.sum(left & top),
            (1, 0): np.sum(~left & top),
            (0, 1): np.sum(left & ~top),
            (1, 1): np.sum(~left & ~top),
        }
        # Lower right holds the legend.
        counts[(1, 1)] = np.inf
        return min(counts, key=counts.get)

    def _place_info(self) -> None:
        rect = self._vb.boundingRect()
        cx, cy = self._info_corner
        margin = 8
        x = rect.right() - margin if cx else rect.left() + margin
        y = rect.bottom() - margin if cy else rect.top() + margin
        self._info.setAnchor((cx, cy))
        self._info.setPos(x, y)

    def _in_view(self, scene_pos) -> bool:
        return self._vb.sceneBoundingRect().contains(scene_pos)

    def _on_mouse_moved(self, scene_pos) -> None:
        if not self._in_view(scene_pos):
            self.cursor_moved.emit(float('nan'), float('nan'))
            if self._hover_idx is not None:
                self._hide_hover()
            return
        point = self._vb.mapSceneToView(scene_pos)
        self.cursor_moved.emit(point.x(), point.y())
        if self._tree is None:
            return
        idx = self.nearest(point.x(), point.y(), self._hover_radius)
        if idx is None:
            if self._hover_idx is not None:
                self._hide_hover()
            return
        if idx == self._hover_idx:
            return
        self._hover_idx = idx
        x, y = self._ls[idx]
        self._hover_ring.setData(pos=[(x, y)])
        self._hover_ring.setVisible(True)
        self._place_tip(self._hover_tip, x, y, self._tip_text(idx))

    def _on_mouse_clicked(self, event) -> None:
        # pyqtgraph only reports a click when the press did not turn into a
        # drag, so a pan never selects anything. The second half of a double
        # click is dropped to keep one click, one action.
        if event.double() or not self._in_view(event.scenePos()):
            return
        point = self._vb.mapSceneToView(event.scenePos())
        self.map_clicked.emit(point.x(), point.y(), event.button().value)


class LatentMapToolbar(QToolBar):
    """Navigation controls for a :class:`LatentMapView`.

    Reset, step zoom, pan/box-zoom mode, aspect lock, selection, point size,
    image export and a cursor readout. The shortcuts work while the map or
    the toolbar has focus. Clearing the selection is only requested through
    :attr:`clear_selection_requested`, because the panel owns what a
    selection means (the structure viewer, its side pane).
    """

    clear_selection_requested = Signal()

    def __init__(self, view: LatentMapView, parent: QWidget | None = None):
        super().__init__('Latent map', parent)
        self._view = view
        self.setMovable(False)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)

        self._icons: dict[QAction, str] = {}
        self.reset_action = self._action(
            'home', 'Reset view', ('R', 'Home'), view.reset_view
        )
        self.zoom_in_action = self._action(
            'zoom_in', 'Zoom in', ('+', '='), lambda: view.zoom(_ZOOM_STEP)
        )
        self.zoom_out_action = self._action(
            'zoom_out', 'Zoom out', ('-',), lambda: view.zoom(1 / _ZOOM_STEP)
        )
        self.addSeparator()

        self.pan_action = self._action(
            'pan', 'Pan: drag to move the map', ('P',), checkable=True
        )
        self.box_action = self._action(
            'box_zoom',
            'Box zoom: drag a rectangle to zoom into it (middle drag pans)',
            ('B',),
            checkable=True,
        )
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        for action in (self.pan_action, self.box_action):
            mode_group.addAction(action)
        self.pan_action.setChecked(True)
        self.box_action.toggled.connect(view.set_box_zoom)

        self.aspect_action = self._action(
            'aspect_ratio',
            'Lock aspect ratio: one unit is as long on both axes',
            ('L',),
            checkable=True,
        )
        self.aspect_action.toggled.connect(view.set_aspect_locked)
        self.addSeparator()

        self.center_action = self._action(
            'center_focus',
            'Centre on the selected structure',
            ('C',),
            view.center_on_selection,
        )
        self.clear_action = self._action(
            'close',
            'Clear selection',
            ('Esc',),
            self.clear_selection_requested.emit,
        )
        self._on_selection_changed(view.selected is not None)
        view.selection_changed.connect(self._on_selection_changed)
        self.addSeparator()

        self.size_spin = QSpinBox()
        self.size_spin.setRange(*_POINT_SIZE_RANGE)
        self.size_spin.setValue(int(view.point_size))
        self.size_spin.setSuffix(' px')
        self.size_spin.setToolTip('Point size')
        self.size_spin.valueChanged.connect(view.set_point_size)
        self.addWidget(QLabel(' Points '))
        self.addWidget(self.size_spin)
        self.save_action = self._action(
            'save', 'Save the map as an image', (), self._save_image
        )

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
        self.coords_label = QLabel()
        self.coords_label.setToolTip('Map coordinates under the cursor')
        self.addWidget(self.coords_label)
        view.cursor_moved.connect(self._on_cursor_moved)

        self.apply_theme({})

    def apply_theme(self, colors: dict[str, str]) -> None:
        """Recolour the icons from ``theme_colors()``."""
        from atlas.core.gui.icons import themed_icon

        fg = colors.get('fg', '#1a1a2e')
        for action, name in self._icons.items():
            action.setIcon(themed_icon(name, fg, size=20))

    def _action(
        self,
        icon: str,
        tip: str,
        keys: tuple[str, ...],
        slot=None,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(tip.split(':')[0], self)
        action.setCheckable(checkable)
        action.setShortcuts([QKeySequence(key) for key in keys])
        action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        if keys:
            tip = f'{tip} ({", ".join(keys)})'
        action.setToolTip(tip)
        if slot is not None:
            action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        # Also on the map, so the shortcuts fire while it has focus.
        self._view.addAction(action)
        self._icons[action] = icon
        return action

    def _on_selection_changed(self, selected: bool) -> None:
        self.center_action.setEnabled(selected)
        self.clear_action.setEnabled(selected)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        if np.isnan(x) or np.isnan(y):
            self.coords_label.clear()
        else:
            self.coords_label.setText(f'x={x:.4f}  y={y:.4f} ')

    def _save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save latent map', 'latent_map.png', 'Images (*.png *.jpg *.bmp)'
        )
        if path and not self._view.save_image(path):
            logging.getLogger('atl').warning('Could not save the map to %s', path)
