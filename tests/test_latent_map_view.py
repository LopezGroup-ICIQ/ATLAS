"""Headless tests for the pyqtgraph/OpenGL latent map, its toolbar and wiring."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('pyqtgraph')

from PySide6.QtCore import Qt  # noqa: E402

from tests.test_boundary_panel import _write_npz  # noqa: E402

LEFT = Qt.LeftButton.value
RIGHT = Qt.RightButton.value


@pytest.fixture(scope='module')
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def view(qapp):
    from atlas.core.gui.widgets.latent_map_view import LatentMapView

    view = LatentMapView()
    view.resize(600, 400)
    ls = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    hull = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 2.0], [-1.0, -1.0]])
    view.set_map(ls, np.array(['a', 'b', 'c']), [hull], 'Concave hull', 'info')
    return view


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    from atlas.core.gui.project import Project
    from atlas.core.gui.widgets.boundary_panel import BoundaryPanel

    monkeypatch.delenv('ATL_GUI_LATENT_PLOT', raising=False)
    project = Project.create(tmp_path, 'demo')
    _write_npz(project.dir / 'databases')
    panel = BoundaryPanel(project)
    yield panel
    panel._stop_worker()


class TestLatentMapView:
    """The pyqtgraph map on its own: picking, boundaries, overlays."""

    def test_nearest_respects_radius(self, view):
        assert view.nearest(1.01, 0.99, 0.03) == 1
        # Far from every point relative to the view diagonal.
        assert view.nearest(1.0, 0.5, 0.01) is None

    def test_boundaries_are_joined_with_nan_breaks(self, qapp):
        from atlas.core.gui.widgets.latent_map_view import LatentMapView

        view = LatentMapView()
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        view.set_map(np.zeros((1, 2)), np.array(['a']), [square, square], 'B', '')
        x, _ = view._boundary.getData()
        assert len(x) == 8
        assert np.isnan(x[3]) and np.isnan(x[7])

    def test_overlays_and_labels(self, view):
        view.set_overlays([(0.0, 0.0), (1.0, 1.0)], np.zeros((1, 2)), np.zeros((0, 2)))
        assert len(view._waypoints.data) == 2
        assert len(view._path.getData()[0]) == 2
        assert [label.toPlainText() for label in view._wp_labels] == ['1', '2']

        view.set_overlays([(0.0, 0.0)], np.zeros((0, 2)), np.zeros((0, 2)))
        assert view._path.getData()[0] is None or len(view._path.getData()[0]) == 0
        assert len(view._wp_labels) == 1

    def test_selection_shows_and_clears(self, view):
        view.set_selected(2)
        assert view._sel_ring.isVisible()
        assert 'c' in view._sel_tip.toPlainText()
        view.set_selected(None)
        assert not view._sel_ring.isVisible()

    def test_overlays_do_not_change_auto_range(self, view):
        before = view._vb.childrenBounds()
        view.set_overlays([(100.0, 100.0)], np.zeros((0, 2)), np.zeros((0, 2)))
        assert view._vb.childrenBounds() == before


class TestOpenGLFallback:
    """The GL point layer handing over to the CPU scatter."""

    def test_falls_back_to_cpu_without_gl_context(self, qapp):
        from PySide6.QtGui import QImage, QPainter

        from atlas.core.gui.widgets.latent_map_view import LatentMapView

        view = LatentMapView(use_opengl=True)
        view.resize(400, 300)
        ls = np.random.default_rng(0).normal(size=(50, 2))
        view.set_map(ls, np.arange(50).astype(str), [], 'Boundary', '')
        assert view.uses_opengl
        assert len(view._scatter.data) == 0

        # A raster painter has no current GL context, like a software-only
        # display: the GL item must give up and hand the points to the CPU.
        image = QImage(400, 300, QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(image)
        view.render(painter)
        painter.end()
        for _ in range(3):
            qapp.processEvents()

        assert not view.uses_opengl
        assert len(view._scatter.data) == 50


class TestGLTransform:
    """Item -> NDC uniforms of the GL point layer, without a GL context."""

    class _Device:
        def __init__(self, width, height, dpr):
            self._size, self._dpr = (width, height), dpr

        def width(self):
            return self._size[0]

        def height(self):
            return self._size[1]

        def devicePixelRatioF(self):  # noqa: N802 - Qt naming
            return self._dpr

    class _Painter:
        """What QPainter's GL engine reports on a HiDPI ``QOpenGLWidget``."""

        def __init__(self, world, dpr, logical=(400, 300)):
            from PySide6.QtCore import QRect
            from PySide6.QtGui import QTransform

            self._world = world
            self._dpr = dpr
            self._logical = logical
            self._window = QRect(0, 0, int(logical[0] * dpr), int(logical[1] * dpr))
            self._scale = QTransform.fromScale(dpr, dpr)

        def transform(self):
            return self._world

        def combinedTransform(self):  # noqa: N802 - Qt naming
            return self._world * self._scale

        def deviceTransform(self):  # noqa: N802 - Qt naming
            return self._world * self._scale

        def window(self):
            return self._window

        def device(self):
            return TestGLTransform._Device(*self._logical, self._dpr)

    @pytest.mark.parametrize('dpr', [1.0, 1.25, 2.0])
    def test_points_land_where_qpainter_draws_them(self, qapp, dpr):
        from PySide6.QtGui import QTransform

        from atlas.core.gui.widgets.latent_map_gl import GLScatterItem

        # Data -> logical pixels: x scaled and shifted, y flipped.
        world = QTransform(50.0, 0.0, 0.0, -40.0, 120.0, 160.0)
        painter = self._Painter(world, dpr)
        lin, off, got_dpr, clip = GLScatterItem()._view_uniforms(painter)

        for x, y in [(0.0, 0.0), (1.5, -2.0), (-2.0, 3.0)]:
            px, py = world.map(x, y)
            want = (2 * px / 400 - 1, 1 - 2 * py / 300)
            ndc = (
                lin.x() * x + lin.y() * y + off.x(),
                lin.z() * x + lin.w() * y + off.y(),
            )
            assert ndc == pytest.approx(want, abs=1e-6)
        assert got_dpr == dpr
        # No ViewBox: the scissor box is the whole framebuffer.
        assert clip == (0, 0, int(400 * dpr), int(300 * dpr))


class TestLatentMapToolbar:
    """Toolbar buttons driving the view."""

    @pytest.fixture
    def bar(self, view):
        from atlas.core.gui.widgets.latent_map_view import LatentMapToolbar

        return LatentMapToolbar(view)

    @staticmethod
    def _width(view):
        (x0, x1), _ = view._vb.viewRange()
        return x1 - x0

    def test_zoom_and_reset(self, view, bar):
        start = self._width(view)
        bar.zoom_in_action.trigger()
        assert self._width(view) < start
        bar.reset_action.trigger()
        assert self._width(view) == pytest.approx(start)
        bar.zoom_out_action.trigger()
        assert self._width(view) > start

    def test_pan_and_box_modes_are_exclusive(self, view, bar):
        assert bar.pan_action.isChecked() and not view.box_zoom()
        bar.box_action.trigger()
        assert view.box_zoom() and not bar.pan_action.isChecked()
        bar.pan_action.trigger()
        assert not view.box_zoom()

    def test_aspect_lock(self, view, bar):
        bar.aspect_action.trigger()
        assert view._vb.state['aspectLocked'] == 1.0
        bar.aspect_action.trigger()
        assert not view._vb.state['aspectLocked']

    def test_selection_actions_follow_selection(self, view, bar):
        assert not bar.center_action.isEnabled()
        assert not bar.clear_action.isEnabled()
        view.set_selected(2)
        assert bar.center_action.isEnabled() and bar.clear_action.isEnabled()

        view._vb.setRange(xRange=(-5, -3), yRange=(-5, -3), padding=0)
        bar.center_action.trigger()
        (x0, x1), (y0, y1) = view._vb.viewRange()
        assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx((2.0, 0.0))
        assert x1 - x0 == pytest.approx(2.0)

        requested = []
        bar.clear_selection_requested.connect(lambda: requested.append(True))
        bar.clear_action.trigger()
        assert requested == [True]

    def test_point_size(self, view, bar):
        bar.size_spin.setValue(9)
        assert view.point_size == 9
        assert view._scatter.opts['size'] == 9

    def test_point_size_reaches_gl_layer(self, qapp):
        from atlas.core.gui.widgets.latent_map_view import LatentMapView

        view = LatentMapView(use_opengl=True)
        view.set_point_size(7)
        assert view._gl_points._size == 7

    def test_cursor_readout(self, view, bar):
        view.cursor_moved.emit(1.25, -0.5)
        assert bar.coords_label.text().strip() == 'x=1.2500  y=-0.5000'
        view.cursor_moved.emit(float('nan'), float('nan'))
        assert bar.coords_label.text() == ''

    def test_shortcuts_are_on_the_map(self, view, bar):
        from PySide6.QtGui import QKeySequence

        assert bar.reset_action in view.actions()
        keys = {k.toString() for k in bar.zoom_in_action.shortcuts()}
        assert QKeySequence('+').toString() in keys

    def test_save_image(self, view, bar, tmp_path):
        path = tmp_path / 'map.png'
        assert view.save_image(str(path))
        assert path.stat().st_size > 0

    def test_theme_recolours_icons(self, bar):
        before = bar.reset_action.icon().cacheKey()
        bar.apply_theme({'fg': '#ff0000'})
        assert bar.reset_action.icon().cacheKey() != before


class TestPanelWiring:
    """BoundaryPanel driving the map view instead of matplotlib."""

    def test_opengl_map_is_the_default(self, panel):
        assert panel._map_view is not None
        assert panel._map_view.uses_opengl
        assert panel._canvas is None

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('', 'opengl'),
            ('OpenGL', 'opengl'),
            ('pyqtgraph', 'pyqtgraph'),
            ('matplotlib', 'matplotlib'),
            ('nonsense', 'opengl'),
        ],
    )
    def test_env_var_selects_renderer(self, monkeypatch, value, expected):
        from atlas.core.gui.widgets.boundary_panel import _latent_plot_backend

        monkeypatch.setenv('ATL_GUI_LATENT_PLOT', value)
        assert _latent_plot_backend() == expected

    def test_matplotlib_without_pyqtgraph(self, monkeypatch):
        import sys

        from atlas.core.gui.widgets.boundary_panel import _latent_plot_backend

        monkeypatch.delenv('ATL_GUI_LATENT_PLOT', raising=False)
        monkeypatch.setitem(sys.modules, 'pyqtgraph', None)
        assert _latent_plot_backend() == 'matplotlib'

    def test_toolbar_sits_above_the_map(self, panel):
        from atlas.core.gui.widgets.latent_map_view import LatentMapToolbar

        assert isinstance(panel._toolbar, LatentMapToolbar)

    def test_toolbar_clears_panel_selection(self, panel):
        x, y = panel._npz_data['latent_space'][0]
        panel._on_map_clicked(float(x), float(y), LEFT)
        assert panel._toolbar.clear_action.isEnabled()
        panel._toolbar.clear_action.trigger()
        assert panel._sel_idx is None
        assert not panel._map_view._sel_ring.isVisible()
        assert not panel._toolbar.clear_action.isEnabled()

    def test_click_selects_structure(self, panel):
        x, y = panel._npz_data['latent_space'][0]
        panel._on_map_clicked(float(x), float(y), LEFT)
        assert panel._sel_idx == 0
        assert panel._map_view._sel_ring.isVisible()

    def test_mark_mode_adds_and_removes_waypoints(self, panel):
        panel._mark_btn.setChecked(True)
        panel._on_map_clicked(0.5, 0.25, LEFT)
        panel._on_map_clicked(-1.0, 1.0, LEFT)
        assert panel._waypoints == [(0.5, 0.25), (-1.0, 1.0)]
        assert len(panel._map_view._waypoints.data) == 2

        panel._on_map_clicked(0.5, 0.25, RIGHT)
        assert panel._waypoints == [(-1.0, 1.0)]

    def test_decoded_markers_follow_results(self, panel):
        panel._inverted = [
            {'latent': (0.0, 0.0), 'physical': True},
            {'latent': (1.0, 1.0), 'physical': False},
        ]
        panel._sync_overlays()
        assert len(panel._map_view._decoded_ok.data) == 1
        assert len(panel._map_view._decoded_bad.data) == 1

    def test_theme_is_forwarded(self, panel):
        from atlas.core.gui.themes import theme_colors, theme_names

        name = theme_names()[-1]
        panel.set_theme(name)
        assert panel._map_view.backgroundBrush().color().name() == (
            theme_colors(name)['axes_bg'].lower()
        )
