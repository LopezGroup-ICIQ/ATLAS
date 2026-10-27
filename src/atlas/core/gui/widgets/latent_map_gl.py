"""OpenGL point layer for the pyqtgraph latent map.

Used by :class:`~atlas.core.gui.widgets.latent_map_view.LatentMapView`.

``ScatterPlotItem`` maps every point to device coordinates in numpy and hands
QPainter one fragment per point on *every* frame, so its cost grows with the
number of structures whatever paints it; switching the viewport to OpenGL
alone barely helps. This item uploads the coordinates to a vertex buffer once
and sends only the 2D view transform per frame, so a pan or zoom costs the
same for ten thousand points as for millions.

Only the bulk scatter goes through here. Hover, selection, boundary and the
traversal overlays are few items and stay ordinary pyqtgraph items.

Needs the host ``GraphicsView`` to use an OpenGL viewport
(``view.useOpenGL(True)``); ``on_failure`` is called once if the shaders
cannot be built, so the caller can fall back to ``ScatterPlotItem``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QOpenGLContext, QVector2D, QVector4D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram

# GL enums, spelled out because PySide6 does not export them.
_GL_FLOAT = 0x1406
_GL_POINTS = 0x0000
_GL_BLEND = 0x0BE2
_GL_ONE = 0x0001
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_SCISSOR_TEST = 0x0C11
_GL_PROGRAM_POINT_SIZE = 0x8642
_GL_POINT_SPRITE = 0x8861

# GLSL 1.20 runs on the compatibility context QPainter's GL engine uses, on
# Mesa and on the proprietary drivers alike.
_VERTEX_SRC = """
#version 120
attribute vec2 pos;
uniform vec4 lin;   // (a, c, b, d): NDC = [[a c] [b d]] * pos + off
uniform vec2 off;
uniform float size;
void main() {
    gl_Position = vec4(lin.x * pos.x + lin.y * pos.y + off.x,
                       lin.z * pos.x + lin.w * pos.y + off.y, 0.0, 1.0);
    gl_PointSize = size;
}
"""

_FRAGMENT_SRC = """
#version 120
uniform vec4 color;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    gl_FragColor = vec4(color.rgb * color.a, color.a);  // premultiplied
}
"""


class GLScatterItem(pg.GraphicsObject):
    """Uniform-style scatter drawn with a single ``glDrawArrays`` call."""

    def __init__(
        self,
        size: float = 4.0,
        color: tuple[int, int, int, int] = (70, 130, 180, 153),
        on_failure: Callable[[str], None] | None = None,
    ):
        super().__init__()
        self._size = size
        self._color = QVector4D(*(c / 255 for c in color))
        self._on_failure = on_failure
        self._pos = np.empty((0, 2), dtype=np.float32)
        self._bounds = QRectF()
        self._dirty = True
        self._failed = False
        self._program: QOpenGLShaderProgram | None = None
        self._uniforms: dict[str, int] = {}
        self._vbo: QOpenGLBuffer | None = None
        self._context = None

    # ------------------------------------------------------------ data

    def setData(self, pos: np.ndarray) -> None:  # noqa: N802 - pyqtgraph naming
        self.prepareGeometryChange()
        self._pos = np.ascontiguousarray(pos, dtype=np.float32).reshape(-1, 2)
        if len(self._pos):
            lo, hi = self._pos.min(axis=0), self._pos.max(axis=0)
            self._bounds = QRectF(
                float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])
            )
        else:
            self._bounds = QRectF()
        self._dirty = True
        self.update()

    def setSize(self, size: float) -> None:  # noqa: N802 - pyqtgraph naming
        """Point diameter in logical pixels."""
        self._size = size
        self.update()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):  # noqa: N802, N803
        if not len(self._pos):
            return None
        col = self._pos[:, ax]
        return float(col.min()), float(col.max())

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        return self._bounds

    # ----------------------------------------------------------- paint

    def paint(self, painter, option, widget=None) -> None:
        if self._failed or not len(self._pos):
            return
        painter.beginNativePainting()
        try:
            context = QOpenGLContext.currentContext()
            if context is None:
                self._fail('no current OpenGL context (viewport is not OpenGL)')
                return
            if context is not self._context:
                # A new context (the widget was re-parented or the viewport
                # recreated) owns none of the previous GL objects.
                self._program = self._vbo = None
                self._context = context
                self._dirty = True
            if self._program is None and not self._build_program():
                return
            try:
                self._draw(context, painter)
            except Exception as exc:  # noqa: BLE001 - any GL error disables us
                # Leave the GL state as QPainter expects it before bailing out.
                if self._vbo is not None:
                    self._vbo.release()
                self._program.release()
                self._fail(f'OpenGL draw failed: {exc}')
        finally:
            painter.endNativePainting()

    def _build_program(self) -> bool:
        program = QOpenGLShaderProgram()
        ok = program.addShaderFromSourceCode(QOpenGLShader.Vertex, _VERTEX_SRC)
        ok = ok and program.addShaderFromSourceCode(
            QOpenGLShader.Fragment, _FRAGMENT_SRC
        )
        program.bindAttributeLocation('pos', 0)
        if not (ok and program.link()):
            self._fail(program.log() or 'shader compilation failed')
            return False
        self._program = program
        # Resolve locations once: the string overloads of setUniformValue are
        # ambiguous in PySide6 (a float can bind to the int overload).
        self._uniforms = {
            name: program.uniformLocation(name)
            for name in ('lin', 'off', 'size', 'color')
        }
        return True

    def _draw(self, context, painter) -> None:
        f = context.functions()
        program = self._program
        program.bind()

        if self._vbo is None:
            self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._vbo.create()
        self._vbo.bind()
        if self._dirty:
            data = self._pos.tobytes()
            self._vbo.allocate(data, len(data))
            self._dirty = False

        lin, off, dpr, clip = self._view_uniforms(painter)
        loc = self._uniforms
        program.setUniformValue(loc['lin'], lin)
        program.setUniformValue(loc['off'], off)
        program.setUniformValue1f(loc['size'], float(self._size * dpr))
        program.setUniformValue(loc['color'], self._color)

        # Native painting ignores QPainter's clip, so clip to the ViewBox by
        # hand or points would spill over the axes when zoomed in.
        f.glEnable(_GL_SCISSOR_TEST)
        f.glScissor(*clip)
        f.glEnable(_GL_BLEND)
        # Premultiplied blending, as QPainter's GL engine does; plain SRC_ALPHA
        # leaves a wrong destination alpha in framebuffers that have one.
        f.glBlendFunc(_GL_ONE, _GL_ONE_MINUS_SRC_ALPHA)
        f.glEnable(_GL_PROGRAM_POINT_SIZE)
        # Core contexts always have point sprites and reject the enum; the
        # error flag is cleared so it does not surface in QPainter later.
        f.glEnable(_GL_POINT_SPRITE)
        f.glGetError()

        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, _GL_FLOAT, 0, 2)
        f.glDrawArrays(_GL_POINTS, 0, len(self._pos))
        program.disableAttributeArray(0)

        f.glDisable(_GL_SCISSOR_TEST)
        self._vbo.release()
        program.release()

    def _view_uniforms(self, painter) -> tuple[QVector4D, QVector2D, float, tuple]:
        """Item -> NDC map, device pixel ratio and scissor box for *painter*.

        ``deviceTransform`` maps into the painter window, which QPainter's GL
        engine keeps in framebuffer (physical) pixels: on a HiDPI window both
        already include the device pixel ratio, while ``device.width()`` is
        logical. Mixing the two drew the points scaled by the ratio about the
        top-left corner, so every length here comes from the same window.
        The painter, not the widget, is used so offscreen renders work too.
        """
        t = painter.deviceTransform()
        window = painter.window()
        w, h = max(window.width(), 1), max(window.height(), 1)
        lin = QVector4D(
            2 * t.m11() / w, 2 * t.m21() / w, -2 * t.m12() / h, -2 * t.m22() / h
        )
        off = QVector2D(2 * t.dx() / w - 1, 1 - 2 * t.dy() / h)
        # glScissor counts from the bottom-left corner.
        rect = self._viewbox_device_rect(t, w, h)
        clip = (
            int(rect.left()),
            int(h - rect.bottom()),
            int(rect.width()),
            int(rect.height()),
        )
        return lin, off, painter.device().devicePixelRatioF(), clip

    def _viewbox_device_rect(self, t, w: int, h: int) -> QRectF:
        device_rect = QRectF(0, 0, w, h)
        vb = self.getViewBox()
        if vb is None:
            return device_rect
        rect = t.mapRect(self.mapRectFromItem(vb, vb.boundingRect()))
        return rect.intersected(device_rect)

    def _fail(self, reason: str) -> None:
        self._failed = True
        if self._on_failure is not None:
            self._on_failure(reason)
