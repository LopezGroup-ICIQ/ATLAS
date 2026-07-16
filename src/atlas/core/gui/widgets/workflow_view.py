"""Interactive workflow diagram for ATLAS pipeline visualisation.

``WorkflowView`` renders a top-down flowchart of pipeline steps.  Each
step box shows its name, an optional structure-count estimate, and is
coloured by category.  Clicking a step emits the ``step_clicked`` signal so
the host can navigate to the relevant configuration fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from atlas.core.gui.themes import saved_global_theme, theme_colors

# ── data model ────────────────────────────────────────────────────────


@dataclass
class WorkflowStep:
    """One step in the pipeline diagram."""

    name: str
    description: str = ''
    estimated_count: int | None = None
    is_active: bool = True
    config_key: str = ''
    category: str = 'generation'
    is_filter: bool = False
    group: str = ''


# ── layout constants ──────────────────────────────────────────────────

_MARGIN_X = 16
_GROUP_INDENT = 24
_STEP_HEIGHT = 56
_STEP_RADIUS = 8
_ARROW_GAP = 22
_ARROW_HEAD = 6
_GROUP_PAD_TOP = 28
_GROUP_PAD_BOTTOM = 12
_BADGE_H = 20
_BADGE_PAD_X = 8
_BADGE_RADIUS = 10
_TOTAL_HEIGHT = 44
_TOP_PAD = 12
_BOTTOM_PAD = 24

# ── canvas (the paintable inner widget) ───────────────────────────────


class _WorkflowCanvas(QWidget):
    """Inner widget drawn by ``QPainter``; lives inside a scroll area."""

    step_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._steps: list[WorkflowStep] = []
        self._num_phases: int = 1
        self._total_estimate: int | None = None
        self._step_rects: list[tuple[QRectF, WorkflowStep]] = []
        self._hovered_index: int = -1
        self._theme: str = saved_global_theme()
        self.setMouseTracking(True)

    # ── public API ────────────────────────────────────────────────

    def set_data(
        self,
        steps: Sequence[WorkflowStep],
        num_phases: int = 1,
        total_estimate: int | None = None,
    ) -> None:
        self._steps = list(steps)
        self._num_phases = num_phases
        self._total_estimate = total_estimate
        self._step_rects.clear()
        self.setMinimumHeight(self._compute_height())
        self.update()

    def set_theme(self, theme_name: str) -> None:
        self._theme = theme_name
        self.update()

    # ── geometry ──────────────────────────────────────────────────

    def _compute_height(self) -> int:
        if not self._steps:
            return 120
        h = _TOP_PAD
        prev_group: str = ''
        for i, step in enumerate(self._steps):
            if prev_group and step.group != prev_group:
                h += _GROUP_PAD_BOTTOM
            if step.group and step.group != prev_group:
                h += _GROUP_PAD_TOP
            h += _STEP_HEIGHT
            if i < len(self._steps) - 1:
                h += _ARROW_GAP
            prev_group = step.group
        if prev_group:
            h += _GROUP_PAD_BOTTOM
        h += _ARROW_GAP + _TOTAL_HEIGHT + _BOTTOM_PAD
        return h

    # ── painting ──────────────────────────────────────────────────

    @staticmethod
    def _safe_pt(font: QFont) -> int:
        pt = font.pointSize()
        if pt > 0:
            return pt
        px = font.pixelSize()
        if px > 0:
            return max(round(px * 0.75), 8)
        return 9

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        colors = theme_colors(self._theme)
        fg = QColor(colors['fg'])
        bg = QColor(colors['bg'])
        primary = QColor(colors['primary'])
        surface = QColor(colors['surface'])
        muted = QColor(colors['muted'])
        border_c = QColor(colors['border'])

        painter.fillRect(self.rect(), bg)

        self._step_rects.clear()
        y = float(_TOP_PAD)
        prev_group: str = ''
        group_start_y = 0.0

        for i, step in enumerate(self._steps):
            leaving_group = prev_group and step.group != prev_group
            entering_group = step.group and step.group != prev_group

            if leaving_group:
                self._draw_group_bracket(
                    painter,
                    group_start_y,
                    y - _ARROW_GAP,
                    prev_group,
                    fg,
                    border_c,
                    surface,
                )
                y += _GROUP_PAD_BOTTOM

            if entering_group:
                group_start_y = y
                y += _GROUP_PAD_TOP

            indent = _GROUP_INDENT if step.group else 0
            x = float(_MARGIN_X + indent)
            w = float(self.width() - 2 * _MARGIN_X - 2 * indent)
            w = max(w, 100)
            rect = QRectF(x, y, w, _STEP_HEIGHT)

            self._draw_step_box(
                painter, rect, step, i, fg, bg, primary, surface, muted, border_c
            )
            self._step_rects.append((rect, step))
            y += _STEP_HEIGHT

            if i < len(self._steps) - 1:
                cx = x + w / 2
                self._draw_arrow(painter, cx, y, cx, y + _ARROW_GAP, muted)
                y += _ARROW_GAP

            prev_group = step.group

        if prev_group:
            self._draw_group_bracket(
                painter,
                group_start_y,
                y,
                prev_group,
                fg,
                border_c,
                surface,
            )
            y += _GROUP_PAD_BOTTOM

        y += _ARROW_GAP
        self._draw_total_box(painter, y, fg, bg, primary)

        painter.end()

    def _draw_step_box(
        self,
        p: QPainter,
        rect: QRectF,
        step: WorkflowStep,
        index: int,
        fg: QColor,
        bg: QColor,
        primary: QColor,
        surface: QColor,
        muted: QColor,
        border_c: QColor,
    ) -> None:
        is_hovered = index == self._hovered_index

        if step.is_active:
            fill = self._step_fill(step.category, primary, surface, bg)
            border = self._step_border(step.category, primary, border_c)
            text_color = fg
        else:
            fill = surface
            border = muted
            text_color = muted

        if is_hovered and step.is_active:
            fill = fill.lighter(110)

        pen = QPen(border, 1.5)
        if not step.is_active:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(fill)

        path = QPainterPath()
        path.addRoundedRect(rect, _STEP_RADIUS, _STEP_RADIUS)
        p.drawPath(path)

        base_pt = self._safe_pt(p.font())
        name_font = QFont(p.font())
        name_font.setPointSize(base_pt)
        name_font.setBold(True)
        name_font.setItalic(False)
        desc_font = QFont(p.font())
        desc_font.setItalic(False)
        desc_font.setPointSize(base_pt)

        text_x = rect.x() + 12
        badge_space = 0

        if step.estimated_count is not None and step.is_active:
            badge_space = self._draw_count_badge(p, rect, step, primary, bg)

        p.setPen(text_color)
        p.setFont(name_font)
        name_rect = QRectF(text_x, rect.y() + 8, rect.width() - 24 - badge_space, 22)
        display_name = step.name
        if not step.is_active:
            display_name = f'{step.name} (disabled)'
        p.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, display_name)

        if step.description:
            p.setFont(desc_font)
            desc_color = QColor(text_color)
            desc_color.setAlpha(180)
            p.setPen(desc_color)
            desc_rect = QRectF(
                text_x, rect.y() + 30, rect.width() - 24 - badge_space, 20
            )
            elided = QFontMetrics(desc_font).elidedText(
                step.description, Qt.ElideRight, int(desc_rect.width())
            )
            p.drawText(desc_rect, Qt.AlignLeft | Qt.AlignVCenter, elided)

    def _draw_count_badge(
        self,
        p: QPainter,
        rect: QRectF,
        step: WorkflowStep,
        primary: QColor,
        bg: QColor,
    ) -> float:
        prefix = '−' if step.is_filter else '~'
        text = f'{prefix}{step.estimated_count:,}'

        badge_font = QFont(p.font())
        badge_font.setPointSize(max(self._safe_pt(p.font()) - 1, 7))
        badge_font.setBold(False)
        fm = QFontMetrics(badge_font)
        text_w = fm.horizontalAdvance(text)
        badge_w = text_w + 2 * _BADGE_PAD_X

        badge_x = rect.right() - badge_w - 10
        badge_y = rect.y() + (rect.height() - _BADGE_H) / 2
        badge_rect = QRectF(badge_x, badge_y, badge_w, _BADGE_H)

        badge_bg = QColor('#d97706') if step.is_filter else QColor(primary)

        p.setPen(Qt.NoPen)
        p.setBrush(badge_bg)
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, _BADGE_RADIUS, _BADGE_RADIUS)
        p.drawPath(badge_path)

        p.setPen(QColor('#ffffff'))
        p.setFont(badge_font)
        p.drawText(badge_rect, Qt.AlignCenter, text)

        return badge_w + 16

    def _draw_arrow(
        self, p: QPainter, x1: float, y1: float, x2: float, y2: float, color: QColor
    ) -> None:
        p.setPen(QPen(color, 1.5))
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2 - _ARROW_HEAD))

        p.setPen(Qt.NoPen)
        p.setBrush(color)
        head = QPolygonF(
            [
                QPointF(x2, y2),
                QPointF(x2 - _ARROW_HEAD / 2, y2 - _ARROW_HEAD),
                QPointF(x2 + _ARROW_HEAD / 2, y2 - _ARROW_HEAD),
            ]
        )
        p.drawPolygon(head)

    def _draw_group_bracket(
        self,
        p: QPainter,
        y_start: float,
        y_end: float,
        group: str,
        fg: QColor,
        border_c: QColor,
        surface: QColor,
    ) -> None:
        x = float(_MARGIN_X / 2)
        w = float(self.width() - _MARGIN_X)
        h = y_end - y_start + _GROUP_PAD_BOTTOM

        bracket_rect = QRectF(x, y_start, w, h)
        p.setPen(QPen(border_c, 1.0, Qt.DashDotLine))
        bg = QColor(surface)
        bg.setAlpha(60)
        p.setBrush(bg)
        bracket_path = QPainterPath()
        bracket_path.addRoundedRect(bracket_rect, 6, 6)
        p.drawPath(bracket_path)

        label = group
        if group == 'per_phase' and self._num_phases > 1:
            label = f'Per Phase (×{self._num_phases})'
        elif group == 'per_phase':
            label = 'Per Phase'
        elif group == 'post_phase':
            label = 'Post Phase'

        label_font = QFont(p.font())
        label_font.setPointSize(max(self._safe_pt(p.font()) - 1, 7))
        label_font.setItalic(True)
        p.setFont(label_font)

        fm = QFontMetrics(label_font)
        label_w = fm.horizontalAdvance(label) + 12
        label_h = fm.height() + 4

        label_x = x + 12
        label_y = y_start - label_h / 2 + 2

        p.setPen(Qt.NoPen)
        p.setBrush(surface)
        label_bg_rect = QRectF(label_x - 4, label_y, label_w, label_h)
        p.drawRoundedRect(label_bg_rect, 3, 3)

        label_fg = QColor(fg)
        label_fg.setAlpha(200)
        p.setPen(label_fg)
        p.drawText(label_bg_rect, Qt.AlignCenter, label)

    def _draw_total_box(
        self,
        p: QPainter,
        y: float,
        fg: QColor,
        bg: QColor,
        primary: QColor,
    ) -> None:
        x = float(_MARGIN_X)
        w = float(self.width() - 2 * _MARGIN_X)
        rect = QRectF(x, y, w, _STEP_HEIGHT)

        total_step = WorkflowStep(
            name='Estimated Total',
            description=f'~{self._total_estimate:,} structures'
            if self._total_estimate is not None
            else 'unknown',
            estimated_count=self._total_estimate,
            category='generation',
        )
        p.setFont(self.font())
        surface = QColor(theme_colors(self._theme)['surface'])
        muted = QColor(theme_colors(self._theme)['muted'])
        border_c = QColor(theme_colors(self._theme)['border'])
        self._draw_step_box(
            p, rect, total_step, -1, fg, bg, primary, surface, muted, border_c
        )

    # ── colour helpers ────────────────────────────────────────────

    @staticmethod
    def _step_fill(
        category: str, primary: QColor, surface: QColor, bg: QColor
    ) -> QColor:
        if category == 'generation':
            c = QColor(primary)
            c.setAlpha(25)
            return c
        if category == 'modification':
            c = QColor('#0891b2')
            c.setAlpha(25)
            return c
        if category == 'filter':
            c = QColor('#d97706')
            c.setAlpha(25)
            return c
        return surface

    @staticmethod
    def _step_border(category: str, primary: QColor, border_c: QColor) -> QColor:
        if category == 'generation':
            return primary
        if category == 'modification':
            return QColor('#0891b2')
        if category == 'filter':
            return QColor('#d97706')
        return border_c

    # ── interaction ───────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        pos = event.position()
        for rect, step in self._step_rects:
            if rect.contains(pos) and step.is_active and step.config_key:
                self.step_clicked.emit(step.config_key)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        old = self._hovered_index
        self._hovered_index = -1
        for i, (rect, step) in enumerate(self._step_rects):
            if rect.contains(pos) and step.is_active and step.config_key:
                self._hovered_index = i
                break
        if self._hovered_index != old:
            self.update()
        if self._hovered_index >= 0:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)


# ── public wrapper ────────────────────────────────────────────────────


class WorkflowView(QWidget):
    """Scrollable workflow diagram.

    Wraps :class:`_WorkflowCanvas` inside a ``QScrollArea`` and exposes
    a convenience API for setting step data and receiving click signals.
    """

    step_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = _WorkflowCanvas()
        self._canvas.step_clicked.connect(self.step_clicked)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._canvas)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        layout.addWidget(self._scroll)

    def set_steps(
        self,
        steps: Sequence[WorkflowStep],
        num_phases: int = 1,
        total_estimate: int | None = None,
    ) -> None:
        self._canvas.set_data(steps, num_phases, total_estimate)

    def set_theme(self, theme_name: str) -> None:
        self._canvas.set_theme(theme_name)
