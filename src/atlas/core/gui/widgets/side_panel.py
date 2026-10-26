"""Blender-style switchable panel with a vertical icon strip.

``SidePanel`` presents a narrow column of icon buttons on the left edge
and a ``QStackedWidget`` on the right.  Clicking a button switches the
visible content pane.  Each pane is registered via :meth:`add_view`.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from atlas.core.gui.icons import themed_icon
from atlas.core.gui.themes import saved_global_theme, theme_colors


class SidePanel(QWidget):
    """Switchable panel with vertical icon-button strip."""

    view_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._icon_names: list[str] = []
        self._buttons: list[QToolButton] = []
        self._theme_name: str = saved_global_theme()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._strip_widget = QFrame()
        self._strip_widget.setObjectName('sidePanelStrip')
        self._strip_layout = QVBoxLayout(self._strip_widget)
        self._strip_layout.setContentsMargins(3, 6, 3, 6)
        self._strip_layout.setSpacing(2)
        self._strip_layout.addStretch()
        self._strip_widget.setFixedWidth(34)
        layout.addWidget(self._strip_widget)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._collapsed = False
        self._active_index: int = 0

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(False)
        self._button_group.idClicked.connect(self._on_button_clicked)

    def add_view(
        self,
        icon_name: str,
        tooltip: str,
        widget: QWidget,
        *,
        visible: bool = True,
    ) -> int:
        """Register a content pane and return its index."""
        idx = self._stack.addWidget(widget)

        btn = QToolButton()
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        btn.setFixedSize(28, 28)
        btn.setIconSize(QSize(20, 20))
        btn.setObjectName('sidePanelButton')

        colors = theme_colors(self._theme_name)
        btn.setIcon(self._stateful_icon(icon_name, colors))

        self._strip_layout.insertWidget(self._strip_layout.count() - 1, btn)
        self._button_group.addButton(btn, idx)
        self._icon_names.append(icon_name)
        self._buttons.append(btn)

        if not visible:
            btn.hide()

        if idx == 0 and visible:
            btn.setChecked(True)

        return idx

    def set_current_view(self, index: int) -> None:
        self._active_index = index
        self._collapsed = False
        self._stack.setCurrentIndex(index)
        self._stack.show()
        self.setMaximumWidth(16777215)
        for i, b in enumerate(self._buttons):
            b.setChecked(i == index)

    def current_view(self) -> int:
        return self._stack.currentIndex()

    def show_view(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].show()

    def hide_view(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].hide()

    def set_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        colors = theme_colors(theme_name)
        for icon_name, btn in zip(self._icon_names, self._buttons, strict=True):
            btn.setIcon(self._stateful_icon(icon_name, colors))
        self._update_stylesheet(colors)

    @staticmethod
    def _stateful_icon(icon_name: str, colors: dict[str, str]) -> QIcon:
        """Foreground icon normally, accent-readable icon while checked.

        The checked background is the theme's primary colour, which is dark
        enough in several themes that the ordinary foreground icon nearly
        disappears on it.
        """
        size = 20
        icon = QIcon()
        icon.addPixmap(
            themed_icon(icon_name, colors['fg'], size=size).pixmap(size, size),
            QIcon.Mode.Normal,
            QIcon.State.Off,
        )
        icon.addPixmap(
            themed_icon(
                icon_name, colors.get('on_primary', '#ffffff'), size=size
            ).pixmap(size, size),
            QIcon.Mode.Normal,
            QIcon.State.On,
        )
        return icon

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_stylesheet(theme_colors(self._theme_name))

    def _update_stylesheet(self, colors: dict[str, str]) -> None:
        primary = colors['primary']
        surface = colors['surface']
        self._strip_widget.setStyleSheet(
            f'QFrame#sidePanelStrip {{ background: {surface}; }}'
        )
        for btn in self._buttons:
            btn.setStyleSheet(
                f'QToolButton#sidePanelButton {{'
                f'  border: none; border-radius: 4px; padding: 2px;'
                f'}}'
                f'QToolButton#sidePanelButton:checked {{'
                f'  background: {primary};'
                f'}}'
                f'QToolButton#sidePanelButton:hover:!checked {{'
                f'  background: {surface};'
                f'}}'
            )

    def _on_button_clicked(self, index: int) -> None:
        btn = self._button_group.button(index)
        if index == self._active_index and not self._collapsed:
            self._collapsed = True
            self._stack.hide()
            self.setMaximumWidth(34)
            if btn:
                btn.setChecked(False)
            return

        self._collapsed = False
        self._active_index = index
        self._stack.setCurrentIndex(index)
        self._stack.show()
        self.setMaximumWidth(16777215)

        for b in self._buttons:
            b.setChecked(b is btn)

        self.view_changed.emit(index)
