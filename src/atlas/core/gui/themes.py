"""Application themes for the ATLAS GUI.

Each theme is derived from a Gogh terminal color scheme and mapped onto
Qt widget roles.  The mapping uses the 16 ANSI colours plus background,
foreground, and cursor from the Gogh palette:

    background  → window / base background
    foreground  → text colour
    color_01    → dark accent (borders, subtle text in dark themes)
    color_02    → red / error / danger
    color_03    → green / success / done
    color_04    → yellow / warning / in-progress
    color_05    → blue / primary / links
    color_06    → magenta / accent
    color_07    → cyan / info
    color_08    → light grey (alt row, muted text in dark themes)
    color_09–16 → bright variants of 01–08

``build_stylesheet(theme_name)`` returns a full QSS string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:  # noqa
    name: str
    variant: str  # 'light' or 'dark'
    background: str
    foreground: str
    cursor: str
    c01: str
    c02: str
    c03: str
    c04: str  # noqa: E702
    c05: str
    c06: str
    c07: str
    c08: str  # noqa: E702
    c09: str
    c10: str
    c11: str
    c12: str  # noqa: E702
    c13: str
    c14: str
    c15: str
    c16: str  # noqa: E702


def _t(
    name,
    variant,
    bg,
    fg,
    cur,
    c01,
    c02,
    c03,
    c04,
    c05,
    c06,
    c07,
    c08,
    c09,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    c16,
) -> Theme:
    return Theme(
        name,
        variant,
        bg,
        fg,
        cur,
        c01,
        c02,
        c03,
        c04,
        c05,
        c06,
        c07,
        c08,
        c09,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
    )


THEMES: dict[str, Theme] = {}


def _register(*themes: Theme):
    for t in themes:
        THEMES[t.name] = t


_register(
    _t(
        'Default (Light)',
        'light',
        '#ffffff',
        '#1a1a2e',
        '#1a1a2e',
        '#2b2b2b',
        '#dc322f',
        '#2e8b57',
        '#b8860b',
        '#2962ff',
        '#8b2fc9',
        '#0891b2',
        '#6c757d',
        '#495057',
        '#e74c3c',
        '#28a745',
        '#e5c07b',
        '#4a90d9',
        '#a855f7',
        '#22d3ee',
        '#adb5bd',
    ),
    _t(
        'Mimi (Light)',
        'light',
        '#FFF0F5',
        '#4A3040',
        '#4A3040',
        '#6B4C5A',
        '#D94E6E',
        '#7BAE7F',
        '#C9956B',
        '#B06098',
        '#D4789C',
        '#6BA5A5',
        '#B8A0AC',
        '#8A6878',
        '#E8648A',
        '#8EC493',
        '#DAAD82',
        '#C87AB0',
        '#E895B5',
        '#82BFBF',
        '#D4C0CC',
    ),
    _t(
        'Gruvbox Dark',
        'dark',
        '#282828',
        '#EBDBB2',
        '#EBDBB2',
        '#282828',
        '#CC241D',
        '#98971A',
        '#D79921',
        '#458588',
        '#B16286',
        '#689D6A',
        '#A89984',
        '#928374',
        '#FB4934',
        '#B8BB26',
        '#FABD2F',
        '#83A598',
        '#D3869B',
        '#8EC07C',
        '#EBDBB2',
    ),
    _t(
        'Catppuccin Mocha',
        'dark',
        '#1E1E2E',
        '#CDD6F4',
        '#CDD6F4',
        '#45475A',
        '#F38BA8',
        '#A6E3A1',
        '#F9E2AF',
        '#89B4FA',
        '#F5C2E7',
        '#94E2D5',
        '#BAC2DE',
        '#585B70',
        '#F38BA8',
        '#A6E3A1',
        '#F9E2AF',
        '#89B4FA',
        '#F5C2E7',
        '#94E2D5',
        '#A6ADC8',
    ),
    _t(
        'Catppuccin Latte',
        'light',
        '#EFF1F5',
        '#4C4F69',
        '#4C4F69',
        '#5C5F77',
        '#D20F39',
        '#40A02B',
        '#DF8E1D',
        '#1E66F5',
        '#EA76CB',
        '#179299',
        '#ACB0BE',
        '#6C6F85',
        '#D20F39',
        '#40A02B',
        '#DF8E1D',
        '#1E66F5',
        '#EA76CB',
        '#179299',
        '#BCC0CC',
    ),
    _t(
        'Dracula',
        'dark',
        '#282A36',
        '#F8F8F2',
        '#F8F8F2',
        '#262626',
        '#E64747',
        '#42E66C',
        '#E4F34A',
        '#9B6BDF',
        '#E356A7',
        '#75D7EC',
        '#F8F8F2',
        '#7A7A7A',
        '#FF5555',
        '#50FA7B',
        '#F1FA8C',
        '#BD93F9',
        '#FF79C6',
        '#8BE9FD',
        '#F9F9FB',
    ),
    _t(
        'Nord',
        'dark',
        '#2E3440',
        '#D8DEE9',
        '#D8DEE9',
        '#3B4252',
        '#BF616A',
        '#A3BE8C',
        '#EBCB8B',
        '#81A1C1',
        '#B48EAD',
        '#88C0D0',
        '#E5E9F0',
        '#4C566A',
        '#BF616A',
        '#A3BE8C',
        '#EBCB8B',
        '#81A1C1',
        '#B48EAD',
        '#8FBCBB',
        '#ECEFF4',
    ),
    _t(
        'Solarized Dark',
        'dark',
        '#002B36',
        '#839496',
        '#839496',
        '#073642',
        '#DC322F',
        '#859900',
        '#CF9A6B',
        '#268BD2',
        '#D33682',
        '#2AA198',
        '#EEE8D5',
        '#657B83',
        '#CB4B16',
        '#859900',
        '#CF9A6B',
        '#6c71c4',
        '#D33682',
        '#2AA198',
        '#FDF6E3',
    ),
    _t(
        'Solarized Light',
        'light',
        '#FDF6E3',
        '#657B83',
        '#657B83',
        '#EEE8D5',
        '#DC322F',
        '#859900',
        '#B58900',
        '#268BD2',
        '#D33682',
        '#2AA198',
        '#002B36',
        '#657b83',
        '#CB4B16',
        '#859900',
        '#B58900',
        '#6C71C4',
        '#D33682',
        '#2AA198',
        '#073642',
    ),
    _t(
        'Tokyo Night',
        'dark',
        '#1A1B26',
        '#C0CAF5',
        '#C0CAF5',
        '#414868',
        '#F7768E',
        '#9ECE6A',
        '#E0AF68',
        '#7AA2F7',
        '#BB9AF7',
        '#7DCFFF',
        '#A9B1D6',
        '#414868',
        '#F7768E',
        '#9ECE6A',
        '#E0AF68',
        '#7AA2F7',
        '#BB9AF7',
        '#7DCFFF',
        '#C0CAF5',
    ),
    _t(
        'Tokyo Night Light',
        'light',
        '#D5D6DB',
        '#565A6E',
        '#565A6E',
        '#0F0F14',
        '#8C4351',
        '#485E30',
        '#8F5E15',
        '#34548A',
        '#5A4A78',
        '#0F4B6E',
        '#343B58',
        '#9699A3',
        '#8C4351',
        '#485E30',
        '#8F5E15',
        '#34548A',
        '#5A4A78',
        '#0F4B6E',
        '#343B58',
    ),
    _t(
        'Everforest Dark',
        'dark',
        '#272E33',
        '#D3C6AA',
        '#D3C6AA',
        '#2E383C',
        '#E67E80',
        '#A7C080',
        '#DBBC7F',
        '#7FBBB3',
        '#D699B6',
        '#83C092',
        '#D3C6AA',
        '#5C6A72',
        '#F85552',
        '#8DA101',
        '#DFA000',
        '#3A94C5',
        '#DF69BA',
        '#35A77C',
        '#DFDDC8',
    ),
    _t(
        'Everforest Light',
        'light',
        '#FFFBEF',
        '#5C6A72',
        '#5C6A72',
        '#5C6A72',
        '#F85552',
        '#8DA101',
        '#DFA000',
        '#3A94C5',
        '#DF69BA',
        '#35A77C',
        '#DFDDC8',
        '#2E383C',
        '#E67E80',
        '#A7C080',
        '#DBBC7F',
        '#7FBBB3',
        '#D699B6',
        '#83C092',
        '#D3C6AA',
    ),
    _t(
        'Kanagawa Wave',
        'dark',
        '#1F1F28',
        '#DCD7BA',
        '#DCD7BA',
        '#090618',
        '#C34043',
        '#76946A',
        '#C0A36E',
        '#7E9CD8',
        '#957FB8',
        '#6A9589',
        '#C8C093',
        '#727169',
        '#E82424',
        '#98BB6C',
        '#E6C384',
        '#7FB4CA',
        '#938AA9',
        '#7AA89F',
        '#DCD7BA',
    ),
    _t(
        'Kanagawa Lotus',
        'light',
        '#f2ecbc',
        '#545464',
        '#43436c',
        '#1f1f28',
        '#c84053',
        '#6f894e',
        '#77713f',
        '#4d699b',
        '#b35b79',
        '#597b75',
        '#545464',
        '#8a8980',
        '#d7474b',
        '#6e915f',
        '#836f4a',
        '#6693bf',
        '#624c83',
        '#5e857a',
        '#43436c',
    ),
    _t(
        'Monokai Dark',
        'dark',
        '#272822',
        '#F8F8F2',
        '#F8F8F2',
        '#75715E',
        '#F92672',
        '#A6E22E',
        '#F4BF75',
        '#66D9EF',
        '#AE81FF',
        '#2AA198',
        '#F9F8F5',
        '#272822',
        '#F92672',
        '#A6E22E',
        '#F4BF75',
        '#66D9EF',
        '#AE81FF',
        '#2AA198',
        '#F8F8F2',
    ),
    _t(
        'One Dark',
        'dark',
        '#1E2127',
        '#ABB2BF',
        '#ABB2BF',
        '#000000',
        '#E06C75',
        '#98C379',
        '#D19A66',
        '#61AFEF',
        '#C678DD',
        '#56B6C2',
        '#ABB2BF',
        '#5C6370',
        '#E06C75',
        '#98C379',
        '#D19A66',
        '#61AFEF',
        '#C678DD',
        '#56B6C2',
        '#FFFEFE',
    ),
)

DEFAULT_THEME = 'Default (Light)'


def _mix(hex_a: str, hex_b: str, t: float = 0.5) -> str:
    """Linearly interpolate two hex colours. *t* = 0 → a, 1 → b."""
    ra, ga, ba = int(hex_a[1:3], 16), int(hex_a[3:5], 16), int(hex_a[5:7], 16)
    rb, gb, bb = int(hex_b[1:3], 16), int(hex_b[3:5], 16), int(hex_b[5:7], 16)
    r = int(ra + (rb - ra) * t)
    g = int(ga + (gb - ga) * t)
    b = int(ba + (bb - ba) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def _alpha(hex_color: str, alpha: int) -> str:
    """Return ``rgba(r,g,b,a)`` from a hex colour + 0–255 alpha."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'


_STATIC_QSS: str | None = None


def build_static_stylesheet() -> str:
    """Return a static QSS that uses palette(...) references for all colors.

    The same QSS works for every theme, only the QPalette changes.
    """
    global _STATIC_QSS  # noqa: PLW0603
    if _STATIC_QSS is not None:
        return _STATIC_QSS

    _STATIC_QSS = """
/* ====== Base properties (targeted, not universal QWidget) ====== */
QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
QComboBox, QCheckBox, QRadioButton, QGroupBox, QToolButton {
    background-color: palette(window);
    color: palette(window-text);
    font-size: 13px;
}

/* ====== Menu Bar ====== */
QMenuBar {
    background-color: palette(tooltip-base);
    color: palette(window-text);
    border-bottom: 1px solid palette(mid);
}
QMenuBar::item:selected {
    background-color: palette(light);
}
QMenu {
    background-color: palette(tooltip-base);
    color: palette(window-text);
    border: 1px solid palette(mid);
}
QMenu::item:selected {
    background-color: palette(highlight);
    color: palette(highlighted-text);
}
QMenu::separator {
    height: 1px;
    background: palette(midlight);
    margin: 4px 8px;
}

/* ====== Sidebar (QListWidget) ====== */
QListWidget {
    background-color: palette(alternate-base);
    border: none;
    border-right: 1px solid palette(mid);
    padding: 4px;
}
QListWidget::item {
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 4px;
    color: palette(window-text);
}
QListWidget::item:selected {
    background-color: palette(highlight);
    color: palette(highlighted-text);
    font-weight: bold;
}
QListWidget::item:hover:!selected {
    background-color: palette(light);
}

/* ====== Tabs ====== */
QTabWidget::pane {
    border: 1px solid palette(mid);
    border-top: none;
    background-color: palette(window);
}
QTabBar::tab {
    background-color: palette(alternate-base);
    color: palette(placeholder-text);
    padding: 8px 16px;
    border: 1px solid palette(mid);
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: palette(window);
    color: palette(window-text);
    border-bottom: 2px solid palette(link);
}
QTabBar::tab:hover:!selected {
    background-color: palette(light);
    color: palette(window-text);
}

/* ── Page-level (outer) tab bar ── */
QTabWidget#pageTab > QTabBar::tab {
    padding: 9px 18px;
    font-weight: 600;
}
QTabWidget#pageTab > QTabBar::tab:disabled {
    min-width: 0px;
    max-width: 2px;
    padding: 4px 0px;
    margin-left: 6px;
    margin-right: 6px;
    background-color: palette(mid);
    border: none;
    border-radius: 1px;
}
QTabWidget#pageTab::pane {
    border-top: 2px solid palette(mid);
}

/* ====== Buttons ====== */
QPushButton {
    background-color: palette(button);
    color: palette(window-text);
    border: 1px solid palette(mid);
    border-radius: 4px;
    padding: 5px 14px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: palette(bright-text);
    border-color: palette(mid);
}
QPushButton:pressed {
    background-color: palette(dark);
}
QPushButton:disabled {
    color: palette(placeholder-text);
    background-color: palette(alternate-base);
    border-color: palette(midlight);
}

/* ====== Input Fields ====== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: palette(base);
    color: palette(window-text);
    border: 1px solid palette(mid);
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: palette(link);
}
QComboBox::drop-down {
    border: none;
    width: 20px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid palette(placeholder-text);
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: palette(tooltip-base);
    color: palette(window-text);
    border: 1px solid palette(mid);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}

/* ====== Spin Box Buttons ====== */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid palette(midlight);
    border-top-right-radius: 4px;
    background: palette(button);
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background: palette(bright-text);
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid palette(midlight);
    border-bottom-right-radius: 4px;
    background: palette(button);
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: palette(bright-text);
}

/* ====== Tables ====== */
QTableWidget, QTableView {
    background-color: palette(window);
    color: palette(window-text);
    gridline-color: palette(midlight);
    border: 1px solid palette(mid);
    alternate-background-color: palette(alternate-base);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}
QHeaderView::section {
    background-color: palette(tooltip-base);
    color: palette(window-text);
    padding: 4px 8px;
    border: none;
    border-bottom: 1px solid palette(mid);
    border-right: 1px solid palette(midlight);
    font-weight: bold;
}

/* ====== Group Boxes ====== */
QGroupBox {
    background-color: palette(window);
    color: palette(window-text);
    border: 1px solid palette(mid);
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: palette(window-text);
}

/* ====== Frames ====== */
QFrame {
    background-color: palette(window);
    color: palette(window-text);
}

/* ====== Splitter ====== */
QSplitter {
    background-color: palette(window);
}
QSplitter::handle {
    background-color: palette(midlight);
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}

/* ====== Scroll Area ====== */
QScrollArea {
    border: none;
    background-color: palette(window);
}

/* ====== Containers ====== */
QStackedWidget, QTabWidget, QMainWindow,
QWidget#atlasPage, QWidget#atlasPanel {
    background-color: palette(window);
    color: palette(window-text);
}

/* ====== Log Viewer (QTextEdit) ====== */
QTextEdit {
    background-color: palette(window);
    color: palette(window-text);
    border: 1px solid palette(mid);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}

/* ====== Dock Widget ====== */
QDockWidget {
    color: palette(window-text);
    titlebar-close-icon: none;
}
QDockWidget::title {
    background-color: palette(tooltip-base);
    padding: 6px;
    border-bottom: 1px solid palette(mid);
}

/* ====== Log Toggle Bar ====== */
QWidget#logToggleBar {
    background-color: palette(tooltip-base);
    border-top: 1px solid palette(mid);
    padding: 4px 8px;
}
QWidget#logToggleBar:hover {
    background-color: palette(light);
}
QWidget#logToggleBar:pressed {
    background-color: palette(dark);
}

/* ====== Scrollbars ====== */
QScrollBar:vertical {
    background: palette(alternate-base);
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background: palette(link-visited);
    min-height: 30px;
    border-radius: 4px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: palette(shadow);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: palette(alternate-base);
    height: 10px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: palette(link-visited);
    min-width: 30px;
    border-radius: 4px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background: palette(shadow);
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ====== Progress Bar ====== */
QProgressBar {
    border: 1px solid palette(mid);
    border-radius: 4px;
    background: palette(alternate-base);
    text-align: center;
    color: palette(window-text);
}
QProgressBar::chunk {
    background: palette(link);
    border-radius: 3px;
}

/* ====== Tooltips ====== */
QToolTip {
    font-size: 13px;
    padding: 4px;
}

/* ====== Status Bar ====== */
QStatusBar {
    background-color: palette(tooltip-base);
    color: palette(window-text);
    border-top: 1px solid palette(mid);
}

/* ====== Labels ====== */
QLabel {
    background-color: transparent;
    color: palette(window-text);
}

/* ====== Disabled Group Boxes (optional sections) ====== */
QGroupBox[section_off="true"] {
    background-color: palette(alternate-base);
    border-style: dashed;
}
QGroupBox[section_off="true"] QLabel,
QGroupBox[section_off="true"] QLineEdit,
QGroupBox[section_off="true"] QSpinBox,
QGroupBox[section_off="true"] QDoubleSpinBox,
QGroupBox[section_off="true"] QComboBox,
QGroupBox[section_off="true"] QCheckBox,
QGroupBox[section_off="true"] QTableWidget,
QGroupBox[section_off="true"] QPushButton {
    color: palette(placeholder-text);
}
QGroupBox[section_off="true"] QLineEdit,
QGroupBox[section_off="true"] QSpinBox,
QGroupBox[section_off="true"] QDoubleSpinBox,
QGroupBox[section_off="true"] QComboBox,
QGroupBox[section_off="true"] QCheckBox,
QGroupBox[section_off="true"] QTableWidget {
    background-color: palette(alternate-base);
    border-color: palette(midlight);
}
"""
    return _STATIC_QSS


def build_theme_palette(theme_name: str):
    """Build a QPalette with all theme-derived colors mapped to roles."""
    from PySide6.QtGui import QColor, QPalette

    t = THEMES.get(theme_name)
    if t is None:
        t = THEMES[DEFAULT_THEME]

    is_dark = t.variant == 'dark'
    bg = t.background
    fg = t.foreground

    alt_bg = _mix(bg, fg, 0.04) if is_dark else _mix(bg, fg, 0.05)
    surface = _mix(bg, fg, 0.07) if is_dark else _mix(bg, fg, 0.06)
    border = _mix(bg, fg, 0.15) if is_dark else _mix(bg, fg, 0.16)
    border_light = _mix(bg, fg, 0.10)
    muted = _mix(bg, fg, 0.45)
    hover = _mix(bg, t.c05, 0.15) if is_dark else _mix(bg, t.c05, 0.10)
    selected_bg = _mix(bg, t.c05, 0.25) if is_dark else _mix(bg, t.c05, 0.15)
    selected_fg = t.c13 if is_dark else t.c05
    primary = t.c05 if not is_dark else t.c13
    btn_bg = _mix(bg, fg, 0.08) if not is_dark else surface
    btn_hover = _mix(btn_bg, fg, 0.08)
    btn_pressed = _mix(btn_bg, fg, 0.14)
    input_bg = bg if is_dark else '#ffffff'
    scrollbar_handle = _mix(bg, fg, 0.20)
    scrollbar_handle_hover = _mix(bg, fg, 0.30)

    p = QPalette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        p.setColor(group, QPalette.ColorRole.Window, QColor(bg))
        p.setColor(group, QPalette.ColorRole.WindowText, QColor(fg))
        p.setColor(group, QPalette.ColorRole.Base, QColor(input_bg))
        p.setColor(group, QPalette.ColorRole.AlternateBase, QColor(alt_bg))
        p.setColor(group, QPalette.ColorRole.Text, QColor(fg))
        p.setColor(group, QPalette.ColorRole.Button, QColor(btn_bg))
        p.setColor(group, QPalette.ColorRole.ButtonText, QColor(fg))
        p.setColor(group, QPalette.ColorRole.Highlight, QColor(selected_bg))
        p.setColor(group, QPalette.ColorRole.HighlightedText, QColor(selected_fg))
        p.setColor(group, QPalette.ColorRole.ToolTipBase, QColor(surface))
        p.setColor(group, QPalette.ColorRole.ToolTipText, QColor(fg))
        p.setColor(group, QPalette.ColorRole.Mid, QColor(border))
        p.setColor(group, QPalette.ColorRole.Midlight, QColor(border_light))
        p.setColor(group, QPalette.ColorRole.Light, QColor(hover))
        p.setColor(group, QPalette.ColorRole.Dark, QColor(btn_pressed))
        p.setColor(group, QPalette.ColorRole.Link, QColor(primary))
        p.setColor(group, QPalette.ColorRole.LinkVisited, QColor(scrollbar_handle))
        p.setColor(group, QPalette.ColorRole.PlaceholderText, QColor(muted))
        p.setColor(group, QPalette.ColorRole.BrightText, QColor(btn_hover))
        p.setColor(group, QPalette.ColorRole.Shadow, QColor(scrollbar_handle_hover))

    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window, QColor(bg))
    p.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(muted)
    )
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(muted))
    p.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(muted)
    )
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(alt_bg))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(alt_bg))
    return p


def build_stylesheet(theme_name: str) -> str:
    """Generate a full QSS stylesheet for the given theme.

    .. deprecated:: Use ``build_static_stylesheet`` + ``build_theme_palette``.
    """
    return build_static_stylesheet()


def theme_names() -> list[str]:
    """Return theme names, light themes first, then dark."""
    light = sorted(n for n, t in THEMES.items() if t.variant == 'light')
    dark = sorted(n for n, t in THEMES.items() if t.variant == 'dark')
    return light + dark


def theme_variant(name: str) -> str:
    """Return 'light' or 'dark' for the given theme name."""
    t = THEMES.get(name)
    return t.variant if t else 'light'


def readable_on(hex_color: str) -> str:
    """Return a foreground colour that stays legible on *hex_color*.

    Used for icons drawn on top of an accent background (a checked side-panel
    button, for instance): the theme's normal foreground is unreadable on a
    dark accent, so pick white or near-black from the accent's luminance.
    """
    try:
        channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    except (ValueError, IndexError):
        return '#ffffff'

    def _linear(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (_linear(c) for c in channels)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#1a1a2e' if luminance > 0.3 else '#ffffff'


def theme_colors(name: str) -> dict[str, str]:
    """Return derived colors for the theme (bg, fg, surface, border, etc.)."""
    t = THEMES.get(name)
    if t is None:
        return {
            'bg': '#ffffff',
            'fg': '#1a1a2e',
            'surface': '#f5f5f5',
            'border': '#d0d4db',
            'axes_bg': '#ffffff',
            'muted': '#888888',
            'primary': '#4a90d9',
            'on_primary': readable_on('#4a90d9'),
        }
    is_dark = t.variant == 'dark'
    bg = t.background
    fg = t.foreground
    surface = _mix(bg, fg, 0.07) if is_dark else _mix(bg, fg, 0.04)
    border = _mix(bg, fg, 0.15) if is_dark else _mix(bg, fg, 0.12)
    axes_bg = _mix(bg, fg, 0.03) if is_dark else _mix(bg, fg, 0.02)
    muted = _mix(bg, fg, 0.45)
    primary = t.c05 if not is_dark else t.c13
    return {
        'bg': bg,
        'fg': fg,
        'surface': surface,
        'border': border,
        'axes_bg': axes_bg,
        'muted': muted,
        'primary': primary,
        'on_primary': readable_on(primary),
    }


def _repolish_visible(app, verbose: bool = False) -> None:
    """Re-polish only currently visible widgets.

    After a QPalette change the ``palette(...)`` references in the static
    QSS are stale.  Calling ``app.setStyleSheet()`` again would re-polish
    *every* widget (slow).  Instead we manually unpolish+polish only
    visible ones, hidden QStackedWidget pages are skipped and get
    re-polished lazily when shown.
    """
    import time

    from PySide6.QtWidgets import QWidget

    t0 = time.perf_counter()
    style = app.style()
    count = 0
    for widget in app.allWidgets():
        if widget.isVisible() and isinstance(widget, QWidget):
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
            count += 1
    if verbose:
        dt = (time.perf_counter() - t0) * 1000
        print(f'[ATLAS] theme:   repolish ({count} visible widgets) {dt:.1f}ms')


def apply_theme_to_app(
    theme_name: str,
    *,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Apply theme to QApplication via QPalette + static QSS.

    The QSS uses ``palette(...)`` references and is set only once (at
    startup, when few widgets exist).  Subsequent theme switches update
    the QPalette and selectively re-polish only visible widgets, hidden
    pages in the QStackedWidget are skipped and re-polished lazily.
    """
    import time

    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication

    def _t(label: str, t0: float) -> float:
        now = time.perf_counter()
        if verbose:
            print(f'[ATLAS] theme:   {label} {(now - t0) * 1000:.1f}ms')
        return now

    app = QApplication.instance()
    if app is None:
        return

    settings = QSettings('ATLAS', 'GUI')
    if not force and settings.value('app_theme') == theme_name and app.styleSheet():
        return

    t0 = time.perf_counter()
    first_time = not app.styleSheet()

    palette = build_theme_palette(theme_name)
    app.setPalette(palette)
    t0 = _t('app.setPalette', t0)

    if first_time:
        qss = build_static_stylesheet()
        app.setStyleSheet(qss)
        t0 = _t('app.setStyleSheet (initial)', t0)
    else:
        _repolish_visible(app, verbose=verbose)
        t0 = time.perf_counter()

    settings.setValue('app_theme', theme_name)

    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    t0 = _t('setFont', t0)

    variant = theme_variant(theme_name)
    try:
        scheme = Qt.ColorScheme.Dark if variant == 'dark' else Qt.ColorScheme.Light
        app.styleHints().setColorScheme(scheme)
    except AttributeError:
        pass
    _t('setColorScheme', t0)


def saved_global_theme() -> str:
    """Return the globally persisted theme name (via QSettings)."""
    from PySide6.QtCore import QSettings

    return QSettings('ATLAS', 'GUI').value('app_theme', DEFAULT_THEME)


def saved_global_toml_theme() -> str:
    from PySide6.QtCore import QSettings

    return QSettings('ATLAS', 'GUI').value('toml_syntax_theme', 'Default')


def save_global_toml_theme(name: str) -> None:
    from PySide6.QtCore import QSettings

    QSettings('ATLAS', 'GUI').setValue('toml_syntax_theme', name)
