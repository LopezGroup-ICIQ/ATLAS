"""Tests for theme colour derivation.

The interesting part is ``readable_on``: the side-panel strip draws icons on a
checked background of the theme's primary colour, and several themes have a
primary dark enough that the normal foreground icon all but disappears.
"""

from __future__ import annotations


class TestReadableOn:
    """Foreground colours that stay legible on an accent background."""

    def test_dark_blue_gets_a_white_foreground(self):
        from atlas.core.gui.themes import readable_on

        assert readable_on('#2962ff') == '#ffffff'

    def test_light_blue_gets_a_dark_foreground(self):
        from atlas.core.gui.themes import readable_on

        assert readable_on('#89B4FA') == '#1a1a2e'

    def test_malformed_input_falls_back_to_white(self):
        from atlas.core.gui.themes import readable_on

        assert readable_on('not-a-colour') == '#ffffff'
        assert readable_on('') == '#ffffff'


class TestThemeColors:
    """Every theme must expose the keys widget code relies on."""

    def test_on_primary_is_defined_for_every_theme(self):
        from atlas.core.gui.themes import THEMES, readable_on, theme_colors

        for name in THEMES:
            colors = theme_colors(name)
            assert colors['on_primary'] == readable_on(colors['primary'])
            assert colors['on_primary'] in ('#ffffff', '#1a1a2e')

    def test_unknown_theme_still_returns_on_primary(self):
        from atlas.core.gui.themes import theme_colors

        assert 'on_primary' in theme_colors('does-not-exist')
