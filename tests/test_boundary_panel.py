"""Headless tests for the latent-space boundary panel and its traversal controls."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QObject, Signal  # noqa: E402


@pytest.fixture(scope='module')
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _matplotlib_map(monkeypatch):
    """Pin the matplotlib canvas, whose artists these tests drive directly.

    The default (pyqtgraph/OpenGL) map is covered in ``test_latent_map_view.py``.
    """
    monkeypatch.setenv('ATL_GUI_LATENT_PLOT', 'matplotlib')


def _write_npz(directory, **overrides):
    """Write a synthetic latent_space.npz, with provenance unless overridden."""
    rng = np.random.default_rng(0)
    n = 60
    latent = rng.normal(size=(n, 2))
    payload = {
        'latent_space': latent,
        'unique_ids': np.array([f'id-{i:03d}' for i in range(n)], dtype=str),
        'boundary_method': np.array(['concave_hull']),
        'hull_vertices': np.array(
            [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0]]
        ),
        'alpha': np.array([5.0]),
        'descriptor_type': np.array(['mace']),
        'descriptor_model': np.array(['mace:mp-small']),
        'autoencoder_path': np.array([str(directory / 'autoencoder_model.pth')]),
        'latent_mode': np.array(['structure_mean']),
        'dtype': np.array(['float64']),
        'atom_latent_space': latent,
        'atom_offsets': np.arange(n + 1, dtype=np.int64),
    }
    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'latent_space.npz'
    np.savez(path, **payload)
    return path


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


class FakeEvent:
    """Minimal stand-in for a matplotlib mouse event."""

    def __init__(self, ax, xdata, ydata, button=1, x=100, y=100):
        self.inaxes = ax
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.x = x
        self.y = y


@pytest.fixture
def project(tmp_path):
    from atlas.core.gui.project import Project

    return Project.create(tmp_path, 'demo')


@pytest.fixture
def panel(qapp, project):
    from atlas.core.gui.widgets.boundary_panel import BoundaryPanel

    _write_npz(project.dir / 'databases')
    panel = BoundaryPanel(project)
    yield panel
    panel._stop_worker()


def _click(panel, x, y, button=1):
    """Drive a full press/release cycle at data coordinates (x, y)."""
    ax = panel._hover_ax
    panel._on_mouse_press(FakeEvent(ax, x, y, button=button))
    panel._on_mouse_release(FakeEvent(ax, x, y, button=button))


class TestTraversalAvailability:
    """The single ordered gate that decides whether inversion can run."""

    def test_no_map_is_not_available(self):
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        gate = traversal_availability(None, None, db_available=True)
        assert not gate.available
        assert 'latent-space data' in gate.reason
        # Regenerating the map is exactly the fix for a missing one.
        assert gate.recompute_fixable

    def test_missing_torch_is_reported_first(self, tmp_path, monkeypatch):
        import importlib.util

        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        path = _write_npz(tmp_path)
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            'find_spec',
            lambda name, *a, **k: None
            if name == 'torch'
            else real_find_spec(name, *a, **k),
        )

        gate = traversal_availability(path, _load(path), db_available=True)
        assert not gate.available
        assert 'PyTorch' in gate.reason
        # No amount of regeneration installs torch.
        assert not gate.recompute_fixable

    def test_legacy_map_without_provenance(self, tmp_path):
        pytest.importorskip('torch')
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        path = _write_npz(tmp_path, descriptor_type=None)
        gate = traversal_availability(path, _load(path), db_available=True)
        assert not gate.available
        assert 'predates descriptor provenance tracking' in gate.reason
        assert 're-run the boundary generation' in gate.reason.lower()
        assert gate.recompute_fixable

    def test_soap_map_says_why_soap_cannot_work(self, tmp_path):
        pytest.importorskip('torch')
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        path = _write_npz(tmp_path, descriptor_type=np.array(['soap']))
        gate = traversal_availability(path, _load(path), db_available=True)
        assert not gate.available
        assert 'SOAP' in gate.reason
        assert 'MACE' in gate.reason
        # Switching the descriptor and regenerating is the fix.
        assert gate.recompute_fixable

    def test_missing_autoencoder_is_reported(self, tmp_path):
        pytest.importorskip('torch')
        pytest.importorskip('mace')
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        path = _write_npz(tmp_path)
        gate = traversal_availability(path, _load(path), db_available=True)
        assert not gate.available
        assert 'autoencoder_model.pth' in gate.reason
        assert gate.recompute_fixable

    def test_missing_standardization_is_reported(self, tmp_path):
        pytest.importorskip('torch')
        pytest.importorskip('mace')
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        path = _write_npz(tmp_path)
        (tmp_path / 'autoencoder_model.pth').write_bytes(b'')
        gate = traversal_availability(path, _load(path), db_available=True)
        assert not gate.available
        assert 'ae_mean_vals.npy' in gate.reason
        assert gate.recompute_fixable

    def test_missing_database_is_reported_last(self, tmp_path):
        pytest.importorskip('torch')
        pytest.importorskip('mace')
        from atlas.core.gui.widgets.latent_inversion import traversal_availability

        path = _write_npz(tmp_path)
        (tmp_path / 'autoencoder_model.pth').write_bytes(b'')
        for name in ('ae_mean_vals.npy', 'ae_std_vals.npy'):
            np.save(tmp_path / name, np.zeros(4))

        gate = traversal_availability(path, _load(path), db_available=False)
        assert not gate.available
        assert 'initial database' in gate.reason
        # Regeneration also needs the database to load structures from.
        assert not gate.recompute_fixable

        gate = traversal_availability(path, _load(path), db_available=True)
        assert gate.available
        assert gate.reason == ''


class TestResolveAutoencoderPath:
    """Finding the model that built a map, including after a project move."""

    def test_recorded_path_wins(self, tmp_path):
        from atlas.core.gui.widgets.latent_inversion import resolve_autoencoder_path

        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        model = elsewhere / 'autoencoder_model.pth'
        model.write_bytes(b'')

        path = _write_npz(tmp_path, autoencoder_path=np.array([str(model)]))
        assert resolve_autoencoder_path(path, _load(path)) == model

    def test_falls_back_beside_the_map(self, tmp_path):
        from atlas.core.gui.widgets.latent_inversion import resolve_autoencoder_path

        path = _write_npz(
            tmp_path,
            autoencoder_path=np.array(['/gone/autoencoder_model.pth']),
        )
        beside = tmp_path / 'autoencoder_model.pth'
        beside.write_bytes(b'')
        assert resolve_autoencoder_path(path, _load(path)) == beside

    def test_default_when_nothing_is_recorded(self, tmp_path):
        from atlas.core.gui.widgets.latent_inversion import resolve_autoencoder_path

        path = _write_npz(tmp_path, autoencoder_path=None)
        resolved = resolve_autoencoder_path(path, _load(path))
        assert resolved == tmp_path / 'autoencoder_model.pth'


class TestNpzLoading:
    """Old and new latent_space.npz files both have to render."""

    def test_new_style_file_populates_provenance(self, panel):
        assert panel._npz_data is not None
        assert str(panel._npz_data['descriptor_type'][0]) == 'mace'
        assert panel._hover_ax is not None

    def test_legacy_file_without_provenance_still_draws(self, qapp, project):
        from atlas.core.gui.widgets.boundary_panel import BoundaryPanel

        _write_npz(
            project.dir / 'databases',
            descriptor_type=None,
            descriptor_model=None,
            autoencoder_path=None,
            latent_mode=None,
            dtype=None,
            atom_latent_space=None,
            atom_offsets=None,
        )
        panel = BoundaryPanel(project)
        assert panel._hover_ax is not None
        assert not panel._availability[0]

    def test_morphological_closing_file_draws(self, qapp, project):
        from atlas.core.gui.widgets.boundary_panel import BoundaryPanel

        contour = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
        _write_npz(
            project.dir / 'databases',
            boundary_method=np.array(['morphological_closing']),
            boundary_coords=contour,
            boundary_counts=np.array([len(contour)]),
            morph_disk_size=np.array([10]),
            hull_vertices=None,
            alpha=None,
        )
        panel = BoundaryPanel(project)
        # Overlays are created on both drawing paths, not just the hull one.
        assert panel._wp_line is not None


class TestWaypointMarking:
    """Clicking the map to build a path."""

    def test_mark_mode_collects_clicks(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.5, 0.25)
        _click(panel, -1.0, 1.0)

        assert panel._waypoints == [(0.5, 0.25), (-1.0, 1.0)]

    def test_clicks_select_structures_when_not_marking(self, panel):
        panel._mark_btn.setChecked(False)
        _click(panel, *panel._npz_data['latent_space'][0])

        assert panel._waypoints == []
        assert panel._sel_idx == 0

    def test_navigation_mode_suppresses_both(self, panel, monkeypatch):
        panel._mark_btn.setChecked(True)
        # ``mode`` is an instance attribute on the navigation toolbar.
        monkeypatch.setattr(panel._toolbar, 'mode', 'zoom rect', raising=False)

        _click(panel, 0.5, 0.25)

        assert panel._waypoints == []

    def test_right_click_removes_the_nearest_waypoint(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        _click(panel, 2.0, 2.0)

        panel._on_mouse_release(FakeEvent(panel._hover_ax, 1.98, 2.02, button=3))

        assert panel._waypoints == [(0.0, 0.0)]

    def test_right_click_far_away_removes_nothing(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)

        panel._on_mouse_release(FakeEvent(panel._hover_ax, 50.0, 50.0, button=3))

        assert len(panel._waypoints) == 1

    def test_clear_points_resets_both_overlays(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        panel._inverted = [{'latent': (0.1, 0.1), 'physical': True}]

        panel._on_clear_points()

        assert panel._waypoints == []
        assert panel._inverted == []


class TestOverlays:
    """Artists are rebuilt from state, so redraws never lose the path."""

    def test_refresh_keeps_waypoints_and_drops_results(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.5, 0.5)
        panel._inverted = [{'latent': (0.4, 0.4), 'physical': True}]
        old_artist = panel._wp_line

        panel.refresh()

        assert panel._waypoints == [(0.5, 0.5)]
        assert panel._inverted == []
        # A redraw clears the figure, so the artist must be a new one that was
        # repopulated from the surviving state.
        assert panel._wp_line is not old_artist
        assert list(panel._wp_line.get_xdata()) == [0.5]

    def test_decoded_points_are_split_by_feasibility(self, panel):
        panel._inverted = [
            {'latent': (0.0, 0.0), 'physical': True},
            {'latent': (1.0, 1.0), 'physical': False},
            {'latent': (2.0, 2.0), 'physical': True},
        ]
        panel._sync_overlays()

        assert list(panel._inv_ok_line.get_xdata()) == [0.0, 2.0]
        assert list(panel._inv_bad_line.get_xdata()) == [1.0]

    def test_point_ready_appends_live(self, panel):
        panel._on_point_ready({'latent': (0.3, 0.3), 'physical': True})
        panel._on_point_ready({'latent': (0.6, 0.6), 'physical': True})

        assert len(panel._inverted) == 2
        assert list(panel._inv_ok_line.get_xdata()) == [0.3, 0.6]

    def test_path_line_needs_two_points(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        assert len(panel._path_line.get_xdata()) == 0

        _click(panel, 1.0, 1.0)
        assert len(panel._path_line.get_xdata()) == 2

    def test_waypoint_labels_track_the_waypoints(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        _click(panel, 1.0, 1.0)
        assert [label.get_text() for label in panel._wp_labels] == ['1', '2']

        panel._on_mouse_release(FakeEvent(panel._hover_ax, 0.0, 0.0, button=3))
        assert [label.get_text() for label in panel._wp_labels] == ['1']


class TestTraversalControls:
    """Gating: the run button only lights up when a run could succeed."""

    def test_run_is_disabled_without_waypoints(self, panel):
        assert not panel._run_btn.isEnabled()
        assert panel._run_container.toolTip() != ''

    def test_unavailable_reason_is_shown_on_the_container(self, panel):
        # The container carries the tooltip because a disabled QPushButton
        # receives no hover events and would never show its own.
        available, reason, _fixable = panel._availability
        assert not available
        assert panel._run_container.isEnabled()
        assert reason in panel._run_container.toolTip()

    def test_clear_button_follows_the_waypoints(self, panel):
        assert not panel._clear_btn.isEnabled()
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        assert panel._clear_btn.isEnabled()


class TestSettingsPersistence:
    """Traversal settings live on the project, not on the panel."""

    def test_absent_until_configured(self, project):
        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets.latent_inversion import (
            has_saved_settings,
            load_settings,
            save_settings,
        )

        assert not has_saved_settings(project)
        assert load_settings(project) == LatentInversionSettings()

        settings = LatentInversionSettings(n_structures=7, w_disp=0.0)
        save_settings(project, settings)

        assert has_saved_settings(project)
        assert load_settings(project) == settings

    def test_unreadable_blob_falls_back_to_defaults(self, project):
        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets.latent_inversion import (
            SETTINGS_META_KEY,
            load_settings,
        )

        project.set_meta(SETTINGS_META_KEY, 'not json')
        assert load_settings(project) == LatentInversionSettings()

    def test_unknown_keys_are_dropped(self, project):
        import json

        from atlas.core.gui.widgets.latent_inversion import (
            SETTINGS_META_KEY,
            load_settings,
        )

        project.set_meta(
            SETTINGS_META_KEY,
            json.dumps({'n_structures': 5, 'from_a_future_version': True}),
        )
        assert load_settings(project).n_structures == 5


class TestSettingsDialog:
    """The settings dialog and the settings dataclass cannot drift apart."""

    def test_every_field_has_a_widget_and_every_widget_a_field(self, qapp):
        from dataclasses import fields

        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets.latent_inversion import (
            _FIELD_GROUPS,
            LatentTraversalSettingsDialog,
        )

        named = {name for _, specs in _FIELD_GROUPS for name, *_ in specs}
        known = {field.name for field in fields(LatentInversionSettings)}
        assert named == known

        dialog = LatentTraversalSettingsDialog()
        assert set(dialog._widgets) == named

    def test_values_round_trip_through_the_widgets(self, qapp):
        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalSettingsDialog,
        )

        settings = LatentInversionSettings(
            n_structures=7,
            seed_strategy='by_size',
            target_n_atoms=32,
            tolerance=1e-5,
            device='cuda',
            dtype='float32',
            rep_ramp=(0.5, 2.0),
            w_disp=0.0,
            good_enough_latent_err=5e-3,
        )
        dialog = LatentTraversalSettingsDialog(settings)
        assert dialog.settings() == settings

    def test_an_unparseable_ramp_is_rejected(self, qapp):
        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalSettingsDialog,
        )

        dialog = LatentTraversalSettingsDialog()
        dialog._widgets['rep_ramp'].setText('nonsense')
        with pytest.raises(ValueError, match='Penalty ramp'):
            dialog.settings()

    def test_cell_relaxation_is_disabled_when_not_offered(self, qapp):
        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalSettingsDialog,
        )

        dialog = LatentTraversalSettingsDialog(
            allow_cell_relaxation=False,
            cell_disabled_reason='not yet',
        )
        assert not dialog._widgets['relax_cell'].isEnabled()
        assert dialog._widgets['relax_cell'].toolTip() == 'not yet'
        assert not dialog._widgets['w_strain'].isEnabled()

    def test_structure_count_cannot_go_below_the_marked_points(self, qapp):
        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalSettingsDialog,
        )

        dialog = LatentTraversalSettingsDialog(
            LatentInversionSettings(n_structures=1),
            min_structures=4,
        )
        spin = dialog._widgets['n_structures']

        assert spin.minimum() == 4
        # Loading a smaller saved value is clamped up to the floor.
        assert spin.value() == 4
        assert dialog.settings().n_structures == 4

        spin.setValue(2)
        assert spin.value() == 4

    def test_size_fields_follow_the_seeding_strategy(self, qapp):
        from atlas.core.gui.widgets.latent_inversion import (
            LatentTraversalSettingsDialog,
        )

        dialog = LatentTraversalSettingsDialog()

        dialog._widgets['seed_strategy'].setCurrentText('nearest')
        assert not dialog._widgets['target_n_atoms'].isEnabled()
        assert not dialog._widgets['size_tolerance'].isEnabled()

        dialog._widgets['seed_strategy'].setCurrentText('by_size')
        assert dialog._widgets['target_n_atoms'].isEnabled()
        assert dialog._widgets['size_tolerance'].isEnabled()

        dialog._widgets['seed_strategy'].setCurrentText('fixed_size')
        assert dialog._widgets['target_n_atoms'].isEnabled()
        assert not dialog._widgets['size_tolerance'].isEnabled()


class TestSidePanelLayout:
    """Structure viewer and traversal are two views of one side panel."""

    def test_starts_on_the_structure_view(self, panel):
        assert panel._side_panel.current_view() == panel._structure_view_idx

    def test_traversal_controls_live_in_the_traversal_pane(self, panel):
        assert panel._traversal_pane.isAncestorOf(panel._mark_btn)
        assert panel._traversal_pane.isAncestorOf(panel._clear_btn)
        assert panel._traversal_pane.isAncestorOf(panel._configure_btn)
        assert panel._traversal_pane.isAncestorOf(panel._run_btn)
        assert panel._traversal_pane.isAncestorOf(panel._cancel_btn)
        assert panel._structure_pane.isAncestorOf(panel._viewer)

    def test_selecting_a_structure_shows_the_viewer(self, panel):
        panel._side_panel.set_current_view(panel._traversal_view_idx)
        panel._mark_btn.setChecked(False)

        _click(panel, *panel._npz_data['latent_space'][0])

        assert panel._side_panel.current_view() == panel._structure_view_idx

    def test_leaving_the_traversal_view_disarms_marking(self, panel):
        panel._side_panel.set_current_view(panel._traversal_view_idx)
        panel._mark_btn.setChecked(True)

        panel._side_panel._buttons[panel._structure_view_idx].click()

        assert not panel._mark_btn.isChecked()
        assert not panel._mark_mode

    def test_traversal_view_is_hidden_without_a_map(self, qapp, project):
        from atlas.core.gui.widgets.boundary_panel import BoundaryPanel

        panel = BoundaryPanel(project)
        buttons = panel._side_panel._buttons
        assert buttons[panel._traversal_view_idx].isHidden()
        assert panel._side_panel.current_view() == panel._structure_view_idx

    def test_theme_is_forwarded_to_the_side_panel(self, panel):
        from atlas.core.gui.themes import DEFAULT_THEME

        panel.set_theme(DEFAULT_THEME)

        assert panel._side_panel._theme_name == DEFAULT_THEME


class TestWaypointList:
    """The marked-points list mirrors the path state."""

    def test_list_mirrors_the_marks(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.5, 0.25)
        _click(panel, -1.0, 1.0)

        assert panel._waypoint_list.count() == 2
        assert panel._waypoint_list.item(0).text().startswith('1')
        assert '0.5000' in panel._waypoint_list.item(0).text()
        assert panel._remove_btn.isEnabled()

    def test_removing_the_selected_row_removes_the_waypoint(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        _click(panel, 2.0, 2.0)
        panel._waypoint_list.setCurrentRow(0)

        panel._on_remove_selected_waypoint()

        assert panel._waypoints == [(2.0, 2.0)]
        assert panel._waypoint_list.count() == 1

    def test_double_click_removes_that_row(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        _click(panel, 1.0, 1.0)
        _click(panel, 2.0, 2.0)

        panel._on_waypoint_double_clicked(panel._waypoint_list.item(1))

        assert panel._waypoints == [(0.0, 0.0), (2.0, 2.0)]

    def test_right_click_removal_keeps_the_list_in_sync(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        _click(panel, 2.0, 2.0)

        panel._on_mouse_release(FakeEvent(panel._hover_ax, 1.98, 2.02, button=3))

        assert panel._waypoint_list.count() == 1

    def test_clear_points_empties_the_list(self, panel):
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)

        panel._on_clear_points()

        assert panel._waypoint_list.count() == 0
        assert not panel._remove_btn.isEnabled()


class TestTraversalWarning:
    """The gate reason is surfaced as a dialog whenever traversal is opened."""

    def test_warns_every_time_traversal_is_opened(self, panel, monkeypatch):
        asks = []
        monkeypatch.setattr(
            panel,
            '_ask_recompute_or_close',
            lambda fixable: asks.append(fixable) or False,
        )
        panel._side_panel.set_current_view(panel._structure_view_idx)

        for _ in range(2):
            panel._side_panel._buttons[panel._traversal_view_idx].click()
            panel._side_panel._buttons[panel._structure_view_idx].click()

        # The fixture map has no autoencoder, so regeneration is the offer.
        assert asks == [True, True]

    def test_no_warning_when_the_gate_is_available(self, panel, monkeypatch):
        from atlas.core.gui.widgets.latent_inversion import TraversalGate

        monkeypatch.setattr(
            panel, '_traversal_availability', lambda: TraversalGate(True, '', False)
        )
        asks = []
        monkeypatch.setattr(
            panel,
            '_ask_recompute_or_close',
            lambda fixable: asks.append(fixable) or False,
        )
        panel._side_panel.set_current_view(panel._structure_view_idx)

        panel._side_panel._buttons[panel._traversal_view_idx].click()

        assert asks == []

    def test_regenerate_action_opens_the_generation_view(self, panel, monkeypatch):
        monkeypatch.setattr(panel, '_ask_recompute_or_close', lambda fixable: True)
        panel._side_panel.set_current_view(panel._structure_view_idx)

        panel._side_panel._buttons[panel._traversal_view_idx].click()

        assert panel._side_panel.current_view() == panel._generation_view_idx

    def test_recompute_button_follows_the_gate(self, panel, monkeypatch):
        from atlas.core.gui.widgets.latent_inversion import TraversalGate

        assert not panel._recompute_btn.isHidden()

        monkeypatch.setattr(
            panel,
            '_traversal_availability',
            lambda: TraversalGate(False, 'no fix for this', False),
        )
        panel._update_traversal_controls()
        assert panel._recompute_btn.isHidden()

        monkeypatch.setattr(
            panel,
            '_traversal_availability',
            lambda: TraversalGate(False, 'regenerating helps', True),
        )
        panel._update_traversal_controls()
        assert not panel._recompute_btn.isHidden()


class _FakeRunner(QObject):
    """Stand-in for ProcessRunner that records the command it was given."""

    log_message = Signal(str)
    process_finished = Signal(int)

    def __init__(self, command, parent=None, *, cwd=None, log_file=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd
        self.log_file = log_file

    def start(self):
        pass

    def isRunning(self):  # noqa: N802 - Qt naming
        return False

    def stop(self):
        pass

    def wait(self, _timeout):
        return True


class TestGenerationPane:
    """The third side-panel view rebuilds the latent space."""

    def test_generation_view_lives_in_the_side_panel(self, panel):
        pane = panel._generation_pane
        assert panel._side_panel._stack.widget(panel._generation_view_idx) is pane
        assert pane.isAncestorOf(pane._generate_btn)
        assert pane.isAncestorOf(pane._descriptor_combo)

    def test_soap_selection_shows_the_warning(self, panel):
        pane = panel._generation_pane
        assert pane._descriptor_combo.itemData(0)['type'] == 'SOAP'

        pane._descriptor_combo.setCurrentIndex(0)
        assert not pane._soap_warning.isHidden()

        pane._descriptor_combo.setCurrentIndex(1)
        assert pane._soap_warning.isHidden()

    def test_current_map_summary_reports_provenance(self, panel):
        pane = panel._generation_pane

        assert 'MACE' in pane._map_label.text()

        pane.set_current_map(None)
        assert 'No latent-space map' in pane._map_label.text()

    def test_generate_is_disabled_without_a_database(self, qapp, project):
        from atlas.core.gui.widgets.latent_generation import LatentGenerationPane

        pane = LatentGenerationPane(project)

        assert not pane._generate_btn.isEnabled()
        assert not pane._go_config_btn.isHidden()

    def test_traversal_running_blocks_generation(self, panel):
        pane = panel._generation_pane
        db_path = panel._project.init_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b'')
        panel.refresh()
        assert pane._generate_btn.isEnabled()

        pane.set_traversal_running(True)
        assert not pane._generate_btn.isEnabled()
        assert pane._generate_btn.toolTip() == 'A traversal is running.'

        pane.set_traversal_running(False)
        assert pane._generate_btn.isEnabled()

    def test_generation_running_disables_traversal_controls(self, panel, monkeypatch):
        from atlas.core.gui.widgets.latent_inversion import TraversalGate

        monkeypatch.setattr(
            panel, '_traversal_availability', lambda: TraversalGate(True, '', False)
        )
        panel._update_traversal_controls()
        assert panel._mark_btn.isEnabled()

        panel._generation_pane._set_busy(True)
        assert not panel._mark_btn.isEnabled()
        assert not panel._run_btn.isEnabled()

        panel._generation_pane._set_busy(False)
        assert panel._mark_btn.isEnabled()

    def test_run_saves_the_config_and_builds_the_command(
        self, qapp, project, monkeypatch
    ):
        import tomli

        import atlas.core.gui.widgets.latent_generation as lg

        db_path = project.init_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(b'')

        runners = []

        def _make_runner(command, parent=None, *, cwd=None, log_file=None):
            runner = _FakeRunner(command, parent, cwd=cwd, log_file=log_file)
            runners.append(runner)
            return runner

        monkeypatch.setattr(lg, 'ProcessRunner', _make_runner)

        pane = lg.LatentGenerationPane(project)
        pane._descriptor_combo.setCurrentIndex(1)  # first MACE entry
        pane._boundary_combo.setCurrentIndex(1)  # morphological closing
        updates = []
        pane.config_updated.connect(lambda: updates.append(True))

        pane.run()

        assert updates == [True]
        snapshot = project.active_config('database_generation')[1]
        concave = tomli.loads(snapshot)['concave_hull']
        assert concave['descriptor'] == 'MACE'
        assert concave['descriptor_model'].startswith('mace:')
        assert concave['boundary_method'] == 'morphological_closing'
        assert concave['gen_concave_hull'] is True

        assert len(runners) == 1
        command = runners[0].command
        assert command[:2] == ['atl_gen_init_db', 'boundary']
        assert str(db_path) in command
        assert pane.is_running()

        pane._on_finished(0)
        assert not pane.is_running()

    def test_soap_selection_leaves_the_model_out_of_the_config(self, panel):
        pane = panel._generation_pane
        pane._descriptor_combo.setCurrentIndex(0)

        values = pane._selected_config_values()

        assert values['descriptor'] == 'SOAP'
        assert 'descriptor_model' not in values

    def test_pane_loads_settings_from_the_config(self, qapp, project):
        from atlas.core.gui.widgets.latent_generation import (
            LatentGenerationPane,
            merge_concave_config,
        )

        project.save_config_snapshot(
            'database_generation',
            merge_concave_config(
                '',
                {
                    'descriptor': 'MACE',
                    'descriptor_model': 'mace:mp-medium',
                    'boundary_method': 'morphological_closing',
                    'device': 'cuda',
                },
            ),
        )

        pane = LatentGenerationPane(project)

        assert pane._descriptor_combo.currentData()['model'] == 'mace:mp-medium'
        assert pane._boundary_combo.currentData() == 'morphological_closing'
        assert pane._device_combo.currentText() == 'cuda'

    def test_custom_model_path_is_added_to_the_combo(self, qapp, project):
        from atlas.core.gui.widgets.latent_generation import (
            LatentGenerationPane,
            merge_concave_config,
        )

        project.save_config_snapshot(
            'database_generation',
            merge_concave_config(
                '',
                {
                    'descriptor': 'MACE',
                    'descriptor_model': '/models/my_trained.model',
                },
            ),
        )

        pane = LatentGenerationPane(project)

        data = pane._descriptor_combo.currentData()
        assert data['type'] == 'MACE'
        assert data['model'] == '/models/my_trained.model'


class TestConfigureTraversal:
    """The settings dialog must be run through its class-level Qt enum."""

    def test_acceptance_saves_the_settings(self, panel, monkeypatch):
        from PySide6.QtWidgets import QDialog

        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets import latent_inversion as li

        class FakeDialog:
            def __init__(self, *_args, **_kwargs):
                self._settings = LatentInversionSettings(n_structures=3)

            def exec(self):
                return QDialog.DialogCode.Accepted

            def settings(self):
                return self._settings

        monkeypatch.setattr(li, 'LatentTraversalSettingsDialog', FakeDialog)

        assert panel._on_configure() is True
        assert li.load_settings(panel._project).n_structures == 3

    def test_rejection_keeps_the_settings_untouched(self, panel, monkeypatch):
        from PySide6.QtWidgets import QDialog

        from atlas.core.gui.widgets import latent_inversion as li

        class FakeDialog:
            def __init__(self, *_args, **_kwargs):
                pass

            def exec(self):
                return QDialog.DialogCode.Rejected

            def settings(self):
                raise AssertionError('settings() must not be read on cancel')

        monkeypatch.setattr(li, 'LatentTraversalSettingsDialog', FakeDialog)

        assert panel._on_configure() is False
        assert not li.has_saved_settings(panel._project)

    def test_too_few_structures_reopens_the_settings(self, panel, monkeypatch):
        from atlas.active_learning.extrapolation.latent_inversion import (
            LatentInversionSettings,
        )
        from atlas.core.gui.widgets import latent_inversion as li

        li.save_settings(
            panel._project,
            LatentInversionSettings(n_structures=1, snap_waypoints=True),
        )
        panel._mark_btn.setChecked(True)
        _click(panel, 0.0, 0.0)
        _click(panel, 1.0, 1.0)

        opened = []
        monkeypatch.setattr(
            panel, '_on_configure', lambda: opened.append(True) or False
        )

        panel._on_run_traversal()

        # Two points cannot be covered by one structure, so the settings are
        # reopened; cancelling that dialog aborts the run cleanly.
        assert opened == [True]
        assert panel._worker is None

    def test_traversal_progress_drives_the_shared_bar(self, panel):
        updates = []
        panel.progress_changed.connect(lambda *args: updates.append(args))

        panel._on_traversal_progress(3, 7, 'Decoded 3 of 7')

        assert updates == [(3, 7, 'Decoded 3 of 7')]

    def test_generation_busy_resets_the_shared_bar_to_indeterminate(self, panel):
        busy = []
        progress = []
        panel.busy_changed.connect(busy.append)
        panel.progress_changed.connect(lambda *args: progress.append(args))

        panel._on_generation_busy(True)

        assert busy == [True]
        assert progress == [(0, 0, 'Generating latent space...')]

    def test_traversal_logs_points_and_finish(self, panel, monkeypatch):
        from atlas.core.gui.widgets import latent_inversion as li

        class FakeResultsDialog(QObject):
            structures_added = Signal(int)

            def __init__(self, *_args, **_kwargs):
                super().__init__()

            def exec(self):
                return None

        monkeypatch.setattr(li, 'LatentTraversalResultsDialog', FakeResultsDialog)

        messages = []
        panel.log_message.connect(messages.append)

        # No worker is started: only the logging seams are exercised.
        panel._worker = None
        panel._on_point_ready(
            {
                'latent': (0.1, 0.1),
                'latent_err': 1.5e-3,
                'physical': True,
                'n_fragments': 1,
            }
        )
        panel._on_traversal_finished(
            [{'point_index': 0, 'latent': (0.1, 0.1), 'physical': True}], ''
        )

        assert any('Decoded point 1' in message for message in messages)
        assert any('finished' in message for message in messages)
        assert panel._worker is None
