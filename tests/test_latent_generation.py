"""Tests for the boundary-only regeneration path.

Covers the config-to-call translation used by both the full database pipeline
and the ``atl_gen_init_db boundary`` command, the GUI config merge, and the
command the generation pane builds. No descriptors are computed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class _FakeDB:
    """Records the arguments ``descriptors_concave_hull`` was called with."""

    def __init__(self):
        self.calls: dict = {}

    def descriptors_concave_hull(self, **kwargs):
        self.calls.update(kwargs)


class TestGenerateLatentSpaceFromConfig:
    """The shared config -> call translation."""

    def test_relative_plot_filename_lands_beside_the_database(self, tmp_path):
        from atlas.core.initial_db import generate_latent_space_from_config

        db = _FakeDB()
        generate_latent_space_from_config(
            db,
            {'descriptor': 'MACE', 'boundary_method': 'concave_hull'},
            db_dir=tmp_path,
        )

        assert db.calls['descriptor_type'] == 'mace'
        assert (
            Path(db.calls['plot_filename']) == tmp_path / 'descriptors_concave_hull.png'
        )
        assert db.calls['boundary_method'] == 'concave_hull'

    def test_absolute_plot_filename_is_left_alone(self, tmp_path):
        from atlas.core.initial_db import generate_latent_space_from_config

        db = _FakeDB()
        target = tmp_path / 'somewhere' / 'map.png'
        generate_latent_space_from_config(
            db, {'descriptor': 'SOAP', 'plot_filename': str(target)}, db_dir=tmp_path
        )

        assert Path(db.calls['plot_filename']) == target

    def test_descriptor_model_and_device_reach_the_engine(self, tmp_path):
        from atlas.core.initial_db import generate_latent_space_from_config

        db = _FakeDB()
        generate_latent_space_from_config(
            db,
            {
                'descriptor': 'MACE',
                'descriptor_model': 'mace:mp-medium',
                'device': 'cpu',
            },
            db_dir=tmp_path,
        )

        assert db.calls['descriptor_settings'] == {'model': 'mace:mp-medium'}
        assert db.calls['device'] == 'cpu'

    def test_soap_ignores_the_model(self, tmp_path):
        from atlas.core.initial_db import generate_latent_space_from_config

        db = _FakeDB()
        generate_latent_space_from_config(
            db,
            {'descriptor': 'SOAP', 'descriptor_model': 'mace:mp-small'},
            db_dir=tmp_path,
        )

        assert db.calls['descriptor_settings'] == {}


class TestBoundaryCli:
    """``atl_gen_init_db boundary`` loads the database and delegates."""

    def test_gen_boundary_delegates_with_the_config_block(self, tmp_path, monkeypatch):
        import atlas.core.initial_db as idb
        from atlas.core.command_line import cli_generate_init_db as cli

        db_file = tmp_path / 'init_db.xz'
        db_file.write_bytes(b'')

        loaded = object()
        monkeypatch.setattr(
            idb.InitialDatabase,
            'load_database',
            classmethod(lambda cls, _path: loaded),
        )
        calls: dict = {}

        def _fake_helper(db, concave, db_dir, rng_seed):
            calls.update(db=db, concave=concave, db_dir=db_dir, rng_seed=rng_seed)

        monkeypatch.setattr(idb, 'generate_latent_space_from_config', _fake_helper)

        cli.gen_boundary(
            {
                'concave_hull': {'descriptor': 'MACE'},
                'database': {'rng_seed': 7},
            },
            str(db_file),
        )

        assert calls['db'] is loaded
        assert calls['concave'] == {'descriptor': 'MACE'}
        assert Path(calls['db_dir']) == tmp_path
        assert calls['rng_seed'] == 7

    def test_gen_boundary_without_a_seed_passes_none(self, tmp_path, monkeypatch):
        import atlas.core.initial_db as idb
        from atlas.core.command_line import cli_generate_init_db as cli

        db_file = tmp_path / 'init_db.xz'
        db_file.write_bytes(b'')
        monkeypatch.setattr(
            idb.InitialDatabase,
            'load_database',
            classmethod(lambda cls, _path: object()),
        )
        calls: dict = {}
        monkeypatch.setattr(
            idb,
            'generate_latent_space_from_config',
            lambda db, concave, db_dir, rng_seed: calls.update(rng_seed=rng_seed),
        )

        cli.gen_boundary({}, str(db_file))

        assert calls['rng_seed'] is None

    def test_missing_database_raises(self, tmp_path):
        from atlas.core.command_line import cli_generate_init_db as cli

        with pytest.raises(FileNotFoundError):
            cli.gen_boundary({}, str(tmp_path / 'absent.xz'))


class TestGenerationHelpers:
    """Pure helpers behind the generation pane."""

    def test_merge_concave_config_updates_and_enables(self):
        from atlas.core.gui.widgets.latent_generation import merge_concave_config

        merged = merge_concave_config(
            '[database]\ndatabase_name = "demo"\n',
            {'descriptor': 'MACE', 'descriptor_model': 'mace:mp-small'},
        )

        import tomli

        data = tomli.loads(merged)
        assert data['database']['database_name'] == 'demo'
        assert data['concave_hull']['descriptor'] == 'MACE'
        assert data['concave_hull']['descriptor_model'] == 'mace:mp-small'
        assert data['concave_hull']['gen_concave_hull'] is True

    def test_merge_concave_config_accepts_an_empty_snapshot(self):
        import tomli

        from atlas.core.gui.widgets.latent_generation import merge_concave_config

        data = tomli.loads(merge_concave_config('', {'device': 'cpu'}))

        assert data['concave_hull'] == {'device': 'cpu', 'gen_concave_hull': True}

    def test_descriptor_options_start_with_soap(self):
        from atlas.core.gui.widgets.latent_generation import descriptor_options

        options = descriptor_options()

        assert options[0][0] == 'SOAP'
        assert options[0][1] == ''
        assert all(spec.startswith('mace:') for _label, spec in options[1:])

    def test_build_boundary_command(self):
        from atlas.core.gui.widgets.latent_generation import build_boundary_command

        command = build_boundary_command(
            '/proj/database_generation_settings.toml', '/db/x.xz'
        )

        assert command == [
            'atl_gen_init_db',
            'boundary',
            '-c',
            '/proj/database_generation_settings.toml',
            '-d',
            '/db/x.xz',
        ]
