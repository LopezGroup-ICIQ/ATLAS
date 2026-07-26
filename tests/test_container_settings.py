"""Tests for per-backend container settings resolution."""

import logging

import pytest

import atlas  # noqa: F401  (resolves circular imports)
from atlas.active_learning import active_learning_utils as atl_al_ut
from atlas.active_learning.active_learning_utils import resolve_container_settings


@pytest.fixture(autouse=True)
def _reset_warning_state():
    """Clear the once-per-process warning cache between tests."""
    atl_al_ut._WARNED_CONTAINER_OVERRIDES.clear()
    yield
    atl_al_ut._WARNED_CONTAINER_OVERRIDES.clear()


@pytest.fixture
def mdb_logs():
    """Collect messages emitted through the 'mdb' logger.

    `custom_print` sets ``propagate = False`` on that logger, so pytest's
    ``caplog`` (which listens at the root logger) never sees these records.
    Attaching a handler to 'mdb' directly is independent of test order and of
    whichever handlers another test happened to configure first.
    """
    logger = logging.getLogger('mdb')
    messages: list[str] = []

    class _CollectHandler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _CollectHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

# Mirrors the shape produced by `[code.container]` in the shipped TOML template.
GLOBAL_SETTINGS = {
    'use_container': True,
    'image_name': '/images/atlas_mace.sif',
    'engine_command': 'singularity exec --bind .:/atl_data --nv {image_name}',
    'prepend_text': 'module load singularity',
}


class TestBackwardCompatibility:
    """A config without `per_backend` must behave exactly as it does today."""

    def test_settings_without_per_backend_are_unchanged(self):
        assert resolve_container_settings(GLOBAL_SETTINGS) == GLOBAL_SETTINGS

    def test_settings_without_per_backend_unchanged_with_backend_name(self):
        assert resolve_container_settings(GLOBAL_SETTINGS, 'mace') == GLOBAL_SETTINGS

    def test_none_returns_empty_dict(self):
        assert resolve_container_settings(None) == {}

    def test_none_with_backend_name_returns_empty_dict(self):
        assert resolve_container_settings(None, 'mace') == {}

    def test_empty_dict_returns_empty_dict(self):
        assert resolve_container_settings({}, 'mace') == {}


class TestOverrideResolution:
    """Per-backend overrides merge over the global settings."""

    def _config(self):
        return dict(
            GLOBAL_SETTINGS,
            per_backend={
                'allegro': {'image_name': '/images/atlas_allegro.sif'},
                'mace': {
                    'image_name': '/images/atlas_mace_gpu.sif',
                    'engine_command': 'docker run {image_name}',
                },
            },
        )

    def test_per_backend_key_is_stripped(self):
        resolved = resolve_container_settings(self._config(), 'allegro')
        assert 'per_backend' not in resolved

    def test_backend_without_override_falls_back_to_global(self):
        config = dict(GLOBAL_SETTINGS, per_backend={'allegro': {'image_name': 'a.sif'}})
        resolved = resolve_container_settings(config, 'mace')
        assert resolved['image_name'] == GLOBAL_SETTINGS['image_name']
        assert 'per_backend' not in resolved

    def test_no_backend_name_falls_back_to_global(self):
        resolved = resolve_container_settings(self._config())
        assert resolved['image_name'] == GLOBAL_SETTINGS['image_name']

    def test_override_replaces_image_name(self):
        resolved = resolve_container_settings(self._config(), 'allegro')
        assert resolved['image_name'] == '/images/atlas_allegro.sif'

    def test_non_overridden_keys_fall_back_to_global(self):
        resolved = resolve_container_settings(self._config(), 'allegro')
        assert resolved['engine_command'] == GLOBAL_SETTINGS['engine_command']
        assert resolved['prepend_text'] == GLOBAL_SETTINGS['prepend_text']
        assert resolved['use_container'] is True

    def test_override_can_replace_several_keys(self):
        resolved = resolve_container_settings(self._config(), 'mace')
        assert resolved['image_name'] == '/images/atlas_mace_gpu.sif'
        assert resolved['engine_command'] == 'docker run {image_name}'

    def test_distinct_backends_resolve_to_distinct_images(self):
        config = self._config()
        mace = resolve_container_settings(config, 'mace')
        allegro = resolve_container_settings(config, 'allegro')
        assert mace['image_name'] != allegro['image_name']

    def test_input_is_not_mutated(self):
        config = self._config()
        expected = dict(config)
        resolve_container_settings(config, 'allegro')
        assert config == expected
        assert 'per_backend' in config


class TestWarnings:
    """Resolver warnings compensate for the schema gap.

    The schema accepts `per_backend` without validating its contents, so the
    resolver warns rather than letting typos pass silently.

    These use the `mdb_logs` fixture rather than `capsys`/`caplog`: whether a
    warning reaches stdout depends on which handlers are attached, and hence on
    test order, and `custom_print` disables propagation to the root logger.
    """

    def test_unknown_override_key_warns_and_is_ignored(self, mdb_logs):
        config = dict(
            GLOBAL_SETTINGS,
            per_backend={'mace': {'image_nam': 'typo.sif'}},
        )
        resolved = resolve_container_settings(config, 'mace')
        assert 'image_nam' not in resolved
        assert resolved['image_name'] == GLOBAL_SETTINGS['image_name']
        assert any('image_nam' in message for message in mdb_logs)

    def test_unregistered_backend_warns_but_does_not_raise(self, mdb_logs):
        config = dict(
            GLOBAL_SETTINGS,
            per_backend={'not_a_backend': {'image_name': 'x.sif'}},
        )
        resolved = resolve_container_settings(config, 'mace')
        assert resolved['image_name'] == GLOBAL_SETTINGS['image_name']
        assert any('not_a_backend' in message for message in mdb_logs)

    def test_known_backend_does_not_warn(self, mdb_logs):
        config = dict(
            GLOBAL_SETTINGS,
            per_backend={'mace': {'image_name': 'ok.sif'}},
        )
        resolve_container_settings(config, 'mace')
        assert not any('Registered backends' in message for message in mdb_logs)

    def test_unregistered_backend_warns_only_once(self, mdb_logs):
        # The resolver runs at every code-building site on every AL iteration,
        # so a repeated warning would flood the loop logs.
        config = dict(
            GLOBAL_SETTINGS,
            per_backend={'not_a_backend': {'image_name': 'x.sif'}},
        )
        for _ in range(5):
            resolve_container_settings(config, 'mace')
        assert sum('not_a_backend' in message for message in mdb_logs) == 1
