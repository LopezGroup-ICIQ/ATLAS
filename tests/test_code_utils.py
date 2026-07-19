"""Tests for code utility functions."""

import logging
import os
import pathlib
import tempfile
import warnings

import pytest

from atlas.core.code_utils import (
    ATLHighlighter,
    ATLRichHandler,
    deprecated,
    get_cache_path,
    get_config_path,
    init_config_dir,
    init_logger,
)


class TestATLHighlighter:
    """Tests for the ATLHighlighter Rich syntax highlighting class."""

    def test_base_style(self):
        h = ATLHighlighter()
        assert h.base_style == 'atl.'

    def test_highlights_contains_path_pattern(self):
        h = ATLHighlighter()
        patterns = [r'(?P<path>', r'(?P<number>', r'(?P<success>', r'(?P<failure>']
        for pattern_desc in patterns:
            found = any(pattern_desc in p for p in h.highlights)
            assert found, f'Pattern {pattern_desc} not found in highlights'


class TestATLRichHandler:
    """Tests for the ATLRichHandler custom Rich logging handler."""

    def test_level_map_contains_expected_keys(self):
        level_map = ATLRichHandler._LEVEL_MAP
        assert 'DEBUG' in level_map
        assert 'INFO' in level_map
        assert 'WARNING' in level_map
        assert 'ERROR' in level_map
        assert 'CRITICAL' in level_map
        assert 'REPORT' in level_map
        assert 'SUCCESS' in level_map

    def test_level_map_mappings(self):
        level_map = ATLRichHandler._LEVEL_MAP
        assert level_map['INFO'] == '[ i ]'
        assert level_map['WARNING'] == '[ ! ]'
        assert level_map['SUCCESS'] == '[ ✔ ]'
        assert level_map['ERROR'] == '[ X ]'


class TestInitLogger:
    """Tests for init_logger, the main logger initialization function."""

    def test_init_logger_returns_logger_and_path(self):
        logger, filepath = init_logger('test_source', show_log_path=False)
        assert isinstance(logger, logging.Logger)
        assert isinstance(filepath, str)
        assert '.log' in filepath
        assert logger.name == 'mdb'
        logger.handlers.clear()

    def test_init_logger_with_log_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger, filepath = init_logger(
                'test_source', log_path=tmp, show_log_path=False
            )
            assert tmp in filepath
            logger.handlers.clear()

    def test_init_logger_handlers(self):
        logger, filepath = init_logger('test_source', show_log_path=False)
        handler_names = [h.get_name() for h in logger.handlers]
        assert 'atl_rich_handler' in handler_names
        assert 'atl_file_handler' in handler_names
        logger.handlers.clear()

    def test_init_logger_level(self):
        logger, filepath = init_logger('test_source', show_log_path=False)
        assert logger.level == logging.DEBUG
        logger.handlers.clear()

    def test_logger_propagate_false(self):
        logger, filepath = init_logger('test_source', show_log_path=False)
        assert logger.propagate is False
        logger.handlers.clear()


class TestGetConfigPath:
    """Tests for get_config_path, the XDG config directory resolver."""

    def test_returns_pathlib_path(self):
        path = get_config_path()
        assert isinstance(path, pathlib.Path)

    def test_falls_back_to_home_config(self, monkeypatch):
        monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
        path = get_config_path()
        expected = pathlib.Path().home() / '.config'
        assert str(path) == str(expected)


class TestGetCachePath:
    """Tests for get_cache_path, the XDG cache directory resolver."""

    def test_returns_pathlib_path(self):
        path = get_cache_path()
        assert isinstance(path, pathlib.Path)

    def test_uses_literal_dollar_prefix_bug(self, monkeypatch):
        """Known bug: uses '$XDG_CACHE_HOME' instead of 'XDG_CACHE_HOME'.
        This means the env var is never picked up. This test verifies the bug exists.
        """
        monkeypatch.setenv('XDG_CACHE_HOME', '/custom/cache')
        path = get_cache_path()
        # If the bug is present, the env var with the literal '$' is not found
        # and it falls through to the home/.cache check
        assert str(path) != '/custom/cache'


class TestInitConfigDir:
    """Tests for init_config_dir, the configuration directory setup utility."""

    def test_creates_config_file(self, tmp_path):
        config_dir = tmp_path / 'config'
        success, created_dir = init_config_dir(config_dir, 'secrets.json')
        assert success is True
        assert (created_dir / 'secrets.json').exists()

    def test_returns_false_if_exists(self, tmp_path):
        config_dir = tmp_path / 'config'
        init_config_dir(config_dir, 'secrets.json')
        success, _ = init_config_dir(config_dir, 'secrets.json')
        assert success is False

    def test_writes_json_template(self, tmp_path):
        config_dir = tmp_path / 'config'
        init_config_dir(config_dir, 'secrets.json')
        content = (config_dir / 'mdb' / 'secrets.json').read_text()
        assert '"API_KEY": ""' in content

    def test_creates_atl_subdir(self, tmp_path):
        config_dir = tmp_path / 'config'
        init_config_dir(config_dir, 'test.json')
        assert (config_dir / 'mdb').exists()

    def test_file_permissions(self, tmp_path):
        config_dir = tmp_path / 'config'
        init_config_dir(config_dir, 'secrets.json')
        file_path = config_dir / 'mdb' / 'secrets.json'
        mode = os.stat(file_path).st_mode & 0o777
        assert mode == 0o700


class TestDeprecatedDecorator:
    """Tests for the @deprecated decorator and its warning behavior."""

    def test_emits_deprecation_warning(self):
        @deprecated(reason='Use new_func instead.')
        def old_func():
            return 'result'

        with pytest.warns(DeprecationWarning, match='old_func'):
            result = old_func()

        assert result == 'result'

    def test_includes_reason_in_warning(self):
        @deprecated(reason='Use new_func instead.')
        def old_func():
            pass

        with pytest.warns(DeprecationWarning) as record:
            old_func()

        assert len(record) == 1
        assert 'Use new_func instead' in str(record[0].message)

    def test_includes_since_ver_in_warning(self):
        @deprecated(reason='Use new_func instead.', since_ver='0.5.0')
        def old_func():
            pass

        with pytest.warns(DeprecationWarning) as record:
            old_func()

        assert "since version: '0.5.0'" in str(record[0].message)

    def test_preserves_function_name(self):
        @deprecated(reason='Use new_func instead.')
        def my_special_func():
            pass

        assert my_special_func.__name__ == 'my_special_func'

    def test_consecutive_calls_both_warn(self):
        @deprecated(reason='Use new_func instead.')
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            assert old_func() == 42
            assert old_func() == 42

        # Should have at least one DeprecationWarning per call
        dw = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dw) >= 2


class TestCustomPrint:
    """Test custom_print via the public API."""

    def test_custom_print_importable(self):
        from atlas.core.code_utils import custom_print

        assert callable(custom_print)

    def test_custom_print_info_type(self, mock_logger):
        from atlas.core.code_utils import custom_print

        _ = custom_print('test message', 'info', logger=mock_logger)
        mock_logger.log.assert_called_once()

    def test_custom_print_error_type(self, mock_logger):
        from atlas.core.code_utils import custom_print

        _ = custom_print('error message', 'error', logger=mock_logger)
        mock_logger.log.assert_called_once()

    def test_custom_print_warning_type(self, mock_logger):
        from atlas.core.code_utils import custom_print

        _ = custom_print('warning message', 'warn', logger=mock_logger)
        mock_logger.log.assert_called_once()

    def test_custom_print_done_type(self, mock_logger):
        from atlas.core.code_utils import custom_print

        _ = custom_print('done message', 'done', logger=mock_logger)
        mock_logger.log.assert_called_once()


class TestGatherHFToken:
    """Reading the Hugging Face token from secrets.json / environment."""

    def test_none_when_absent(self, monkeypatch, tmp_path):
        from atlas.core import code_utils as atl_cut

        monkeypatch.delenv('HF_TOKEN', raising=False)
        monkeypatch.delenv('HUGGING_FACE_HUB_TOKEN', raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(atl_cut, 'get_config_path', lambda: tmp_path)
        assert atl_cut.gather_hf_token() is None

    def test_read_from_cwd_secrets(self, monkeypatch, tmp_path):
        import json

        from atlas.core import code_utils as atl_cut

        monkeypatch.delenv('HF_TOKEN', raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'secrets.json').write_text(json.dumps({'HF_TOKEN': 'hf_abc'}))
        assert atl_cut.gather_hf_token() == 'hf_abc'

    def test_read_from_config_dir(self, monkeypatch, tmp_path):
        import json

        from atlas.core import code_utils as atl_cut

        monkeypatch.delenv('HF_TOKEN', raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / 'cfg'
        (cfg / 'atl').mkdir(parents=True)
        (cfg / 'atl' / 'secrets.json').write_text(json.dumps({'HF_TOKEN': 'hf_cfg'}))
        monkeypatch.setattr(atl_cut, 'get_config_path', lambda: cfg)
        # no secrets.json in the (empty) cwd
        assert atl_cut.gather_hf_token() == 'hf_cfg'

    def test_env_fallback(self, monkeypatch, tmp_path):
        from atlas.core import code_utils as atl_cut

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(atl_cut, 'get_config_path', lambda: tmp_path)
        monkeypatch.setenv('HF_TOKEN', 'hf_env')
        assert atl_cut.gather_hf_token() == 'hf_env'


class TestApplyHFToken:
    """Exporting the ATLAS-managed token to the environment."""

    def test_sets_env_from_secrets(self, monkeypatch, tmp_path):
        import json

        from atlas.core import code_utils as atl_cut

        monkeypatch.delenv('HF_TOKEN', raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(atl_cut, 'get_config_path', lambda: tmp_path)
        (tmp_path / 'secrets.json').write_text(json.dumps({'HF_TOKEN': 'hf_s'}))
        assert atl_cut.apply_hf_token() == 'hf_s'
        assert os.environ['HF_TOKEN'] == 'hf_s'

    def test_existing_env_takes_precedence(self, monkeypatch, tmp_path):
        import json

        from atlas.core import code_utils as atl_cut

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(atl_cut, 'get_config_path', lambda: tmp_path)
        (tmp_path / 'secrets.json').write_text(json.dumps({'HF_TOKEN': 'hf_s'}))
        monkeypatch.setenv('HF_TOKEN', 'hf_pre')
        assert atl_cut.apply_hf_token() == 'hf_pre'

    def test_no_token_leaves_env_unset(self, monkeypatch, tmp_path):
        from atlas.core import code_utils as atl_cut

        monkeypatch.delenv('HF_TOKEN', raising=False)
        monkeypatch.delenv('HUGGING_FACE_HUB_TOKEN', raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(atl_cut, 'get_config_path', lambda: tmp_path)
        assert atl_cut.apply_hf_token() is None
        assert 'HF_TOKEN' not in os.environ
