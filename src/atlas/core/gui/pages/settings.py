"""Settings page (theme picker, AiiDA profile, MP API key)."""

from __future__ import annotations

import json
import os
import pathlib

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from atlas.core.gui.pages.base import WorkflowPage
from atlas.core.gui.themes import (
    save_global_toml_theme,
    saved_global_theme,
    saved_global_toml_theme,
    theme_names,
)
from atlas.core.gui.widgets.toml_editor import TomlHighlighter


class SettingsPage(WorkflowPage):
    """Settings: app theme picker, TOML syntax theme, AiiDA profile, MP API key."""

    DISPLAY_NAME = 'Settings'
    NAVIGATION_KEY = 'settings'

    theme_changed = Signal(str)
    app_theme_changed = Signal(str)

    def __init__(
        self,
        project,
        schema_data,
        application_font,
        log,
        navigate,
        notification=None,
        parent=None,
    ):
        super().__init__(
            project,
            schema_data,
            application_font,
            log,
            navigate,
            notification,
            parent,
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(16)

        # ── Appearance ──────────────────────────────────────────────
        appearance_group = QGroupBox('Appearance')
        appearance_form = QFormLayout(appearance_group)

        self.app_theme_combo = QComboBox()
        for name in theme_names():
            self.app_theme_combo.addItem(name)
        saved_theme = saved_global_theme()
        idx = self.app_theme_combo.findText(saved_theme)
        if idx >= 0:
            self.app_theme_combo.setCurrentIndex(idx)
        self.app_theme_combo.currentTextChanged.connect(self._on_app_theme_changed)
        appearance_form.addRow('App theme', self.app_theme_combo)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(TomlHighlighter.THEMES.keys()))
        saved_toml = saved_global_toml_theme()
        toml_idx = self.theme_combo.findText(saved_toml)
        if toml_idx >= 0:
            self.theme_combo.setCurrentIndex(toml_idx)
        self.theme_combo.currentTextChanged.connect(self._on_toml_theme_changed)
        appearance_form.addRow('TOML syntax theme', self.theme_combo)

        outer.addWidget(appearance_group)

        # ── AiiDA ───────────────────────────────────────────────────
        aiida_group = QGroupBox('AiiDA')
        aiida_form = QFormLayout(aiida_group)

        self._profile_combo = QComboBox()
        self._profile_combo.setMinimumWidth(250)
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        aiida_form.addRow('AiiDA profile', self._profile_combo)

        self._profile_status = QLabel()
        self._profile_status.setWordWrap(True)
        aiida_form.addRow('', self._profile_status)

        refresh_row = QHBoxLayout()
        refresh_btn = QPushButton('Refresh profiles')
        refresh_btn.setFixedWidth(140)
        refresh_btn.clicked.connect(self._refresh_profiles)
        refresh_row.addWidget(refresh_btn)
        refresh_row.addStretch()
        aiida_form.addRow('', refresh_row)

        outer.addWidget(aiida_group)

        # ── Materials Project ───────────────────────────────────────
        mp_group = QGroupBox('Materials Project')
        mp_form = QFormLayout(mp_group)

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        self._api_key_edit.setPlaceholderText('Enter your MP API key')
        self._api_key_edit.setMinimumWidth(350)
        mp_form.addRow('API key', self._api_key_edit)

        key_buttons = QHBoxLayout()
        self._toggle_vis_btn = QPushButton('Show')
        self._toggle_vis_btn.setFixedWidth(60)
        self._toggle_vis_btn.clicked.connect(self._toggle_key_visibility)
        key_buttons.addWidget(self._toggle_vis_btn)

        save_key_btn = QPushButton('Save')
        save_key_btn.setFixedWidth(60)
        save_key_btn.clicked.connect(self._save_api_key)
        key_buttons.addWidget(save_key_btn)

        key_buttons.addStretch()

        self._api_key_source = QLabel()
        self._api_key_source.setStyleSheet('color: palette(mid);')
        key_buttons.addWidget(self._api_key_source)

        mp_form.addRow('', key_buttons)
        outer.addWidget(mp_group)

        # ── Hugging Face ────────────────────────────────────────────
        hf_group = QGroupBox('Hugging Face')
        hf_form = QFormLayout(hf_group)

        self._hf_token_edit = QLineEdit()
        self._hf_token_edit.setEchoMode(QLineEdit.Password)
        self._hf_token_edit.setPlaceholderText(
            'Token for gated models (fairchem/UMA, EquiformerV2)'
        )
        self._hf_token_edit.setMinimumWidth(350)
        hf_form.addRow('Token', self._hf_token_edit)

        hf_buttons = QHBoxLayout()
        self._toggle_hf_vis_btn = QPushButton('Show')
        self._toggle_hf_vis_btn.setFixedWidth(60)
        self._toggle_hf_vis_btn.clicked.connect(self._toggle_hf_visibility)
        hf_buttons.addWidget(self._toggle_hf_vis_btn)

        save_hf_btn = QPushButton('Save')
        save_hf_btn.setFixedWidth(60)
        save_hf_btn.clicked.connect(self._save_hf_token)
        hf_buttons.addWidget(save_hf_btn)

        hf_buttons.addStretch()

        self._hf_token_source = QLabel()
        self._hf_token_source.setStyleSheet('color: palette(mid);')
        hf_buttons.addWidget(self._hf_token_source)

        hf_form.addRow('', hf_buttons)
        outer.addWidget(hf_group)

        # ── Setup Status ───────────────────────────────────────────
        from atlas.core.gui.widgets.setup_wizard import SetupStatusPanel

        self._setup_status = SetupStatusPanel()
        self._setup_status.setup_completed.connect(self.workflow_state_changed)
        outer.addWidget(self._setup_status)

        outer.addStretch()

    def on_shown(self) -> None:
        self._load_api_key()
        self._update_key_source_label()
        self._load_hf_token()
        self._update_hf_token_source_label()
        self._populate_aiida_profiles()
        if not _current_aiida_profile():
            profiles, default = _list_aiida_profiles()
            if default:
                self._on_profile_changed(default)
        self._update_profile_status()
        self._setup_status.refresh()

    # ── Appearance ──────────────────────────────────────────────────

    def _on_app_theme_changed(self, name: str) -> None:
        self.app_theme_changed.emit(name)

    def _on_toml_theme_changed(self, name: str) -> None:
        save_global_toml_theme(name)
        self.theme_changed.emit(name)

    # ── AiiDA profiles ──────────────────────────────────────────────

    def _populate_aiida_profiles(self) -> None:
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        profiles, default = _list_aiida_profiles()
        if profiles is None:
            self._profile_combo.addItem('(AiiDA not available)')
            self._profile_combo.setEnabled(False)
        elif not profiles:
            self._profile_combo.addItem('(no profiles found)')
            self._profile_combo.setEnabled(False)
        else:
            self._profile_combo.setEnabled(True)
            for p in profiles:
                self._profile_combo.addItem(p)
            current = _current_aiida_profile()
            if current and current in profiles:
                self._profile_combo.setCurrentText(current)
            elif default and default in profiles:
                self._profile_combo.setCurrentText(default)
        self._profile_combo.blockSignals(False)

    def _refresh_profiles(self) -> None:
        self._populate_aiida_profiles()
        self._update_profile_status()

    def _on_profile_changed(self, name: str) -> None:
        if name.startswith('('):
            return
        current = _current_aiida_profile()
        if current == name:
            return
        try:
            from aiida import load_profile

            load_profile(name, allow_switch=True)
            self._update_profile_status()
            self._log(f'Loaded AiiDA profile: {name}')
        except Exception as exc:
            self._update_profile_status()
            self._log(f'Failed to load AiiDA profile "{name}": {exc}')

    def _update_profile_status(self) -> None:
        current = _current_aiida_profile()
        if current:
            self._profile_status.setText(f'Active profile: {current}')
            self._profile_status.setStyleSheet(
                'color: #4caf50; border: none; background: transparent;'
            )
        else:
            self._profile_status.setText('No profile loaded')
            self._profile_status.setStyleSheet(
                'color: palette(mid); border: none; background: transparent;'
            )

    # ── MP API key ──────────────────────────────────────────────────

    def _load_api_key(self) -> None:
        key = _read_mp_api_key()
        if key:
            self._api_key_edit.setText(key)

    def _toggle_key_visibility(self) -> None:
        if self._api_key_edit.echoMode() == QLineEdit.Password:
            self._api_key_edit.setEchoMode(QLineEdit.Normal)
            self._toggle_vis_btn.setText('Hide')
        else:
            self._api_key_edit.setEchoMode(QLineEdit.Password)
            self._toggle_vis_btn.setText('Show')

    def _save_api_key(self) -> None:
        key = self._api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, 'MP API Key', 'Enter a key before saving.')
            return
        try:
            _write_mp_api_key(key)
            self._update_key_source_label()
            self._log('MP API key saved.')
        except Exception as exc:
            QMessageBox.critical(
                self,
                'MP API Key',
                f'Failed to save key: {exc}',
            )

    def _update_key_source_label(self) -> None:
        source = _mp_key_source()
        if source:
            self._api_key_source.setText(f'Source: {source}')
        else:
            self._api_key_source.setText('')

    # ── Hugging Face token ──────────────────────────────────────────

    def _load_hf_token(self) -> None:
        token = _read_hf_token()
        if token:
            self._hf_token_edit.setText(token)

    def _toggle_hf_visibility(self) -> None:
        if self._hf_token_edit.echoMode() == QLineEdit.Password:
            self._hf_token_edit.setEchoMode(QLineEdit.Normal)
            self._toggle_hf_vis_btn.setText('Hide')
        else:
            self._hf_token_edit.setEchoMode(QLineEdit.Password)
            self._toggle_hf_vis_btn.setText('Show')

    def _save_hf_token(self) -> None:
        token = self._hf_token_edit.text().strip()
        if not token:
            QMessageBox.warning(
                self, 'Hugging Face Token', 'Enter a token before saving.'
            )
            return
        try:
            _write_hf_token(token)
            self._update_hf_token_source_label()
            self._log('Hugging Face token saved.')
        except Exception as exc:
            QMessageBox.critical(
                self, 'Hugging Face Token', f'Failed to save token: {exc}'
            )

    def _update_hf_token_source_label(self) -> None:
        source = _hf_token_source()
        if source:
            self._hf_token_source.setText(f'Source: {source}')
        else:
            self._hf_token_source.setText('')


# ── helpers ─────────────────────────────────────────────────────────


def _list_aiida_profiles() -> tuple[list[str] | None, str | None]:
    """Return (profile_names, default_name) or (None, None) if unavailable."""
    try:
        from aiida.manage.configuration import get_config

        config = get_config()
        default = config.default_profile_name
        return list(config.profile_names), default
    except Exception:
        return None, None


def _current_aiida_profile() -> str | None:
    try:
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        return profile.name if profile is not None else None
    except Exception:
        return None


def _mp_config_dir() -> pathlib.Path:
    from atlas.core.code_utils import get_config_path

    return get_config_path() / 'atl'


def _read_mp_api_key() -> str | None:
    """Read MP API key from secrets.json or env var."""
    for path in [
        pathlib.Path('secrets.json'),
        _mp_config_dir() / 'secrets.json',
    ]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get('API_KEY', '')
            except Exception:
                pass
    return os.environ.get('MP_API_KEY', '')


def _write_mp_api_key(key: str) -> None:
    """Write MP API key to the ATLAS config directory."""
    config_dir = _mp_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = config_dir / 'secrets.json'
    data = {}
    if secrets_path.exists():
        try:
            with open(secrets_path) as f:
                data = json.load(f)
        except Exception:
            pass
    data['API_KEY'] = key
    with open(secrets_path, 'w') as f:
        json.dump(data, f, indent=2)


def _mp_key_source() -> str | None:
    """Return a description of where the key is coming from."""
    if pathlib.Path('secrets.json').exists():
        return 'secrets.json (current directory)'
    config_path = _mp_config_dir() / 'secrets.json'
    if config_path.exists():
        return str(config_path)
    if os.environ.get('MP_API_KEY'):
        return 'MP_API_KEY environment variable'
    return None


def _read_hf_token() -> str | None:
    """Read the Hugging Face token from secrets.json or env var."""
    for path in [
        pathlib.Path('secrets.json'),
        _mp_config_dir() / 'secrets.json',
    ]:
        if path.exists():
            try:
                with open(path) as f:
                    token = json.load(f).get('HF_TOKEN')
                if token:
                    return token
            except Exception:
                pass
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')


def _write_hf_token(token: str) -> None:
    """Write the Hugging Face token to the ATLAS config directory."""
    config_dir = _mp_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = config_dir / 'secrets.json'
    data = {}
    if secrets_path.exists():
        try:
            with open(secrets_path) as f:
                data = json.load(f)
        except Exception:
            pass
    data['HF_TOKEN'] = token
    with open(secrets_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.chmod(secrets_path, 0o600)


def _hf_token_source() -> str | None:
    """Return a description of where the HF token is coming from."""
    if pathlib.Path('secrets.json').exists():
        try:
            if json.loads(pathlib.Path('secrets.json').read_text()).get('HF_TOKEN'):
                return 'secrets.json (current directory)'
        except Exception:
            pass
    config_path = _mp_config_dir() / 'secrets.json'
    if config_path.exists():
        try:
            if json.loads(config_path.read_text()).get('HF_TOKEN'):
                return str(config_path)
        except Exception:
            pass
    if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        return 'HF_TOKEN environment variable'
    return None
