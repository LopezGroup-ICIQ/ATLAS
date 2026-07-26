"""Hugging Face token setup."""

from __future__ import annotations

import json
import os

import rich.prompt as rp

from atlas.core import code_utils as atl_cut
from atlas.core.command_line.setup._common import heading


def setup_hf_token() -> None:
    """Set up the Hugging Face token interactively (optional).

    The token is only needed for gated pretrained models: fairchem UMA/eSEN and
    the EquiformerV2 OMat24 checkpoints. It is stored alongside the Materials
    Project key in ``secrets.json``.
    """
    config_dir = atl_cut.get_config_path() / 'atl'
    config_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = config_dir / 'secrets.json'

    data = {}
    if secrets_path.exists():
        try:
            data = json.loads(secrets_path.read_text())
        except Exception:  # noqa: BLE001 - overwrite a malformed file below
            data = {}

    heading(
        'Hugging Face Token (optional)',
        'Only needed for gated pretrained models (fairchem UMA/eSEN, '
        'EquiformerV2 OMat24).\n\n'
        'Create a read token at: https://huggingface.co/settings/tokens\n'
        'You must also accept each gated model\'s licence on its Hugging Face '
        'page.',
    )

    if data.get('HF_TOKEN'):
        atl_cut.custom_print(
            f"A Hugging Face token is already configured at '{secrets_path}'.",
            'info',
        )
        if not rp.Confirm.ask('Replace it?', default=False):
            return

    token = rp.Prompt.ask(
        'Enter your Hugging Face token (leave blank to skip)',
        password=True,
        default='',
    )
    print()

    if not token or not token.strip():
        atl_cut.custom_print('No token entered. Skipped.', 'info')
        return

    data['HF_TOKEN'] = token.strip()
    secrets_path.write_text(json.dumps(data, indent=2) + '\n')
    os.chmod(secrets_path, 0o600)

    atl_cut.custom_print(f"Hugging Face token saved to '{secrets_path}'", 'done')
    atl_cut.custom_print(
        'File permissions set to 0o600 (owner read/write only).', 'info'
    )
