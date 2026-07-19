"""Show current ATLAS setup status."""

from __future__ import annotations

import json
import os

from rich import print as rprint
from rich.table import Table

from atlas.core import code_utils as atl_cut
from atlas.core.command_line.setup._common import check_aiida, heading


def show_status() -> None:
    """Display a summary of the current ATLAS configuration."""
    heading(
        'ATLAS Setup Status',
        'Overview of your current ATLAS and AiiDA configuration.\n'
        'Run individual subcommands to set up missing components:\n'
        '  atl_init_setup mpkey | hftoken | profile | computer | code',
    )

    # MP API key
    _show_mp_status()
    rprint()

    # Hugging Face token
    _show_hf_status()
    rprint()

    # AiiDA
    _show_aiida_status()


def _show_hf_status() -> None:
    config_dir = atl_cut.get_config_path() / 'atl'
    secrets_path = config_dir / 'secrets.json'

    table = Table(title='Hugging Face', show_lines=False)
    table.add_column('Item', style='cyan')
    table.add_column('Status')

    token = None
    if secrets_path.exists():
        try:
            token = json.loads(secrets_path.read_text()).get('HF_TOKEN')
        except Exception:
            token = None

    if token:
        masked = token[:4] + '...' + token[-4:] if len(token) > 8 else '***'
        table.add_row('Token', f'[green]Configured[/green] ({masked})')
        table.add_row('Path', str(secrets_path))
    elif os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        table.add_row('Token', '[green]Set via environment variable[/green]')
    else:
        table.add_row(
            'Token',
            '[yellow]Not configured[/yellow] (only needed for gated models)',
        )

    rprint(table)


def _show_mp_status() -> None:
    config_dir = atl_cut.get_config_path() / 'atl'
    secrets_path = config_dir / 'secrets.json'

    table = Table(title='Materials Project', show_lines=False)
    table.add_column('Item', style='cyan')
    table.add_column('Status')

    if secrets_path.exists():
        try:
            data = json.loads(secrets_path.read_text())
            key = data.get('API_KEY', '')
            masked = key[:4] + '...' + key[-4:] if len(key) > 8 else '***'
            table.add_row('API key', f'[green]Configured[/green] ({masked})')
            table.add_row('Path', str(secrets_path))
        except Exception:
            table.add_row('API key', '[yellow]File exists but unreadable[/yellow]')
    elif os.environ.get('MP_API_KEY'):
        table.add_row('API key', '[green]Set via MP_API_KEY env var[/green]')
    else:
        table.add_row('API key', '[red]Not configured[/red]')

    cache_dir = atl_cut.get_cache_path() / 'mdb'
    if cache_dir.is_dir():
        table.add_row('Cache directory', f'[green]{cache_dir}[/green]')
    else:
        table.add_row('Cache directory', '[yellow]Not created[/yellow]')

    rprint(table)


def _show_aiida_status() -> None:
    if not check_aiida():
        rprint('[yellow]AiiDA is not installed.[/yellow]')
        return

    from atlas.core.command_line.setup._common import (
        list_codes,
        list_computers,
        list_profiles,
    )

    profiles, default = list_profiles()

    table = Table(title='AiiDA', show_lines=False)
    table.add_column('Item', style='cyan')
    table.add_column('Details')

    table.add_row('Installed', '[green]Yes[/green]')

    if profiles:
        profile_list = ', '.join(
            f'[bold]{p}[/bold]' if p == default else p for p in profiles
        )
        table.add_row('Profiles', profile_list)
        if default:
            table.add_row('Default profile', default)
    else:
        table.add_row('Profiles', '[yellow]None[/yellow]')

    computers = list_computers()
    if computers:
        table.add_row('Computers', ', '.join(computers))
    else:
        table.add_row('Computers', '[yellow]None[/yellow]')

    codes = list_codes()
    if codes:
        table.add_row('Codes', ', '.join(codes))
    else:
        table.add_row('Codes', '[yellow]None[/yellow]')

    rprint(table)
