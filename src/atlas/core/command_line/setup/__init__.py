"""ATLAS setup subcommands."""

from atlas.core.command_line.setup._code import setup_code
from atlas.core.command_line.setup._computer import setup_computer
from atlas.core.command_line.setup._hftoken import setup_hf_token
from atlas.core.command_line.setup._mpkey import setup_mp_key
from atlas.core.command_line.setup._potcar import setup_potcar
from atlas.core.command_line.setup._profile import setup_profile
from atlas.core.command_line.setup._status import show_status

__all__ = [
    'setup_code',
    'setup_computer',
    'setup_hf_token',
    'setup_mp_key',
    'setup_potcar',
    'setup_profile',
    'show_status',
]
