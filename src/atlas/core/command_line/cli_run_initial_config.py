"""Run initial configuration to set up the ATLAS environment."""

from __future__ import annotations

import argparse
import pathlib as pl
import warnings

import rich.prompt as rp

from atlas.core import code_utils as atl_cut

warnings.filterwarnings('ignore')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='atl_init_setup',
        description=(
            'ATLAS setup wizard, configure MP API key, Hugging Face token, '
            'AiiDA profiles, computers, and codes.'
        ),
    )
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('mpkey', help='Set up the Materials Project API key')
    sub.add_parser('hftoken', help='Set up the Hugging Face token (gated models)')
    sub.add_parser('profile', help='Create or select an AiiDA profile')
    sub.add_parser('computer', help='Set up an AiiDA computer')
    sub.add_parser('code', help='Set up an AiiDA code (VASP, LAMMPS, MACE, ...)')
    sub.add_parser('potcar', help='Upload a VASP POTCAR potential family')
    sub.add_parser('status', help='Show current configuration status')

    return parser


def _run_full_wizard() -> None:
    """Run all setup steps interactively."""
    from atlas.core.command_line.setup import (
        setup_code,
        setup_computer,
        setup_hf_token,
        setup_mp_key,
        setup_potcar,
        setup_profile,
        show_status,
    )

    setup_mp_key()

    if rp.Confirm.ask(
        '\nSet up a Hugging Face token for gated pretrained models '
        '(fairchem/UMA, EquiformerV2)?',
        default=False,
    ):
        setup_hf_token()

    if rp.Confirm.ask('\nSet up AiiDA for active learning?', default=False):
        setup_profile()

        if rp.Confirm.ask('\nAdd a compute cluster?', default=False):
            setup_computer()

        if rp.Confirm.ask('\nSet up a remote code?', default=False):
            setup_code()

        if rp.Confirm.ask('\nUpload a VASP POTCAR family?', default=False):
            setup_potcar()

    print()
    show_status()


def run_initial_config() -> None:
    atl_cut.init_logger(source=pl.Path(__file__).stem, log_path='/tmp')

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        _run_full_wizard()
        return

    from atlas.core.command_line.setup import (
        setup_code,
        setup_computer,
        setup_hf_token,
        setup_mp_key,
        setup_potcar,
        setup_profile,
        show_status,
    )

    dispatch = {
        'mpkey': setup_mp_key,
        'hftoken': setup_hf_token,
        'profile': setup_profile,
        'computer': setup_computer,
        'code': setup_code,
        'potcar': setup_potcar,
        'status': show_status,
    }
    dispatch[args.command]()
