"""Standalone CLI for cleaning an ATLAS initial database."""

import argparse
import pathlib as pl
import sys
import tomllib
import warnings
from argparse import RawTextHelpFormatter

# dscribe (and some other third-party packages) ship docstrings with invalid
# escape sequences. Python 3.12 reports them as SyntaxWarning at import time.
warnings.filterwarnings('ignore', category=SyntaxWarning, module='dscribe')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='spglib')

from rich.traceback import install as traceback_install  # noqa: E402

import atlas.core.code_utils as atl_cut  # noqa: E402
from atlas.core.filtering.db_cleanup import clean_database_file  # noqa: E402


def _read_config_file(config_file: pl.Path) -> dict:
    """Load a TOML configuration file."""
    try:
        with open(config_file, 'rb') as file_handle:
            return tomllib.load(file_handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"The config file '{config_file}' does not exist. "
            'Please use a different path or omit --config-file to use defaults.'
        ) from exc


def _load_config(config_file: pl.Path | None) -> dict:
    """Load the requested config, falling back to defaults when implicit."""
    if config_file is not None:
        return _read_config_file(config_file)

    default_config = pl.Path('./database_generation_settings.toml')
    if default_config.exists():
        return _read_config_file(default_config)

    return {}


def _print_summary(report: dict) -> None:
    """Print a human-readable cleanup summary."""
    atl_cut.custom_print(
        (
            f"Cleanup finished: kept {report.get('n_kept', 0)} of "
            f"{report.get('n_input', 0)} structures "
            f"(removed {report.get('n_removed', 0)})."
        ),
        'done',
    )

    if report.get('removed_by_filter'):
        atl_cut.custom_print(
            f"Removed by filters: {report['removed_by_filter']} "
            f"({report.get('by_filter', {})}).",
            'info',
        )

    if report.get('removed_by_duplicate'):
        atl_cut.custom_print(
            f"Removed as duplicates: {report['removed_by_duplicate']}.",
            'info',
        )

    if report.get('reassigned_ids'):
        atl_cut.custom_print(
            f"Reassigned {len(report['reassigned_ids'])} repeated atl_id values.",
            'warn',
        )

    if report.get('filter_failures'):
        atl_cut.custom_print(
            f"{len(report['filter_failures'])} filter calls failed and were "
            'skipped. See the JSON report for details.',
            'warn',
        )

    if report.get('invalid_filters'):
        atl_cut.custom_print(
            f"Ignored unknown filters: {report['invalid_filters']}.",
            'warn',
        )

    if report.get('output_path'):
        atl_cut.custom_print(f"Cleaned database: {report['output_path']}", 'info')

    if report.get('removed_path'):
        atl_cut.custom_print(f"Removed structures: {report['removed_path']}", 'info')

    if report.get('report_path'):
        atl_cut.custom_print(f"Cleanup report: {report['report_path']}", 'info')


def build_parser() -> argparse.ArgumentParser:
    """Create the ``atl_cleanup_db`` argument parser."""
    parser = argparse.ArgumentParser(
        prog='atl_cleanup_db',
        description=(
            'Clean an ATLAS initial database: standardize metadata, apply '
            'unphysical-structure filters, and remove duplicates.'
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        '--input',
        '-i',
        help='Path to the database (.xyz/.extxyz, .xz or .pkl).',
        type=pl.Path,
        required=False,
        metavar='PATH',
    )
    parser.add_argument(
        '--output',
        '-o',
        help=(
            'Destination for the cleaned database.\n'
            'Defaults to <input>_cleaned<suffix>.'
        ),
        type=pl.Path,
        default=None,
        metavar='PATH',
    )
    parser.add_argument(
        '--config-file',
        '-c',
        help=(
            'TOML file containing the [cleanup] and [struct_filters] sections.\n'
            'Defaults to ./database_generation_settings.toml when present.'
        ),
        type=pl.Path,
        default=None,
        metavar='PATH',
    )
    parser.add_argument(
        '--in-place',
        help='Overwrite the input database after creating a .bak backup.',
        action='store_true',
    )
    parser.add_argument(
        '--dry-run',
        help='Run the pipeline and write the report without writing structures.',
        action='store_true',
    )
    parser.add_argument(
        '--skip-duplicates',
        help='Disable SOAP duplicate detection.',
        action='store_true',
    )
    parser.add_argument(
        '--skip-filters',
        help='Disable the [struct_filters] unphysical-structure checks.',
        action='store_true',
    )
    parser.add_argument(
        '--skip-standardize',
        help='Do not migrate legacy metadata keys.',
        action='store_true',
    )
    parser.add_argument(
        '--no-removed-file',
        help='Do not write removed structures to a separate file.',
        action='store_true',
    )
    parser.add_argument(
        '--removed',
        help='Destination for removed structures in extended XYZ format.',
        type=pl.Path,
        default=None,
        metavar='PATH',
    )
    parser.add_argument(
        '--report',
        help='Destination for the JSON cleanup report.',
        type=pl.Path,
        default=None,
        metavar='PATH',
    )
    parser.add_argument(
        '--no-progress',
        help='Disable progress bars.',
        action='store_true',
    )
    parser.add_argument(
        '--list-filters',
        help='List the available structure filters and exit.',
        action='store_true',
    )

    return parser


def run_cleanup_database() -> None:
    """Entry point for ``atl_cleanup_db``."""
    parser = build_parser()
    args = parser.parse_args()

    if args.list_filters:
        from atlas.core.filtering.structure_filters import get_available_filters

        for filter_name in sorted(get_available_filters()):
            print(filter_name)
        return

    if args.input is None:
        parser.error('--input is required unless --list-filters is used.')

    config = _load_config(args.config_file)
    cleanup_config = dict(config.get('cleanup', {}))
    filter_settings = config.get('struct_filters', {})

    if not filter_settings and not args.skip_filters:
        atl_cut.custom_print(
            'No \\[struct_filters] section found; unphysical-structure filters '
            'are disabled. Pass --config-file to enable them.',
            'warn',
        )

    # The CLI was invoked explicitly, so the generation-only ``enabled`` flag
    # must not silently skip the requested cleanup.
    cleanup_config['enabled'] = True

    if args.skip_duplicates:
        cleanup_config['remove_duplicates'] = False
    if args.skip_filters:
        filter_settings = None
    if args.skip_standardize:
        cleanup_config['standardize_metadata'] = False
    if args.no_removed_file:
        cleanup_config['write_removed'] = False
    if args.removed is not None:
        cleanup_config['write_removed'] = True

    if args.in_place and args.output is not None:
        parser.error('--in-place and --output cannot be used together.')

    report = clean_database_file(
        input_path=args.input,
        output_path=args.output,
        cleanup_config=cleanup_config,
        filter_settings=filter_settings,
        in_place=args.in_place,
        dry_run=args.dry_run,
        removed_path=args.removed,
        report_path=args.report,
        progress=not args.no_progress,
    )

    if report.get('enabled') is False:
        sys.exit(0)

    _print_summary(report)


if __name__ == '__main__':
    traceback_install(width=88)
    run_cleanup_database()
