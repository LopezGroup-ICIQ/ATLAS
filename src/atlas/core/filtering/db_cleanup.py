"""Database cleanup: metadata standardization, duplicate and filter removal.

The cleanup pipeline is intentionally independent from AiiDA and the active
learning workflow. It operates on ``list[ase.Atoms]`` or on an
``InitialDatabase`` and is shared by the standalone ``atl_cleanup_db`` CLI and
the database-generation pipeline.
"""

from __future__ import annotations

import json
import lzma
import pathlib as pl
import pickle
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, fields
from uuid import uuid4

from ase import Atoms
from ase import io as aseio

import atlas.core.code_utils as atl_cut
from atlas.core.filtering.structure_filters import (
    DuplicateOutcome,
    apply_structure_filters,
    coerce_to_atoms_list,
    find_duplicate_structures,
)
from atlas.core.metadata import (
    DFT_CONVERGED_KEY,
    DFT_IONIC_CONVERGED_KEY,
    DFT_IS_RELAXATION_KEY,
    DFT_SCF_CONVERGED_KEY,
    check_atl_ids,
    standardize_atoms_info,
)


@dataclass
class CleanupSettings:
    """User-facing cleanup options."""

    enabled: bool = True
    standardize_metadata: bool = True
    reassign_duplicate_ids: bool = True
    remove_duplicates: bool = True
    duplicate_r_cut: float = 4.0
    duplicate_n_max: int = 4
    duplicate_l_max: int = 4
    duplicate_similarity_threshold: float = 0.999
    group_duplicates_by_phase: bool = True
    apply_struct_filters: bool = True
    skip_base_structures: bool = False
    skip_isolated_atoms: bool = True
    write_removed: bool = True
    removed_file_suffix: str = '_removed'
    output_suffix: str = '_cleaned'

    @classmethod
    def from_dict(cls, config: dict | None) -> CleanupSettings:
        """Build settings from a ``[cleanup]`` TOML section."""
        if not config:
            return cls()

        valid_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(config) - valid_fields)
        if unknown:
            atl_cut.custom_print(
                f'Unknown cleanup settings ignored: {unknown}.',
                'warn',
            )

        kwargs = {key: value for key, value in config.items() if key in valid_fields}
        return cls(**kwargs)


@dataclass
class CleanupResult:
    """Structures and report produced by :func:`clean_database`."""

    kept: list[Atoms]
    removed: list[Atoms]
    report: dict
    database: object | None = None


def _copy_atoms(atoms: Atoms) -> Atoms:
    """Return an independent copy that keeps the metadata dictionary."""
    new_atoms = atoms.copy()
    new_atoms.info = dict(atoms.info)
    return new_atoms


def _ensure_unique_ids(
    atoms_list: list[Atoms],
    reassign_duplicates: bool,
) -> tuple[list[dict], list[dict]]:
    """Ensure every structure has an ``atl_id`` and optionally make IDs unique."""
    seen_ids: set[str] = set()
    reassigned = []
    missing = []

    for idx, atoms in enumerate(atoms_list):
        atl_id = atoms.info.get('atl_id')

        if atl_id is None or str(atl_id) == '':
            atl_id = str(uuid4())
            atoms.info['atl_id'] = atl_id
            missing.append({'index': idx, 'atl_id': atl_id})
        else:
            atl_id = str(atl_id)
            atoms.info['atl_id'] = atl_id

        if atl_id in seen_ids and reassign_duplicates:
            new_id = str(uuid4())
            reassigned.append(
                {
                    'index': idx,
                    'old_id': atl_id,
                    'new_id': new_id,
                }
            )
            atoms.info['atl_id'] = new_id
            atl_id = new_id

        seen_ids.add(atl_id)

    return reassigned, missing


def _remaining_duplicate_ids(atoms_list: list[Atoms]) -> dict[str, int]:
    """Return repeated ``atl_id`` values that survived the cleanup."""
    _, duplicates = check_atl_ids(atoms_list)
    return duplicates


def _set_removal_reason(
    atoms: Atoms,
    reasons: list[str],
) -> None:
    """Annotate a removed structure with the reason(s) for its removal."""
    existing = atoms.info.get('atl_cleanup_reason')
    if existing:
        reasons = [existing, *reasons]
    atoms.info['atl_cleanup_reason'] = ', '.join(dict.fromkeys(reasons))


def _summarize_dft_convergence(atoms_list: list[Atoms]) -> dict:
    """Summarize DFT convergence flags, distinguishing unknown from false."""
    summary = {}
    for label, key in (
        ('scf', DFT_SCF_CONVERGED_KEY),
        ('ionic', DFT_IONIC_CONVERGED_KEY),
        ('overall', DFT_CONVERGED_KEY),
    ):
        counts = Counter()
        for atoms in atoms_list:
            value = atoms.info.get(key)
            if value is True:
                counts['true'] += 1
            elif value is False:
                counts['false'] += 1
            else:
                counts['unknown'] += 1
        summary[label] = dict(counts)

    summary['relaxations'] = sum(
        1 for atoms in atoms_list if atoms.info.get(DFT_IS_RELAXATION_KEY) is True
    )
    return summary


def _build_report(
    settings: CleanupSettings,
    atoms_list: list[Atoms],
    kept: list[Atoms],
    filter_removed: list[Atoms],
    duplicate_removed: list[Atoms],
    filter_outcome,
    duplicate_outcome,
    reassigned_ids: list[dict],
    missing_ids: list[dict],
) -> dict:
    """Assemble the serializable cleanup report."""
    return {
        'n_input': len(atoms_list),
        'n_kept': len(kept),
        'n_removed': len(filter_removed) + len(duplicate_removed),
        'removed_by_filter': len(filter_removed),
        'removed_by_duplicate': len(duplicate_removed),
        'removed_ids': [
            str(atoms.info.get('atl_id'))
            for atoms in [*filter_removed, *duplicate_removed]
        ],
        'by_filter': filter_outcome.by_filter,
        'first_filter': filter_outcome.first_filter,
        'by_phase': filter_outcome.by_phase,
        'by_type': filter_outcome.by_type,
        'by_modification': filter_outcome.by_modification,
        'filter_failures': filter_outcome.failures,
        'invalid_filters': filter_outcome.invalid_filters,
        'duplicate_pairs': duplicate_outcome.pairs,
        'reassigned_ids': reassigned_ids,
        'missing_ids_assigned': [entry['atl_id'] for entry in missing_ids],
        'remaining_duplicate_ids': _remaining_duplicate_ids(kept),
        'dft_convergence': _summarize_dft_convergence(kept),
        'settings': asdict(settings),
    }


def clean_database(
    structures,
    cleanup_config: dict | CleanupSettings | None = None,
    *,
    filter_settings: dict | None = None,
    progress: bool = True,
) -> CleanupResult:
    """Clean an ``InitialDatabase`` or a list of ASE Atoms.

    The pipeline runs in this order:

    1. Standardize legacy metadata and ensure every structure has an
       ``atl_id``.
    2. Apply the ``[struct_filters]`` unphysical-structure filters.
    3. Remove duplicate structures with SOAP descriptors.

    Filters run before duplicate detection so that unphysical structures do
    not keep a valid duplicate alive (or vice versa).

    Parameters
    ----------
    structures : InitialDatabase | list[ase.Atoms]
        Structures to clean.
    cleanup_config : dict | CleanupSettings, optional
        Cleanup options. A dictionary is interpreted as the ``[cleanup]``
        TOML section.
    filter_settings : dict, optional
        The ``[struct_filters]`` section. When omitted, the filters stored in
        ``structures`` (if any) cannot be recovered and no filters run.
    progress : bool, optional
        Show progress bars.

    Returns
    -------
    CleanupResult
        Kept and removed structures, a serializable report, and the updated
        database object when one was supplied.
    """
    if isinstance(cleanup_config, CleanupSettings):
        settings = cleanup_config
    else:
        settings = CleanupSettings.from_dict(cleanup_config)

    database_obj = None
    source_indices = []
    if hasattr(structures, 'df') and hasattr(structures, 'db_struct_to_ase'):
        database_obj = structures
        source_indices = list(structures.df.index)
        atoms_list = [
            structures.db_struct_to_ase(row=row)
            for _, row in structures.df.iterrows()
        ]
    else:
        atoms_list = coerce_to_atoms_list(structures)

    atoms_list = [_copy_atoms(atoms) for atoms in atoms_list]
    for atoms in atoms_list:
        atoms.info.pop('atl_cleanup_reason', None)

    source_by_atom = {
        id(atoms): source_idx
        for atoms, source_idx in zip(atoms_list, source_indices, strict=False)
    }

    if not settings.enabled:
        return CleanupResult(
            kept=atoms_list,
            removed=[],
            report={
                'enabled': False,
                'n_input': len(atoms_list),
                'n_kept': len(atoms_list),
                'n_removed': 0,
            },
            database=database_obj,
        )

    if settings.standardize_metadata:
        standardize_atoms_info(atoms_list)

    reassigned_ids, missing_ids = _ensure_unique_ids(
        atoms_list,
        reassign_duplicates=settings.reassign_duplicate_ids,
    )

    if database_obj is not None:
        for entry in reassigned_ids:
            source_idx = source_indices[entry['index']]
            database_obj.df.loc[source_idx, 'unique_id'] = entry['new_id']
        for entry in missing_ids:
            source_idx = source_indices[entry['index']]
            database_obj.df.loc[source_idx, 'unique_id'] = entry['atl_id']

    effective_filter_settings = (
        filter_settings if settings.apply_struct_filters else None
    )
    filter_outcome = apply_structure_filters(
        atoms_list,
        effective_filter_settings or {},
        skip_base=settings.skip_base_structures,
        skip_isolated_atoms=settings.skip_isolated_atoms,
        progress=progress,
    )

    for atoms in filter_outcome.removed:
        reasons = filter_outcome.reasons.get(str(atoms.info.get('atl_id')), [])
        _set_removal_reason(atoms, reasons or ['structure_filter'])

    if settings.remove_duplicates:
        duplicate_outcome = find_duplicate_structures(
            filter_outcome.kept,
            r_cut=settings.duplicate_r_cut,
            n_max=settings.duplicate_n_max,
            l_max=settings.duplicate_l_max,
            similarity_threshold=settings.duplicate_similarity_threshold,
            group_by_phase=settings.group_duplicates_by_phase,
            progress=progress,
        )
    else:
        duplicate_outcome = DuplicateOutcome()

    duplicate_indices = set(duplicate_outcome.duplicate_indices)
    kept = [
        atoms
        for idx, atoms in enumerate(filter_outcome.kept)
        if idx not in duplicate_indices
    ]
    duplicate_removed = [
        atoms
        for idx, atoms in enumerate(filter_outcome.kept)
        if idx in duplicate_indices
    ]
    for atoms in duplicate_removed:
        _set_removal_reason(atoms, ['duplicate'])

    removed = [*filter_outcome.removed, *duplicate_removed]
    report = _build_report(
        settings=settings,
        atoms_list=atoms_list,
        kept=kept,
        filter_removed=filter_outcome.removed,
        duplicate_removed=duplicate_removed,
        filter_outcome=filter_outcome,
        duplicate_outcome=duplicate_outcome,
        reassigned_ids=reassigned_ids,
        missing_ids=missing_ids,
    )

    if database_obj is not None:
        kept_source_indices = [source_by_atom[id(atoms)] for atoms in kept]
        database_obj.df = database_obj.df.loc[kept_source_indices].reset_index(
            drop=True
        )

        removal_history = getattr(database_obj, '_removal_history', None)
        if removal_history is not None:
            removal_history.setdefault('cleanup', {}).update(report)
            removal_history.setdefault('filtering', {}).update(
                {
                    'filtered_uuids': report['removed_ids'],
                    'by_phase': report['by_phase'],
                    'by_type': report['by_type'],
                    'by_modification': report['by_modification'],
                    'by_filter': report['first_filter'],
                }
            )

    return CleanupResult(
        kept=kept,
        removed=removed,
        report=report,
        database=database_obj,
    )


def _is_compressed_database(path: pl.Path) -> bool:
    """Return whether ``path`` is a pickled ATLAS database."""
    return '.xz' in path.suffixes or '.pkl' in path.suffixes


def _read_structures(path: pl.Path):
    """Load either an ``InitialDatabase`` or a list of ASE Atoms."""
    if _is_compressed_database(path):
        from atlas.core.initial_db import InitialDatabase

        loaded = InitialDatabase.load_database(path)
        if hasattr(loaded, 'df'):
            return loaded

        # Legacy .pkl files load as a bare DataFrame.
        database = InitialDatabase(create_db=False)
        database.df = loaded
        return database

    structures = aseio.read(str(path), index=':')
    if isinstance(structures, Atoms):
        structures = [structures]
    return structures


def _write_structures(path: pl.Path, structures) -> None:
    """Write structures using the format implied by ``path``."""
    if _is_compressed_database(path):
        if not hasattr(structures, 'df'):
            raise ValueError(
                "Writing a compressed ATLAS database requires an "
                "InitialDatabase input and output."
            )
        structures.database_name = path.stem
        structures.database_path = path.parent
        if '.xz' in path.suffixes:
            with lzma.open(path, 'wb') as file_handle:
                pickle.dump(structures, file_handle)
        else:
            # Legacy .pkl databases store the bare DataFrame.
            legacy_data = getattr(structures, 'df', structures)
            with open(path, 'wb') as file_handle:
                pickle.dump(legacy_data, file_handle)
        return

    suffix = path.suffix.lower()
    file_format = (
        'extxyz' if suffix in ('.xyz', '.extxyz') else suffix.lstrip('.')
    )

    if not structures:
        path.write_text('', encoding='utf-8')
        return

    aseio.write(str(path), structures, format=file_format)


def _default_report_path(path: pl.Path) -> pl.Path:
    return path.with_name(f'{path.stem}_cleanup_report.json')


def clean_database_file(
    input_path: str | pl.Path,
    output_path: str | pl.Path | None = None,
    *,
    cleanup_config: dict | CleanupSettings | None = None,
    filter_settings: dict | None = None,
    in_place: bool = False,
    dry_run: bool = False,
    write_removed: bool | None = None,
    removed_path: str | pl.Path | None = None,
    report_path: str | pl.Path | None = None,
    progress: bool = True,
) -> dict:
    """Clean a database file and write the cleaned structures and report.

    Parameters
    ----------
    input_path : str | Path
        Extended XYZ file or compressed ATLAS database (``.xz`` / ``.pkl``).
    output_path : str | Path, optional
        Destination for the cleaned database. Defaults to
        ``<input>_cleaned<suffix>``. Ignored when ``in_place`` is true.
    cleanup_config : dict | CleanupSettings, optional
        ``[cleanup]`` settings.
    filter_settings : dict, optional
        ``[struct_filters]`` settings.
    in_place : bool, optional
        Overwrite ``input_path`` after writing a ``.bak`` backup.
    dry_run : bool, optional
        Run the pipeline and write the report without writing structures.
    write_removed : bool, optional
        Write removed structures to a separate file. Defaults to the value in
        ``cleanup_config``.
    removed_path : str | Path, optional
        Destination for removed structures.
    report_path : str | Path, optional
        Destination for the JSON report.
    progress : bool, optional
        Show progress bars.

    Returns
    -------
    dict
        The cleanup report with the resolved output paths.
    """
    input_path = pl.Path(input_path).resolve(strict=True)

    if isinstance(cleanup_config, CleanupSettings):
        settings = cleanup_config
    else:
        settings = CleanupSettings.from_dict(cleanup_config)

    if settings.enabled is False:
        atl_cut.custom_print(
            'Cleanup is disabled in the configuration. Nothing to do.',
            'warn',
        )
        return {'enabled': False, 'input_path': str(input_path)}

    structures = _read_structures(input_path)
    result = clean_database(
        structures,
        settings,
        filter_settings=filter_settings,
        progress=progress,
    )

    if in_place:
        if output_path is not None:
            atl_cut.custom_print(
                'Ignoring output_path because in_place is enabled.',
                'warn',
            )
        output_path = input_path
    elif output_path is None:
        output_path = input_path.with_name(
            f'{input_path.stem}{settings.output_suffix}{input_path.suffix}'
        )
    else:
        output_path = pl.Path(output_path)

    if write_removed is None:
        write_removed = settings.write_removed

    if removed_path is None:
        removed_path = output_path.with_name(
            f'{output_path.stem}{settings.removed_file_suffix}.extxyz'
        )
    else:
        removed_path = pl.Path(removed_path)

    if report_path is None:
        report_path = _default_report_path(output_path)
    else:
        report_path = pl.Path(report_path)

    report = dict(result.report)
    report['input_path'] = str(input_path)
    report['output_path'] = str(output_path)
    report['removed_path'] = str(removed_path) if write_removed else None
    report['report_path'] = str(report_path)

    if dry_run:
        atl_cut.custom_print(
            'Dry run enabled: structures were not written.',
            'info',
        )
    else:
        if in_place:
            backup_path = input_path.with_suffix(input_path.suffix + '.bak')
            shutil.copy2(input_path, backup_path)
            atl_cut.custom_print(
                f'Backup of the original database written to {backup_path}.',
                'info',
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if _is_compressed_database(output_path):
            structures_to_write = (
                result.database if result.database is not None else result.kept
            )
        else:
            structures_to_write = result.kept
        _write_structures(output_path, structures_to_write)

        if write_removed:
            removed_path.parent.mkdir(parents=True, exist_ok=True)
            if result.removed:
                aseio.write(str(removed_path), result.removed, format='extxyz')
            else:
                atl_cut.custom_print(
                    'No structures were removed; skipping removed file.',
                    'info',
                )
                report['removed_path'] = None

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as file_handle:
        json.dump(report, file_handle, indent=2)

    return report
