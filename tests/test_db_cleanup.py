"""Tests for the database cleanup pipeline and structure filters."""

import json
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
from ase import Atoms
from ase import io as aseio
from ase.build import bulk, fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from atlas.core.filtering.db_cleanup import (
    CleanupSettings,
    clean_database,
    clean_database_file,
)
from atlas.core.filtering.structure_filters import (
    apply_filter_evaporation,
    apply_filter_fragmented,
    apply_filter_layer_distance,
    apply_filter_low_coordination,
    apply_filter_no_neighbors,
    apply_struct_filters_atl_db,
    apply_structure_filters,
    find_duplicate_structures,
)
from atlas.core.initial_db import InitialDatabase
from atlas.core.metadata import check_atl_ids, standardize_atoms_info
from atlas.core.structure import Bulk


def _bulk_atoms(symbol='Cu', crystal='fcc', lattice=3.6, repeat=(2, 2, 2)):
    atoms = bulk(symbol, crystal, a=lattice).repeat(repeat)
    atoms.info['phase'] = 'alpha'
    atoms.info['atl_struct_type'] = 'bulk'
    atoms.info['bulk'] = True
    return atoms


def _make_database(atoms_list) -> InitialDatabase:
    database = InitialDatabase(create_db=False)
    database.df = pd.DataFrame()

    for idx, atoms in enumerate(atoms_list):
        structure = Bulk(
            structure=AseAtomsAdaptor().get_structure(atoms),
            material_name=f'material_{idx}',
            material_id=idx,
            phase=atoms.info.get('phase', 'alpha'),
            base=True,
            unique_id=atoms.info.get('atl_id', f'id_{idx}'),
        )
        database.df = structure.save_to_db(database.df)

    return database


def _write_extxyz(path, atoms_list):
    aseio.write(str(path), atoms_list, format='extxyz')


class TestMetadataStandardization:
    """Tests for legacy metadata migration and ID checks."""

    def test_migrates_legacy_keys_and_assigns_ids(self):
        atoms = _bulk_atoms()
        atoms.info.update(
            {
                'mdb_id': 'legacy-id',
                'mdb_struct_type': 'bulk',
                'mdb_db_index': 3,
                'mdb_al_step': 1,
            }
        )

        missing = Atoms('Cu', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)

        standardize_atoms_info([atoms, missing])

        assert atoms.info['atl_id'] == 'legacy-id'
        assert atoms.info['atl_struct_type'] == 'bulk'
        assert atoms.info['atl_db_index'] == 3
        assert atoms.info['atl_al_step'] == 1
        assert not {'mdb_id', 'mdb_struct_type', 'mdb_db_index', 'mdb_al_step'} & set(
            atoms.info
        )
        assert 'atl_id' in missing.info

    def test_sanitizes_non_serializable_metadata(self):
        atoms = _bulk_atoms()
        atoms.info['atl_id'] = 'sanitize'
        atoms.info['callable'] = lambda: None
        atoms.info['numpy_scalar'] = np.float64(1.5)
        atoms.info['tuple_value'] = (1, 2, 3)
        atoms.info['stress'] = np.zeros((3, 3))

        standardize_atoms_info([atoms])

        assert 'callable' not in atoms.info
        assert atoms.info['numpy_scalar'] == 1.5
        assert atoms.info['tuple_value'] == [1, 2, 3]
        assert atoms.info['stress'] == [[0.0, 0.0, 0.0]] * 3

    def test_check_atl_ids_reports_missing_and_duplicates(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'same-id'
        second = _bulk_atoms()
        second.info['atl_id'] = 'same-id'
        third = _bulk_atoms()

        missing, duplicates = check_atl_ids([first, second, third])

        assert missing == [2]
        assert duplicates == {'same-id': 2}


class TestStructureFilters:
    """Tests for the unphysical-structure filter registry."""

    def test_no_neighbors_filter(self):
        disconnected = Atoms(
            'Cu2',
            positions=[[0, 0, 0], [5, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        disconnected.info.update(
            {
                'atl_id': 'disconnected',
                'phase': 'alpha',
                'bulk': True,
                'atl_struct_type': 'bulk',
            }
        )

        connected = _bulk_atoms()
        connected.info['atl_id'] = 'connected'

        outcome = apply_structure_filters(
            [disconnected, connected],
            {'no_neighbors': {'cov_rad_multiplier': 1.05}},
            progress=False,
        )

        assert outcome.kept == [connected]
        assert outcome.removed == [disconnected]
        assert outcome.first_filter == {'no_neighbors': 1}

    def test_layer_distance_filter(self):
        slab = fcc111('Cu', size=(2, 2, 2), vacuum=10.0)
        top_z = slab.positions[:, 2].max()
        extra_atom = Atoms(
            'Cu',
            positions=[[slab.cell[0, 0] / 2, slab.cell[1, 1] / 2, top_z + 10.0]],
        )
        slab += extra_atom
        slab.info.update(
            {
                'atl_id': 'slab',
                'phase': 'alpha',
                'surface': True,
                'atl_struct_type': 'surface',
            }
        )

        outcome = apply_structure_filters(
            [slab],
            {'layer_distance': {'max_layer_distance_ang': 3.0}},
            progress=False,
        )

        assert len(outcome.removed) == 1
        assert outcome.first_filter == {'layer_distance': 1}

    def test_isolated_atoms_are_not_filtered(self):
        isolated = Atoms('Cu', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        isolated.info.update(
            {
                'atl_id': 'isolated',
                'phase': 'alpha',
                'isolated_atom': True,
                'atl_struct_type': 'isolated_atom',
            }
        )

        outcome = apply_structure_filters(
            [isolated],
            {'no_neighbors': {'cov_rad_multiplier': 1.05}},
            progress=False,
        )

        assert outcome.kept == [isolated]
        assert outcome.removed == []

    def test_filters_accept_a_plain_atoms_list(self):
        disconnected = Atoms(
            'Cu2',
            positions=[[0, 0, 0], [5, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        disconnected.info.update(
            {
                'atl_id': 'disconnected',
                'phase': 'alpha',
                'bulk': True,
                'atl_struct_type': 'bulk',
            }
        )
        connected = _bulk_atoms()
        connected.info['atl_id'] = 'connected'

        result = apply_struct_filters_atl_db(
            [disconnected, connected],
            {'struct_filters': {'no_neighbors': {'cov_rad_multiplier': 1.05}}},
            write_filtered=False,
            progress=False,
        )

        assert result['filtered_uuids'] == ['disconnected']
        assert len(result['kept_structures']) == 1
        assert len(result['removed_structures']) == 1

    def test_fragmented_filter(self):
        connected = Atoms(
            'Cu2',
            positions=[[0, 0, 0], [2.5, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        fragmented = Atoms(
            'Cu2',
            positions=[[0, 0, 0], [15, 15, 15]],
            cell=[20, 20, 20],
            pbc=True,
        )

        assert not apply_filter_fragmented(connected)
        assert apply_filter_fragmented(fragmented)

    def test_low_coordination_filter(self):
        connected = _bulk_atoms()
        chain = Atoms(
            'Cu3',
            positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )

        assert not apply_filter_low_coordination(connected)
        assert apply_filter_low_coordination(chain)

    def test_no_neighbors_min_coord_extension(self):
        chain = Atoms(
            'Cu3',
            positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )

        assert not apply_filter_no_neighbors(chain, min_coord=0)
        assert apply_filter_no_neighbors(chain, min_coord=1)

    def test_layer_distance_uses_miller_index(self):
        atoms = Atoms(
            'Cu4',
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )

        assert apply_filter_layer_distance(
            atoms,
            max_layer_distance_ang=3.0,
            is_surface=True,
            miller_index=(1, 0, 0),
        )
        assert not apply_filter_layer_distance(
            atoms,
            max_layer_distance_ang=3.0,
            is_surface=True,
            miller_index=(0, 0, 1),
        )

    def test_evaporation_uses_miller_index(self):
        atoms = Atoms(
            'Cu4',
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )

        assert apply_filter_evaporation(
            atoms,
            max_allowed_thickness=5.0,
            miller_index=(1, 0, 0),
        )
        assert not apply_filter_evaporation(
            atoms,
            max_allowed_thickness=5.0,
            miller_index=(0, 0, 1),
        )

    def test_legacy_string_miller_is_parsed(self):
        atoms = Atoms(
            'Cu4',
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        atoms.info['surface_miller'] = '[1 0 0]'

        assert apply_filter_evaporation(
            atoms,
            max_allowed_thickness=5.0,
        )

    def test_duplicate_slabs_are_skipped_for_bulk(self):
        atoms = _bulk_atoms()
        atoms.info['atl_id'] = 'bulk'

        outcome = apply_structure_filters(
            [atoms],
            {'duplicate_slabs': {'tolerance': 0.1}},
            progress=False,
        )

        assert outcome.kept == [atoms]
        assert outcome.removed == []


class TestDuplicateDetection:
    """Tests for SOAP duplicate detection."""

    def test_identical_structures_are_duplicates(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'first'
        second = _bulk_atoms()
        second.info['atl_id'] = 'second'

        outcome = find_duplicate_structures([first, second], progress=False)

        assert outcome.duplicate_indices == [1]
        assert outcome.pairs[0]['kept_atl_id'] == 'first'
        assert outcome.pairs[0]['duplicate_atl_id'] == 'second'
        assert outcome.pairs[0]['similarity'] >= 0.999

    def test_different_structures_are_kept(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'fcc'
        second = _bulk_atoms(crystal='bcc', lattice=2.9)
        second.info['atl_id'] = 'bcc'

        outcome = find_duplicate_structures([first, second], progress=False)

        assert outcome.duplicate_indices == []

    def test_duplicates_can_be_grouped_by_phase(self):
        first = _bulk_atoms()
        first.info.update({'atl_id': 'alpha', 'phase': 'alpha'})
        second = _bulk_atoms()
        second.info.update({'atl_id': 'beta', 'phase': 'beta'})

        outcome = find_duplicate_structures(
            [first, second],
            group_by_phase=True,
            progress=False,
        )

        assert outcome.duplicate_indices == []


class TestCleanDatabase:
    """Tests for the full cleanup pipeline."""

    def test_full_pipeline(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'first'
        duplicate = _bulk_atoms()
        duplicate.info['atl_id'] = 'duplicate'

        disconnected = Atoms(
            'Cu2',
            positions=[[0, 0, 0], [5, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        disconnected.info.update(
            {
                'atl_id': 'disconnected',
                'phase': 'alpha',
                'bulk': True,
                'atl_struct_type': 'bulk',
            }
        )

        result = clean_database(
            [first, duplicate, disconnected],
            CleanupSettings(),
            filter_settings={'no_neighbors': {'cov_rad_multiplier': 1.05}},
            progress=False,
        )

        assert result.report['n_input'] == 3
        assert result.report['n_kept'] == 1
        assert result.report['n_removed'] == 2
        assert result.report['removed_by_filter'] == 1
        assert result.report['removed_by_duplicate'] == 1
        assert [atoms.info['atl_id'] for atoms in result.kept] == ['first']

    def test_repeated_ids_are_reassigned(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'same-id'
        second = _bulk_atoms(crystal='bcc', lattice=2.9)
        second.info['atl_id'] = 'same-id'

        result = clean_database(
            [first, second],
            CleanupSettings(remove_duplicates=False),
            progress=False,
        )

        assert len(result.report['reassigned_ids']) == 1
        kept_ids = {atoms.info['atl_id'] for atoms in result.kept}
        assert len(kept_ids) == 2

    def test_initial_database_is_updated_in_place(self):
        atoms_list = [_bulk_atoms(), _bulk_atoms(crystal='bcc', lattice=2.9)]
        for idx, atoms in enumerate(atoms_list):
            atoms.info['atl_id'] = f'id_{idx}'
        atoms_list.append(_bulk_atoms())
        atoms_list[-1].info['atl_id'] = 'id_0'

        database = _make_database(atoms_list)

        result = clean_database(
            database,
            CleanupSettings(),
            progress=False,
        )

        assert len(database.df) == 2
        assert result.database is database
        assert 'cleanup' in database._removal_history


class TestCleanDatabaseFile:
    """Tests for file-based cleanup and the CLI."""

    def _write_database(self, path):
        atoms_list = [_bulk_atoms(), _bulk_atoms()]
        atoms_list[0].info['atl_id'] = 'first'
        atoms_list[1].info['atl_id'] = 'second'
        _write_extxyz(path, atoms_list)

    def test_writes_cleaned_removed_and_report_files(self, tmp_path):
        database_path = tmp_path / 'input.xyz'
        self._write_database(database_path)

        report = clean_database_file(
            database_path,
            cleanup_config={
                'remove_duplicates': True,
                'write_removed': True,
                'output_suffix': '_clean',
            },
            progress=False,
        )

        cleaned_path = tmp_path / 'input_clean.xyz'
        removed_path = tmp_path / 'input_clean_removed.extxyz'
        report_path = tmp_path / 'input_clean_cleanup_report.json'

        assert cleaned_path.exists()
        assert removed_path.exists()
        assert report_path.exists()
        assert report['n_kept'] == 1
        assert report['n_removed'] == 1
        assert len(aseio.read(str(cleaned_path), index=':')) == 1

        saved_report = json.loads(report_path.read_text(encoding='utf-8'))
        assert saved_report['n_removed'] == 1

    def test_in_place_writes_backup(self, tmp_path):
        database_path = tmp_path / 'input.xyz'
        self._write_database(database_path)

        clean_database_file(
            database_path,
            cleanup_config={'remove_duplicates': True},
            in_place=True,
            progress=False,
        )

        backup_path = tmp_path / 'input.xyz.bak'
        assert backup_path.exists()
        assert len(aseio.read(str(database_path), index=':')) == 1

    def test_dry_run_writes_only_the_report(self, tmp_path):
        database_path = tmp_path / 'input.xyz'
        self._write_database(database_path)

        report = clean_database_file(
            database_path,
            cleanup_config={'remove_duplicates': True},
            dry_run=True,
            progress=False,
        )

        assert report['n_kept'] == 1
        assert not (tmp_path / 'input_cleaned.xyz').exists()
        assert (tmp_path / 'input_cleaned_cleanup_report.json').exists()

    def test_cli_entry_point(self, tmp_path, monkeypatch):
        from atlas.core.command_line.cli_cleanup_database import (
            run_cleanup_database,
        )

        database_path = tmp_path / 'input.xyz'
        self._write_database(database_path)
        config_path = tmp_path / 'settings.toml'
        config_path.write_text(
            '[cleanup]\nremove_duplicates = true\n',
            encoding='utf-8',
        )

        monkeypatch.setattr(
            sys,
            'argv',
            [
                'atl_cleanup_db',
                '-i',
                str(database_path),
                '-c',
                str(config_path),
                '--no-progress',
            ],
        )
        run_cleanup_database()

        assert (tmp_path / 'input_cleaned.xyz').exists()
        assert (tmp_path / 'input_cleaned_cleanup_report.json').exists()


class TestVacancyGeneration:
    """Tests for rejecting fragmented vacancy candidates."""

    def test_fragmented_candidates_are_not_saved(self, monkeypatch):
        import atlas.core.initial_db as initial_db_module

        atoms = _bulk_atoms()
        atoms.info['atl_id'] = 'base'
        database = _make_database([atoms])
        database.phase_diagram = Mock()
        database.phase_diagram.get_phase.return_value = Mock(name='alpha')

        monkeypatch.setattr(
            initial_db_module,
            'apply_filter_fragmented',
            lambda *args, **kwargs: True,
        )

        initial_size = len(database.df)
        database.apply_vacancies_random(
            filters=['bulk'],
            seed=0,
            repeat=1,
            element_list=['Cu'],
            max_vac_perc=0.2,
            min_vac_perc=0.2,
            lim_num_struc=1,
            phase=None,
            reject_fragmented=True,
            max_fragmented_retries=1,
        )

        assert len(database.df) == initial_size


class TestInitialDatabaseDuplicateDetection:
    """Tests for InitialDatabase.find_repeat_structures."""

    def test_find_repeat_structures_uses_atl_ids(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'first'
        second = _bulk_atoms()
        second.info['atl_id'] = 'second'
        third = _bulk_atoms(crystal='bcc', lattice=2.9)
        third.info['atl_id'] = 'third'

        database = _make_database([first, second, third])

        database.find_repeat_structures(delete=True)

        assert len(database.df) == 2
        assert set(database.df['unique_id']) == {'first', 'third'}

    def test_find_repeat_structures_without_delete_keeps_database(self):
        first = _bulk_atoms()
        first.info['atl_id'] = 'first'
        second = _bulk_atoms()
        second.info['atl_id'] = 'second'

        database = _make_database([first, second])

        database.find_repeat_structures(delete=False)

        assert len(database.df) == 2
