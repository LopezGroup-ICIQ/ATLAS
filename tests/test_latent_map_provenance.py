"""Tests for the latent-map bookkeeping behind ``descriptors_concave_hull``.

The latent map is what latent-space inversion is seeded and targeted from, so
the pairing between ``unique_ids`` and latent rows, and the provenance recorded
alongside them, are correctness requirements rather than conveniences.
"""

import numpy as np
import pytest

from atlas.core.initial_db import (
    InitialDatabase,
    _as_descriptor_array,
    _collect_latent_views,
)


def _entry(rows):
    """Wrap descriptor rows the way the descriptor providers store them."""
    return {'descriptors': rows}


class TestAsDescriptorArray:
    """Normalising the per-structure descriptor payload to a 2D array."""

    def test_single_averaged_row_becomes_two_dimensional(self):
        # SOAP with average='inner' yields one flat vector per structure.
        out = _as_descriptor_array(_entry([np.arange(5.0)]))
        assert out.shape == (1, 5)

    def test_per_atom_rows_are_kept_separate(self):
        out = _as_descriptor_array(_entry([np.zeros((4, 8)), np.ones((3, 8))]))
        assert out.shape == (7, 8)
        assert out[:4].sum() == 0.0
        assert out[4:].sum() == 24.0

    def test_bare_array_is_accepted(self):
        out = _as_descriptor_array(_entry(np.zeros((3, 6))))
        assert out.shape == (3, 6)

    def test_bare_vector_is_promoted(self):
        out = _as_descriptor_array(_entry(np.arange(6.0)))
        assert out.shape == (1, 6)


class TestCollectLatentViews:
    """Splitting encoded coordinates into per-atom and per-structure views."""

    def test_per_atom_descriptors_give_aligned_structure_means(self):
        descriptors = {
            'a': {'latent_space': np.array([[0.0, 0.0], [2.0, 4.0]])},
            'b': {'latent_space': np.array([[1.0, 1.0], [1.0, 3.0], [1.0, 5.0]])},
        }
        unique_ids = np.array(['a', 'b'], dtype=str)

        atom_latent, offsets, structure_latent = _collect_latent_views(
            descriptors, unique_ids
        )

        assert atom_latent.shape == (5, 2)
        assert offsets.tolist() == [0, 2, 5]
        assert structure_latent.shape == (2, 2)
        np.testing.assert_allclose(structure_latent[0], [1.0, 2.0])
        np.testing.assert_allclose(structure_latent[1], [1.0, 3.0])

    def test_structure_rows_are_one_per_unique_id(self):
        # The bug this guards: stacking per-atom rows and pairing them with
        # unique_ids only lines up when every structure contributes one row.
        descriptors = {str(i): {'latent_space': np.zeros((i + 1, 2))} for i in range(4)}
        unique_ids = np.array([str(i) for i in range(4)], dtype=str)

        atom_latent, offsets, structure_latent = _collect_latent_views(
            descriptors, unique_ids
        )

        assert atom_latent.shape == (1 + 2 + 3 + 4, 2)
        assert len(structure_latent) == len(unique_ids)
        assert offsets[-1] == atom_latent.shape[0]

    def test_averaged_descriptors_round_trip_unchanged(self):
        # SOAP: one row per structure, so the means must be the rows themselves.
        rows = np.array([[0.5, -1.0], [2.0, 3.0]])
        descriptors = {
            'a': {'latent_space': rows[0]},
            'b': {'latent_space': rows[1]},
        }
        unique_ids = np.array(['a', 'b'], dtype=str)

        atom_latent, offsets, structure_latent = _collect_latent_views(
            descriptors, unique_ids
        )

        np.testing.assert_allclose(structure_latent, rows)
        np.testing.assert_allclose(atom_latent, rows)
        assert offsets.tolist() == [0, 1, 2]

    def test_numpy_string_ids_index_the_dictionary(self):
        # unique_ids is a numpy unicode array; np.str_ must still hash to the
        # Python str keys the descriptor providers wrote.
        descriptors = {'abc': {'latent_space': np.zeros((2, 2))}}
        unique_ids = np.array(['abc'], dtype=str)

        _, _, structure_latent = _collect_latent_views(descriptors, unique_ids)

        assert structure_latent.shape == (1, 2)


class TestAseStructuresWithIds:
    """Tagging converted structures with the row identifier."""

    def test_every_structure_carries_its_unique_id(self, mock_pmg_structure):
        import pandas as pd

        db = InitialDatabase.__new__(InitialDatabase)
        db.df = pd.DataFrame(
            {
                'unique_id': ['id-0', 'id-1'],
                'structure': [mock_pmg_structure, mock_pmg_structure],
            }
        )

        structures = db.ase_structures_with_ids()

        assert [a.info['atl_id'] for a in structures] == ['id-0', 'id-1']
        assert all(len(a) == len(mock_pmg_structure) for a in structures)

    def test_ids_are_plain_strings(self, mock_pmg_structure):
        # generate_descriptors keys its output dict on this value; a UUID object
        # would produce keys that no later str-based lookup can find.
        import uuid

        import pandas as pd

        db = InitialDatabase.__new__(InitialDatabase)
        db.df = pd.DataFrame(
            {'unique_id': [uuid.uuid4()], 'structure': [mock_pmg_structure]}
        )

        (atoms,) = db.ase_structures_with_ids()

        assert type(atoms.info['atl_id']) is str


class TestConcaveHullProvenance:
    """The npz written next to the boundary plot describes how it was built."""

    @pytest.fixture
    def latent_npz(self, tmp_path, monkeypatch, mock_pmg_structure):
        pytest.importorskip('torch')
        import matplotlib

        matplotlib.use('Agg')

        import pandas as pd

        from atlas.active_learning.extrapolation import autoencoder as atl_ae
        from atlas.active_learning.extrapolation import (
            train_autoencoder as atl_tr_ae,
        )

        rng = np.random.default_rng(0)
        n_struct, n_atoms, dim = 40, 3, 7
        ids = [f'id-{i:03d}' for i in range(n_struct)]
        descriptors = {i: _entry([rng.normal(size=(n_atoms, dim))]) for i in ids}

        def fake_soap(**kwargs):
            return descriptors

        def fake_training(settings):
            # The dataset must already be the stacked per-atom rows.
            assert settings.dataset.shape == (n_struct * n_atoms, dim)
            return object()

        def fake_encode(model, descriptor_dict, device, **kwargs):
            # Standardization has to be requested, or the map is built by
            # feeding raw descriptors to an encoder trained on z-scored ones.
            assert kwargs['standardize_data'] is True
            assert kwargs['autoencoder_path'] is not None
            for key, entry in descriptor_dict.items():
                entry['latent_space'] = rng.normal(
                    size=(n_atoms, 2), scale=1.0
                ) + rng.normal(size=2)
                assert key in descriptors
            return descriptor_dict

        db = InitialDatabase.__new__(InitialDatabase)
        db.df = pd.DataFrame(
            {'unique_id': ids, 'structure': [mock_pmg_structure] * n_struct}
        )
        monkeypatch.setattr(db, 'get_soap_descriptors', fake_soap, raising=False)
        monkeypatch.setattr(atl_tr_ae, 'run_training', fake_training)
        monkeypatch.setattr(atl_ae, 'get_latent_space_autoencoder', fake_encode)

        plot_filename = str(tmp_path / 'descriptors_concave_hull.png')
        db.descriptors_concave_hull(
            descriptor_type='soap',
            device='cpu',
            plot_filename=plot_filename,
        )

        npz_path = tmp_path / 'latent_space.npz'
        assert npz_path.exists()
        with np.load(npz_path, allow_pickle=False) as data:
            yield dict(data)

    def test_records_the_descriptor_it_was_built_with(self, latent_npz):
        assert latent_npz['descriptor_type'][0] == 'soap'
        assert latent_npz['latent_mode'][0] == 'structure_mean'

    def test_records_where_the_autoencoder_lives(self, latent_npz):
        # The GUI gates inversion on finding this model; a CWD-relative path
        # would make the answer depend on where the process was started.
        path = latent_npz['autoencoder_path'][0]
        assert path.endswith('autoencoder_model.pth')
        assert '/' in path

    def test_structure_latent_is_paired_with_unique_ids(self, latent_npz):
        assert latent_npz['latent_space'].shape[0] == len(latent_npz['unique_ids'])
        assert latent_npz['latent_space'].shape[1] == 2

    def test_per_atom_view_is_delimited_by_offsets(self, latent_npz):
        offsets = latent_npz['atom_offsets']
        atom_latent = latent_npz['atom_latent_space']
        assert len(offsets) == len(latent_npz['unique_ids']) + 1
        assert offsets[0] == 0
        assert offsets[-1] == atom_latent.shape[0]

    def test_structure_latent_is_the_mean_of_its_atoms(self, latent_npz):
        offsets = latent_npz['atom_offsets']
        atom_latent = latent_npz['atom_latent_space']
        for i, mean in enumerate(latent_npz['latent_space']):
            rows = atom_latent[offsets[i] : offsets[i + 1]]
            np.testing.assert_allclose(rows.mean(axis=0), mean)

    def test_loads_without_pickle(self, latent_npz):
        # Every value must be a plain array, so readers can stay on
        # allow_pickle=False; the fixture already loaded it that way.
        assert 'hull_vertices' in latent_npz
        assert 'boundary_method' in latent_npz
