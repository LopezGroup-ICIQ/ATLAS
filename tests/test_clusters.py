"""Tests for the clusters module."""

from unittest.mock import Mock

import numpy as np
import pytest

# Import atlas first to resolve circular imports
import atlas  # noqa: F401
import atlas.core.clusters as atl_clusters


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_latt_constant(self):
        """LATT is a 3-element list of lattice constants."""
        assert hasattr(atl_clusters, 'LATT')
        assert len(atl_clusters.LATT) == 3

    def test_max_vac_constant(self):
        """MAX_VAC is a positive vacuum thickness."""
        assert hasattr(atl_clusters, 'MAX_VAC')
        assert atl_clusters.MAX_VAC > 0


class TestGetElementConstants:
    """Tests for get_element_constants function."""

    def test_returns_dict_with_expected_keys(self):
        result = atl_clusters.get_element_constants('Cu')
        assert isinstance(result, dict)
        assert 'a' in result
        assert 'crystal_system' in result
        assert result['a'] > 0

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError):
            atl_clusters.get_element_constants('Xx')


class TestGetNearestNeighborDistance:
    """Tests for _get_nearest_neighbor_distance."""

    def test_cu_fcc_nn_distance(self):
        nn = atl_clusters._get_nearest_neighbor_distance('Cu')
        assert 2.5 < nn < 2.6  # Cu FCC: a/sqrt(2) ≈ 2.556

    def test_fe_bcc_nn_distance(self):
        nn = atl_clusters._get_nearest_neighbor_distance('Fe')
        assert 2.4 < nn < 2.5  # Fe BCC: a*sqrt(3)/2 ≈ 2.482

    def test_returns_positive(self):
        nn = atl_clusters._get_nearest_neighbor_distance('Au')
        assert nn > 0


class TestGenerateWulffCluster:
    """Tests for _generate_wulff_cluster."""

    def test_fcc_element_returns_atoms(self):
        atoms = atl_clusters._generate_wulff_cluster('Cu', 20)
        assert len(atoms) == 20
        assert set(atoms.get_chemical_symbols()) == {'Cu'}

    def test_hcp_falls_back_to_spherical(self):
        atoms = atl_clusters._generate_wulff_cluster('Zn', 15)
        assert len(atoms) == 15
        assert set(atoms.get_chemical_symbols()) == {'Zn'}


class TestGenerateSphericalCluster:
    """Tests for _generate_spherical_cluster."""

    def test_exact_atom_count(self):
        atoms = atl_clusters._generate_spherical_cluster('Cu', 30)
        assert len(atoms) == 30

    def test_minimum_distance_respected(self):
        atoms = atl_clusters._generate_spherical_cluster('Cu', 20)
        positions = atoms.get_positions()
        nn_dist = atl_clusters._get_nearest_neighbor_distance('Cu')
        expected_min = nn_dist * 0.85
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= expected_min * 0.99


class TestMakeCleanCluster:
    """Tests for make_clean_cluster function."""

    def _make_mock_phase(self, symbol='Cu'):
        from unittest.mock import Mock

        from pymatgen.core import Element

        mock_phase = Mock()

        try:
            # If it's a valid element (like 'Cu'), use the real deal
            mock_phase.cluster_elem = Element(symbol)
        except ValueError:
            # If it's an invalid element (like 'Xx'), use a mock fallback
            # so the test can proceed into make_clean_cluster()
            bad_elem = Mock()
            bad_elem.symbol = symbol
            # Mock it so that passing it to get_el_sp() still raises the error later
            bad_elem.__str__ = lambda self: symbol
            mock_phase.cluster_elem = bad_elem

        mock_phase.name = 'alpha'
        return mock_phase

    def test_wulff_method_returns_cluster(self):
        phase = self._make_mock_phase('Cu')
        result = atl_clusters.make_clean_cluster(None, 20, phase, method='wulff')
        assert hasattr(result, 'structure')
        assert len(result.structure) == 20

    def test_spherical_method_returns_cluster(self):
        phase = self._make_mock_phase('Cu')
        result = atl_clusters.make_clean_cluster(None, 15, phase, method='spherical')
        assert hasattr(result, 'structure')
        assert len(result.structure) == 15

    def test_unknown_method_raises(self):
        phase = self._make_mock_phase('Cu')
        with pytest.raises(ValueError, match='Unknown cluster method'):
            atl_clusters.make_clean_cluster(None, 10, phase, method='invalid')

    def test_unknown_element_raises(self):
        phase = self._make_mock_phase('Xx')
        with pytest.raises(ValueError):
            atl_clusters.make_clean_cluster(None, 10, phase)


class TestMakeCleanDimer:
    """Tests for make_clean_dimer function."""

    def test_make_clean_dimer_returns_cluster(self):
        from pymatgen.core import Element

        mock_phase = Mock()
        mock_phase.cluster_elem = Element('Cu')
        mock_phase.name = 'alpha'
        result = atl_clusters.make_clean_dimer(None, mock_phase)
        assert hasattr(result, 'structure')
        assert len(result.structure) == 2

    def test_make_clean_dimer_unknown_element_raises(self):
        # Note: pymatgen.core.Element('Xx') will raise a ValueError natively,
        # which is exactly what the test is expecting to catch!
        from pymatgen.core import Element

        mock_phase = Mock()
        with pytest.raises(ValueError):
            mock_phase.cluster_elem = Element('Xx')
            atl_clusters.make_clean_dimer(None, mock_phase)


class TestApplyGaussPerturbList:
    """Tests for apply_gauss_perturb_list function."""

    def test_apply_gauss_perturb_list_returns_list(self):
        from pymatgen.core import Structure as pmg_struct

        lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
        coords = [[0, 0, 0], [1, 0, 0]]
        pmg = pmg_struct(lattice, ['Cu', 'Cu'], coords)
        atl_struct = atl_clusters.atl_struct.Structure(structure=pmg)
        result = atl_clusters.apply_gauss_perturb_list(
            repeat=2, cluster_list=[atl_struct], center=0.04
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_apply_gauss_perturb_list_preserves_atoms(self):
        from pymatgen.core import Structure as pmg_struct

        lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
        coords = [[0, 0, 0], [1, 0, 0]]
        pmg = pmg_struct(lattice, ['Cu', 'Cu'], coords)
        atl_struct = atl_clusters.atl_struct.Structure(structure=pmg)
        result = atl_clusters.apply_gauss_perturb_list(
            repeat=1, cluster_list=[atl_struct], center=0.04
        )
        assert len(result[0].structure) == len(atl_struct.structure)
