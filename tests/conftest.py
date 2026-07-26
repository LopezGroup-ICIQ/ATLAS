"""Shared fixtures for ATLAS tests."""

import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_NO_DBUS'] = '1'
os.environ['XDG_SESSION_TYPE'] = 'tty'

# AiiDA's pytest fixtures (aiida_profile, aiida_localhost, ...) provide a
# throwaway profile and computer, so integration tests never depend on a
# developer's own AiiDA configuration. Loaded lazily: tests that use them are
# skipped when aiida-core is unavailable.
try:
    import aiida.tools.pytest_fixtures  # noqa: F401

    pytest_plugins = ['aiida.tools.pytest_fixtures']
except ImportError:
    pass

import pathlib
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

_STABLE_CWD = pathlib.Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _stable_cwd():
    """Keep every test starting from a valid working directory.

    Some tests ``chdir`` into a temporary directory that pytest later deletes;
    a subsequent test that calls ``os.getcwd()`` (e.g. via ``monkeypatch.chdir``)
    would then raise ``FileNotFoundError``. Restoring a stable cwd if the current
    one has vanished avoids that cross-test contamination.
    """

    def _ensure_valid():
        try:
            os.getcwd()
        except OSError:
            os.chdir(_STABLE_CWD)

    _ensure_valid()
    yield
    _ensure_valid()


@pytest.fixture
def mock_atoms_simple():
    """Create a minimal ASE Atoms object with atl_id metadata."""
    atoms = Atoms(
        symbols='Cu4',
        positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [0, 0, 1.8]],
        cell=[4, 4, 4],
        pbc=True,
    )
    atoms.info['atl_id'] = 'test-uuid-0001'
    atoms.info['struct_name'] = 'test_structure'
    atoms.info['bulk'] = True
    atoms.info['phase'] = 'alpha'
    atoms.info['base'] = True
    return atoms


@pytest.fixture
def mock_atoms_with_dft():
    """Create an ASE Atoms object with DFT reference data."""
    atoms = Atoms(
        symbols='Cu4',
        positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [0, 0, 1.8]],
        cell=[4, 4, 4],
        pbc=True,
    )
    atoms.info['atl_id'] = 'test-uuid-0002'
    atoms.info['REF_energy'] = -10.5
    atoms.arrays['REF_forces'] = np.array(
        [
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
        ]
    )
    atoms.info['REF_stress'] = np.zeros((3, 3))
    return atoms


@pytest.fixture
def mock_pmg_structure():
    """Create a simple pymatgen Structure for testing."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(3.6)
    return Structure(
        lattice,
        ['Cu', 'Cu', 'Cu', 'Cu'],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


@pytest.fixture
def mock_phase_diagram():
    """Create a simple binary phase diagram for Cu-Zn."""
    from atlas.core.phase_diagram import Phase, PhaseDiagram

    phase_alpha = Phase(
        name='alpha',
        element_list=['Cu', 'Zn'],
        composition={'Cu': {'min': 0.6, 'max': 1.0}, 'Zn': {'min': 0.0, 'max': 0.4}},
        prototype='mp-30',
        offset=0.02,
        base_elem='Cu',
    )
    phase_gamma = Phase(
        name='gamma',
        element_list=['Cu', 'Zn'],
        composition={'Cu': {'min': 0.45, 'max': 0.6}, 'Zn': {'min': 0.4, 'max': 0.55}},
        prototype='mp-45',
        offset=0.02,
        base_elem='Cu',
    )
    return PhaseDiagram('Cu-Zn', ['Cu', 'Zn'], 'Cu', phase_alpha, phase_gamma)


@pytest.fixture
def mock_atom_data():
    """Return a copy of the ATOM_DATA dictionary for testing."""
    return {
        'Cu': {
            'a': 3.5691940,
            'dimer_dist': 2.248,
        }
    }


@pytest.fixture
def mock_logger():
    """Return a MagicMock logger."""
    return MagicMock()
