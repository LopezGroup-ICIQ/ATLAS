"""Tests for the atomic extxyz writer used by the DFT result files."""

import pathlib

import pytest
from ase import Atoms
from ase.io import read as ase_read

from atlas.core import code_utils as atl_cut


@pytest.fixture
def structures():
    """Two small labelled structures."""
    images = []
    for idx in range(2):
        atoms = Atoms(
            symbols='Cu2',
            positions=[[0, 0, 0], [2.0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        atoms.info['atl_id'] = f'uuid-{idx}'
        atoms.info['REF_energy'] = -1.5 * (idx + 1)
        images.append(atoms)
    return images


def test_atomic_write_round_trip(tmp_path, structures):
    """The written file is readable and contains every structure."""
    target = tmp_path / 'results.xyz'

    returned = atl_cut.atomic_write_extxyz(target, structures)

    assert returned == target
    read_back = ase_read(target, format='extxyz', index=':')
    assert len(read_back) == len(structures)
    assert [struct.info['atl_id'] for struct in read_back] == ['uuid-0', 'uuid-1']


def test_atomic_write_leaves_no_temporary_file(tmp_path, structures):
    """The temporary file used for the atomic rename is not left behind."""
    target = tmp_path / 'results.xyz'

    atl_cut.atomic_write_extxyz(target, structures)

    assert sorted(path.name for path in tmp_path.iterdir()) == ['results.xyz']


def test_failed_write_preserves_previous_file(tmp_path, structures, monkeypatch):
    """A write that raises leaves the previously written file untouched."""
    target = tmp_path / 'results.xyz'
    atl_cut.atomic_write_extxyz(target, structures)
    original = target.read_text()

    def boom(*args, **kwargs):
        raise RuntimeError('vasprun exploded')

    monkeypatch.setattr('ase.io.write', boom)

    with pytest.raises(RuntimeError):
        atl_cut.atomic_write_extxyz(target, structures)

    # The good file survives and no half-written temporary is left around.
    assert target.read_text() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ['results.xyz']


def test_atomic_write_accepts_string_path(tmp_path, structures):
    """A plain string path works as well as a Path."""
    target = tmp_path / 'results.xyz'

    returned = atl_cut.atomic_write_extxyz(str(target), structures)

    assert returned == pathlib.Path(target)
    assert target.exists()
