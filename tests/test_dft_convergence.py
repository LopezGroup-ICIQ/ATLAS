"""Tests for DFT convergence metadata and filtering.

The real-AiiDA tests are opt-in: set ``ATLAS_REAL_AIIDA_TESTS=1`` and run
this file without the isolated ``AIIDA_PATH`` used by the normal test suite.
"""

import os

import pytest
from ase import Atoms

from atlas.core.dft_convergence import (
    add_dft_convergence_flags,
    derive_dft_convergence,
    extract_dft_convergence_from_node,
)
from atlas.core.filtering.structure_filters import apply_filter_dft_converged
from atlas.core.metadata import (
    DFT_CONVERGED_KEY,
    DFT_CONVERGENCE_SOURCE_KEY,
    DFT_IONIC_CONVERGED_KEY,
    DFT_IS_RELAXATION_KEY,
    DFT_SCF_CONVERGED_KEY,
)


class FakeVasprun:
    """Minimal Vasprun-like object for unit tests."""

    def __init__(self, ibrion, scf, ionic=None):
        self.parameters = {'ibrion': ibrion}
        self.converged_electronic = scf
        self.converged_ionic = ionic


class TestDeriveDftConvergence:
    """Tests for deriving flags from parsed VASP runs."""

    def test_static_converged_ignores_ionic_flag(self):
        flags = derive_dft_convergence(FakeVasprun(-1, True, False))

        assert flags[DFT_SCF_CONVERGED_KEY] is True
        assert flags[DFT_IS_RELAXATION_KEY] is False
        assert flags[DFT_CONVERGED_KEY] is True
        assert DFT_IONIC_CONVERGED_KEY not in flags
        assert flags[DFT_CONVERGENCE_SOURCE_KEY] == 'vasprun'

    def test_static_scf_failure(self):
        flags = derive_dft_convergence(FakeVasprun(-1, False, False))

        assert flags[DFT_SCF_CONVERGED_KEY] is False
        assert flags[DFT_CONVERGED_KEY] is False
        assert DFT_IONIC_CONVERGED_KEY not in flags

    def test_relaxation_converged(self):
        flags = derive_dft_convergence(FakeVasprun(2, True, True))

        assert flags[DFT_IS_RELAXATION_KEY] is True
        assert flags[DFT_SCF_CONVERGED_KEY] is True
        assert flags[DFT_IONIC_CONVERGED_KEY] is True
        assert flags[DFT_CONVERGED_KEY] is True

    def test_relaxation_ionic_failure(self):
        flags = derive_dft_convergence(FakeVasprun(2, True, False))

        assert flags[DFT_IONIC_CONVERGED_KEY] is False
        assert flags[DFT_CONVERGED_KEY] is False

    def test_missing_scf_information_is_unknown(self):
        assert derive_dft_convergence(FakeVasprun(-1, None, None)) == {}


class TestDftConvergenceFilter:
    """Tests for the dft_converged structure filter."""

    def _atoms(self, **info):
        atoms = Atoms('Cu', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        atoms.info.update(info)
        return atoms

    def test_missing_metadata_is_kept_by_default(self):
        atoms = self._atoms()

        assert not apply_filter_dft_converged(atoms)
        assert apply_filter_dft_converged(atoms, remove_unknown=True)

    def test_explicit_scf_failure_is_flagged(self):
        atoms = self._atoms(**{DFT_SCF_CONVERGED_KEY: False, DFT_CONVERGED_KEY: False})

        assert apply_filter_dft_converged(atoms)

    def test_static_converged_is_kept_even_with_require_ionic(self):
        atoms = self._atoms(
            **{
                DFT_SCF_CONVERGED_KEY: True,
                DFT_IS_RELAXATION_KEY: False,
                DFT_CONVERGED_KEY: True,
            }
        )

        assert not apply_filter_dft_converged(atoms, require_ionic=True)

    def test_relaxation_ionic_failure_is_flagged_when_requested(self):
        atoms = self._atoms(
            **{
                DFT_SCF_CONVERGED_KEY: True,
                DFT_IONIC_CONVERGED_KEY: False,
                DFT_IS_RELAXATION_KEY: True,
            }
        )

        assert not apply_filter_dft_converged(atoms)
        assert apply_filter_dft_converged(atoms, require_ionic=True)

    def test_explicit_overall_failure_is_always_flagged(self):
        atoms = self._atoms(**{DFT_CONVERGED_KEY: False})

        assert apply_filter_dft_converged(atoms)

    def test_relaxation_converged_is_kept(self):
        atoms = self._atoms(
            **{
                DFT_SCF_CONVERGED_KEY: True,
                DFT_IONIC_CONVERGED_KEY: True,
                DFT_IS_RELAXATION_KEY: True,
                DFT_CONVERGED_KEY: True,
            }
        )

        assert not apply_filter_dft_converged(atoms, require_ionic=True)


class TestAddDftConvergenceFlags:
    """Tests for annotating Atoms with convergence flags."""

    def test_adds_flags_from_vasprun(self):
        atoms = Atoms('Cu', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)

        add_dft_convergence_flags(atoms, vasprun=FakeVasprun(2, True, True))

        assert atoms.info[DFT_SCF_CONVERGED_KEY] is True
        assert atoms.info[DFT_IONIC_CONVERGED_KEY] is True
        assert atoms.info[DFT_IS_RELAXATION_KEY] is True
        assert atoms.info[DFT_CONVERGED_KEY] is True


@pytest.mark.skipif(
    os.environ.get('ATLAS_REAL_AIIDA_TESTS') != '1',
    reason='Set ATLAS_REAL_AIIDA_TESTS=1 to query the local AiiDA VASP database.',
)
class TestRealAiidaVaspNodes:
    """Integration tests against real aiida-vasp nodes when available."""

    @staticmethod
    def _first_with_flags(label_pattern):
        from aiida import load_profile, orm

        load_profile()
        qb = orm.QueryBuilder().append(
            orm.CalcJobNode,
            filters={
                'process_type': 'aiida.calculations:vasp.vasp',
                'label': {'like': label_pattern},
            },
            project=['id', 'label'],
        )
        for pk, label in qb.iterall():
            node = orm.load_node(pk)
            flags = extract_dft_convergence_from_node(node)
            if flags:
                return node, label, flags
        return None

    def test_static_vasp_node(self):
        result = self._first_with_flags('%sp_%')
        if result is None:
            pytest.skip('No static aiida-vasp node with a readable vasprun.xml.')

        node, label, flags = result
        assert isinstance(flags[DFT_SCF_CONVERGED_KEY], bool)
        assert flags[DFT_IS_RELAXATION_KEY] is False
        assert flags[DFT_CONVERGED_KEY] == flags[DFT_SCF_CONVERGED_KEY]
        assert DFT_IONIC_CONVERGED_KEY not in flags

        from atlas.active_learning.active_learning_utils import (
            build_ref_labelled_atoms,
        )

        atoms = build_ref_labelled_atoms(node)
        assert atoms.info[DFT_SCF_CONVERGED_KEY] == flags[DFT_SCF_CONVERGED_KEY]
        assert atoms.info[DFT_CONVERGED_KEY] == flags[DFT_CONVERGED_KEY]
        print(f'static node: pk={node.pk} label={label} flags={flags}')

    def test_relaxation_vasp_node(self):
        result = self._first_with_flags('%relax%')
        if result is None:
            pytest.skip('No relaxation aiida-vasp node with a readable vasprun.xml.')

        node, label, flags = result
        assert flags[DFT_IS_RELAXATION_KEY] is True
        assert isinstance(flags[DFT_SCF_CONVERGED_KEY], bool)
        if DFT_IONIC_CONVERGED_KEY in flags:
            assert flags[DFT_CONVERGED_KEY] == (
                flags[DFT_SCF_CONVERGED_KEY] and flags[DFT_IONIC_CONVERGED_KEY]
            )
        print(f'relaxation node: pk={node.pk} label={label} flags={flags}')
