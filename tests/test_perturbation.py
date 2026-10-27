"""Tests for the random atomic displacement helper."""

import numpy as np
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from atlas.core.clusters import apply_gauss_perturb_list
from atlas.core.structure import Structure as AtlStructure
from atlas.core.utils import _RATTLE_STDEV_FACTOR, gauss_perturb


def _pmg_structure(atoms):
    return AseAtomsAdaptor().get_structure(atoms)


def test_perturbations_are_not_rigid_translations():
    structure = _pmg_structure(bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2)))
    original_positions = structure.cart_coords.copy()
    original_distances = structure.distance_matrix

    perturbed = gauss_perturb(structure, center=0.04, seed=1)
    displacements = perturbed.cart_coords - original_positions

    assert not np.allclose(displacements, displacements[0])
    assert not np.allclose(perturbed.distance_matrix, original_distances)


def test_same_seed_is_reproducible():
    structure = _pmg_structure(bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2)))

    first = gauss_perturb(structure, center=0.04, seed=7)
    second = gauss_perturb(structure, center=0.04, seed=7)
    different = gauss_perturb(structure, center=0.04, seed=8)

    assert np.allclose(first.cart_coords, second.cart_coords)
    assert not np.allclose(first.cart_coords, different.cart_coords)


def test_rms_displacement_matches_legacy_uniform_range():
    structure = _pmg_structure(bulk('Cu', 'fcc', a=3.6).repeat((4, 4, 4)))
    original_positions = structure.cart_coords.copy()

    squared_displacements = []
    for seed in range(4):
        perturbed = gauss_perturb(structure, center=0.04, seed=seed)
        displacements = perturbed.cart_coords - original_positions
        squared_displacements.extend(np.sum(displacements**2, axis=1))

    rms_displacement = np.sqrt(np.mean(squared_displacements))
    legacy_rms = np.sqrt((0.02**2 + 0.02 * 0.08 + 0.08**2) / 3)

    assert np.sqrt(7.0 / 12.0) == _RATTLE_STDEV_FACTOR
    assert np.isclose(rms_displacement, legacy_rms, rtol=0.05)


def test_selective_dynamics_are_respected():
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.0),
        ['Cu', 'Cu'],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        site_properties={
            'selective_dynamics': [
                [False, False, False],
                [True, True, True],
            ]
        },
    )

    perturbed = gauss_perturb(structure, center=0.04, seed=3)

    assert np.allclose(perturbed.cart_coords[0], structure.cart_coords[0])
    assert not np.allclose(perturbed.cart_coords[1], structure.cart_coords[1])


def test_cluster_perturbation_list_produces_distinct_structures():
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(8.0),
        ['Cu', 'Cu'],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    cluster = AtlStructure(structure=structure)

    perturbed = apply_gauss_perturb_list(
        repeat=2,
        cluster_list=[cluster],
        center=0.04,
        seed=11,
    )

    first_positions = perturbed[0].structure.cart_coords
    second_positions = perturbed[1].structure.cart_coords

    assert len(perturbed) == 2
    assert not np.allclose(first_positions, second_positions)
