"""End-to-end mock active-learning loop integration test.

Runs one full iteration of the AL workchain with every stage mocked
(``[debug.mock]``), i.e. the remote CalcJobs are replaced by instant in-process
calcfunctions. This is a smoke test of the workchain orchestration (backend
selection, container-setting resolution, stage sequencing, loop control), not of
the science.

It uses AiiDA's ``aiida_localhost`` fixture, which creates a throwaway profile
and ``localhost`` computer, so it never depends on a developer's own AiiDA
configuration. The whole module is skipped when aiida-core is unavailable.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip('aiida')


def _seed_database(path):
    """Write a small Cu bulk seed database (extxyz)."""
    from ase.build import bulk
    from ase.io import write

    structs = []
    for a in (3.5, 3.6, 3.7, 3.8):
        atoms = bulk('Cu', 'fcc', a=a, cubic=True)
        atoms.info['atl_id'] = str(uuid.uuid4())
        atoms.info['mdb_struct_type'] = 'bulk'
        atoms.info['config_type'] = 'bulk'
        structs.append(atoms)
    write(str(path), structs, format='extxyz')


def _mock_config(*, computer: str, profile: str, seed_path, results_dir) -> dict:
    """Return a complete, valid AL config dict with all stages mocked."""
    metadata = {'computer': computer, 'options': {}}
    return {
        'active_learning': {
            'aiida_profile': profile,
            'run_name': 'mock_shakedown',
            'init_db_path': str(seed_path),
            'results_dir': str(results_dir),
            'final_db_name': 'mock_train_db',
            'max_iterations': 1,
            'al_mode': 'md',
        },
        'debug': {'mock': {'enable': True, 'stages': [], 'dft_potential': 'emt'}},
        'mlip': {'training_backend': 'mace'},
        'mace_train': {'metadata': dict(metadata)},
        'committee_eval': {'committee_num_models': 2, 'metadata': dict(metadata)},
        'al_seed': {
            'seed_size_frac': 0.5,
            'seed_min_num_structs': 1,
            'seed_max_num_structs': 2,
            'seed_select_settings': {
                'seed_select_type': 'random',
                'small_first_max_size': 20,
                'small_first_max_iter': 2,
            },
        },
        'md': {
            'metadata': dict(metadata),
            'parameters': {
                'temperature_list_K': [300.0],
                'al_keep_struct_every_n_ps': 0.5,
                'max_temp_multiplier': 1.3,
                'num_steps': 10,
                'timestep_duration_ps': 0.001,
                'langevin_friction_ps-1': 0.01,
                'npt_ttime_fs': 25.0,
                'npt_ptime_fs': 75.0,
                'md_stage_order': [],
                'gather_traj_cnt_lattice': False,
                'use_kokkos': False,
                'device': 'cpu',
                'default_dtype': 'float32',
            },
            'filters': {
                'check_atoms_no_neighbor': {
                    'enable': False,
                    'covalent_radius_multiplier': 1.5,
                },
                'layer_distance': {'enable': False, 'max_layer_distance_ang': 5.0},
                'exploding_structures': {'enable': False},
            },
        },
        'descriptors': {
            'descriptor_type': 'soap',
            'metadata': dict(metadata),
            'autoencoder': {'train_settings': {'num_epochs': 5}},
        },
        'extrapolation': {'check_extrapolation_type': 'none'},
        'interpolation': {
            'model_acc_multiplier': 1.0,
            'disagreement_check_type': 'training',
        },
        'safeguard': {
            'enable': False,
            'target_structure_mode': 'base',
            'metadata': dict(metadata),
        },
        'test_db': {
            'use_test_db': False,
            'model_settings': {},
            'metadata': dict(metadata),
        },
        'code': {'container': {'use_container': False}},
        'dft': {
            'dft_method': 'mace',
            'mace': {'mace_potential_path': str(results_dir / 'dummy.model')},
            'vasp': {
                'potential_family': 'dummy',
                'queue': {
                    'queue_type': 'direct',
                    'computer': computer,
                    'code_string': f'dummy@{computer}',
                    'options_resources': {'num_machines': 1},
                },
            },
        },
    }


def test_mock_al_loop_completes(aiida_localhost, tmp_path, monkeypatch):
    """One fully-mocked AL iteration runs to completion through the workchain."""
    from aiida.engine import run_get_node
    from aiida.manage import get_manager

    import atlas  # noqa: F401  (resolve circular imports)
    from atlas.core.command_line.cli_active_learning import (
        create_active_learning_builder,
    )
    from atlas.core.command_line.command_line_utils import apply_defaults

    profile = get_manager().get_profile().name
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    seed_path = tmp_path / 'seed_db.xyz'
    _seed_database(seed_path)

    config = _mock_config(
        computer=aiida_localhost.label,
        profile=profile,
        seed_path=seed_path,
        results_dir=results_dir,
    )

    # The workchain re-reads the TOML file at runtime (mock config, backend), so
    # it must exist on disk; the builder needs its path as the ``toml_file`` input.
    import tomli_w

    config_file = tmp_path / 'mock_al.toml'
    config_file.write_bytes(tomli_w.dumps(config).encode())

    defaulted = apply_defaults(config, [])

    # The workchain writes its working files under the current directory.
    monkeypatch.chdir(tmp_path)

    from aiida import orm

    builder = create_active_learning_builder(defaulted, toml_dict_path=config_file)
    # debug_mode=True runs stages in the foreground; set by the CLI, not the builder.
    builder.active_learning.debug_mode = orm.Bool(True)
    _, node = run_get_node(builder)

    assert node.is_finished_ok, (
        f'mock AL loop did not finish ok: exit_status={node.exit_status}, '
        f'message={node.exit_message}'
    )
