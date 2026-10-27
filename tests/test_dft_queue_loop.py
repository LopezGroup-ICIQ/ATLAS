"""Tests for the DFT queue polling loop.

``aiida_utils`` imports aiida-core at module level, so the whole module is
skipped where aiida-core is not installed.
"""

import pytest
from ase import Atoms

pytest.importorskip('aiida', reason='aiida-core is required for the DFT queue loop')


@pytest.fixture
def structures():
    """Two small structures carrying the metadata the submission loop reads."""
    images = []
    for idx in range(2):
        atoms = Atoms(
            symbols='Cu2',
            positions=[[0, 0, 0], [2.0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        atoms.info['atl_id'] = f'uuid-{idx}'
        atoms.info['struct_name'] = f'struct-{idx}'
        images.append(atoms)
    return images


class _FakeNode:
    """Minimal stand-in for an AiiDA WorkChainNode."""

    def __init__(self, pk, finished_after=0):
        self.pk = pk
        self._checks = 0
        self._finished_after = finished_after

    @property
    def is_finished(self):
        self._checks += 1
        return self._checks > self._finished_after


@pytest.fixture
def queue_config():
    return {
        'general': {
            'aiida_group_name': 'test_group',
            'result_file_path': 'results',
            'max_batch': 1,
            'queue_check_interval_seconds': 240,
            'first_check_delay_seconds': 30,
            'dry_run': True,
        },
        'calculation': {'calc_type': 'single_point'},
        'incar': {'bulk': {}},
        'kpoints': {},
        'queue': {},
    }


def _patch_loop(monkeypatch, sleeps, submitted):
    """Replace submission and sleeping so the loop runs instantly."""
    from atlas.workflows import aiida_utils as atl_aut

    def fake_submit(index, **kwargs):
        submitted.append(index)
        return _FakeNode(pk=index)

    monkeypatch.setattr(atl_aut, 'submit_aiida_vasp_calculation', fake_submit)
    monkeypatch.setattr(atl_aut, 'update_db_with_dft_results', lambda db, queue: (0, 0))
    monkeypatch.setattr(atl_aut.time, 'sleep', lambda secs: sleeps.append(secs))


def test_first_wait_is_short(tmp_path, monkeypatch, queue_config, structures):
    """The first wait uses first_check_delay_seconds, not the full interval."""
    from atlas.workflows import aiida_utils as atl_aut

    sleeps, submitted = [], []
    _patch_loop(monkeypatch, sleeps, submitted)
    monkeypatch.chdir(tmp_path)

    atl_aut.run_dataframe_vasp_aiida_queue(
        initial_db=[struct.copy() for struct in structures],
        config_dict=queue_config,
        log_file_path=str(tmp_path / 'log'),
    )

    assert sleeps, 'the loop never waited'
    assert sleeps[0] == 30
    assert all(wait == 240 for wait in sleeps[1:])


def test_no_trailing_wait_after_last_calculation(
    tmp_path, monkeypatch, queue_config, structures
):
    """The loop exits without burning another interval once nothing is left."""
    from atlas.workflows import aiida_utils as atl_aut

    sleeps, submitted = [], []
    _patch_loop(monkeypatch, sleeps, submitted)
    monkeypatch.chdir(tmp_path)

    atl_aut.run_dataframe_vasp_aiida_queue(
        initial_db=[struct.copy() for struct in structures],
        config_dict=queue_config,
        log_file_path=str(tmp_path / 'log'),
    )

    # One wait per submitted structure at most: the final check does not sleep.
    assert len(sleeps) <= len(structures)


def test_every_structure_is_submitted(tmp_path, monkeypatch, queue_config):
    """Regression: the submission cap used to drop the tail of small databases.

    With ``max_batch = 1`` and ten structures the old
    ``total_loops < len(sel_struct_db)`` guard stopped submitting at iteration
    ten, silently leaving the last structure uncalculated.
    """
    from atlas.workflows import aiida_utils as atl_aut

    images = []
    for idx in range(10):
        atoms = Atoms(symbols='Cu', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        atoms.info['atl_id'] = f'uuid-{idx}'
        atoms.info['struct_name'] = f'struct-{idx}'
        images.append(atoms)

    sleeps, submitted = [], []
    _patch_loop(monkeypatch, sleeps, submitted)
    monkeypatch.chdir(tmp_path)

    atl_aut.run_dataframe_vasp_aiida_queue(
        initial_db=images,
        config_dict=queue_config,
        log_file_path=str(tmp_path / 'log'),
    )

    assert submitted == list(range(10))


def test_results_file_exists_from_the_first_check(
    tmp_path, monkeypatch, queue_config, structures
):
    """A results file is on disk before the first wait, not only at the end."""
    from atlas.workflows import aiida_utils as atl_aut

    submitted = []
    seen_during_first_wait = []

    def fake_sleep(secs):
        seen_during_first_wait.append(sorted(tmp_path.glob('atl_dft_*.xyz')))

    _patch_loop(monkeypatch, [], submitted)
    monkeypatch.setattr(atl_aut.time, 'sleep', fake_sleep)
    monkeypatch.chdir(tmp_path)

    atl_aut.run_dataframe_vasp_aiida_queue(
        initial_db=[struct.copy() for struct in structures],
        config_dict=queue_config,
        log_file_path=str(tmp_path / 'log'),
    )

    assert seen_during_first_wait
    assert seen_during_first_wait[0], 'no results file written before the first wait'
