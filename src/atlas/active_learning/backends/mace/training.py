"""MACE training utilities.

Extracted from ``atlas.active_learning.active_learning_utils`` as part of
the MLIP-agnostic refactoring.
"""

from __future__ import annotations

from pathlib import Path

import torch
from aiida import orm


def update_mace_train_settings_dict(
    settings_dict: dict,
    train_data_path: str,
    curr_model: str,
    curr_iter: int,
    db_size: int,
    containerized: orm.Bool = False,
):
    """Update the MACE training settings dictionary with the new database path."""
    if isinstance(settings_dict, orm.Dict):
        settings_dict: dict = settings_dict.get_dict()

    if isinstance(train_data_path, orm.Str):
        train_data_path: Path = Path(train_data_path.value)
    elif isinstance(train_data_path, str):
        train_data_path: Path = Path(train_data_path)

    if containerized.value is True:
        train_data_path = Path('/atl_data') / train_data_path.name
        settings_dict['train_file'] = str(train_data_path)
        settings_dict['results_dir'] = str(Path('/atl_data') / 'results')
        settings_dict['checkpoints_dir'] = str(Path('/atl_data') / 'checkpoints')
        settings_dict['model_dir'] = str(Path('/atl_data'))
        settings_dict['log_dir'] = str(Path('/atl_data') / 'logs')
    else:
        settings_dict['train_file'] = str(train_data_path.name)

    curr_name = settings_dict['name']

    if db_size < settings_dict.get('batch_size', 0):
        settings_dict['batch_size'] = db_size // 2

    if isinstance(curr_model, orm.Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, orm.Int):
        curr_iter = curr_iter.value

    settings_dict['name'] = (
        str(curr_model) + '_' + curr_name + '_al-iteration_' + str(curr_iter)
    )

    return orm.Dict(settings_dict)


def create_mace_lammps_model_impl(model_file: orm.SinglefileData):
    """Create a LAMMPS potential from a MACE model (inner logic, no @calcfunction).

    The ``@calcfunction``-decorated version lives in
    ``active_learning_utils.create_mace_lammps_model`` for AiiDA provenance.

    Parameters
    ----------
    model_file : orm.SinglefileData
        A MACE model file to convert to a LAMMPS potential.

    Returns
    -------
    orm.SinglefileData
        A LAMMPS potential file generated from the MACE model.
    """
    from e3nn.util import jit
    from mace.calculators import LAMMPS_MACE

    with model_file.as_path() as model_path:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model = model.double().to('cpu')
        lammps_model = LAMMPS_MACE(model)
        lammps_model_compiled = jit.compile(lammps_model)

        new_model_path = str(model_path) + '-lammps.pt'

        lammps_model_compiled.save(new_model_path)

        return orm.SinglefileData(file=new_model_path)
