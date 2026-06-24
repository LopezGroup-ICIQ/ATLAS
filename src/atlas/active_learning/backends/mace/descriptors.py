"""MACE descriptor generation.

Extracted from ``atlas.active_learning.active_learning_utils`` as part of
the MLIP-agnostic refactoring.
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from pathlib import Path, PosixPath
from uuid import uuid4

import numpy as np
import torch
from ase import Atoms

from atlas.core import code_utils as atl_cut
from atlas.core.exceptions import MissingMandatoryParameterError


def _suppress_stdout():
    """Temporarily redirect standard output to devnull."""
    return redirect_stdout(open(os.devnull, 'w'))


def generate_descriptors_mace(
    model_path: str,
    database: list[Atoms],
    descriptor_settings: dict,
    outer_average: bool = False,
    verbose: bool = False,
) -> tuple[dict, np.ndarray, list[str]]:
    """Generate per-structure MACE descriptors for a database.

    Parameters
    ----------
    model_path : str
        Path to a trained MACE model file, or a foundation model
        identifier (e.g. ``"mace:mp-small"``, ``"mace:off-medium"``).
    database : list[Atoms]
        List of ASE Atoms objects.
    descriptor_settings : dict
        Settings dict containing ``device`` and ``dtype`` keys.
    outer_average : bool
        If True, average atom-level descriptors into a single
        structure-level vector (analogous to SOAP outer averaging).
    verbose : bool
        If True, show a progress bar.

    Returns
    -------
    descriptor_dict : dict
        Mapping ``{atl_id: {'descriptors': [...], 'latent_space': []}}``
    descriptor_arr : np.ndarray
        Vertically stacked descriptor array.
    uuid_list : list[str]
        UUIDs assigned to structures that lacked an ``atl_id``.
    """
    from mace.calculators import MACECalculator

    if model_path is None:
        raise MissingMandatoryParameterError(
            'Missing model path for MACE descriptor generation.'
        )

    device = descriptor_settings.get('device', 'cpu')
    dtype = descriptor_settings.get('dtype', 'float32')

    is_mp_foundation = False
    is_off_foundation = False

    if isinstance(model_path, PosixPath | Path):
        model_path = str(model_path)

    with _suppress_stdout():
        try:
            model_loaded = torch.load(model_path, map_location=torch.device(device))
        except RuntimeError:
            model_loaded = torch.load(model_path, map_location=torch.device('cpu'))
        except FileNotFoundError as e:
            if 'mace:mp-' in model_path:
                model_variant = model_path.split('mace:mp-')[-1]
                if model_variant in ['small', 'medium', 'large', 'medium-mpa-0']:
                    is_mp_foundation = True
                    model_loaded = model_variant
            elif 'mace:off-' in model_path:
                model_variant = model_path.split('mace:off-')[-1]
                if model_variant in ['small', 'medium', 'large']:
                    is_off_foundation = True
                    model_loaded = model_variant
            else:
                raise FileNotFoundError(
                    'Model file not found. Please provide a valid model path'
                    'or a mace foundation model name, using the following syntax:'
                    ' "mace:mp-small", "mace:off-medium", etc.'
                ) from e

        if is_mp_foundation:
            from mace.calculators import mace_mp

            calculator = mace_mp(
                model=model_loaded,
                device=device,
                default_dtype=dtype,
            )
        elif is_off_foundation:
            from mace.calculators import mace_off

            calculator = mace_off(
                model=model_loaded,
                device=device,
                default_dtype=dtype,
            )
        else:
            calculator = MACECalculator(
                models=[model_loaded], device=device, default_dtype=dtype
            )

    descriptor_dict = {}
    descriptor_list = []
    uuid_list = []

    tot_num_structures = len(database)

    iterable = (
        atl_cut.atl_show_progress(
            enumerate(database),
            total=tot_num_structures,
            interval=100,
            prepend='MACE:',
        )
        if verbose
        else enumerate(database)
    )

    for _, struct in iterable:
        if struct.info.get('atl_id'):
            struct_key = struct.info.get('atl_id')
        elif struct.info.get('aiida_uuid'):
            struct_key = struct.info.get('aiida_uuid')
        else:
            struct_key = str(uuid4())
            uuid_list.append(struct_key)
            struct.info['atl_id'] = struct_key

        if descriptor_dict.get(struct_key) is None:
            descriptor_dict[struct_key] = {
                'descriptors': [],
                'latent_space': [],
            }

        curr_struct_descriptors = calculator.get_descriptors(struct)

        if outer_average:
            curr_struct_descriptors = np.mean(
                curr_struct_descriptors, axis=0, keepdims=True
            )

        descriptor_list.append(curr_struct_descriptors)
        descriptor_dict[struct_key]['descriptors'].append(curr_struct_descriptors)

    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr, uuid_list
