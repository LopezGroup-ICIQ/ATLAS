"""Shared bookkeeping for per-structure descriptor generation.

Backends that expose per-atom embeddings (MACE via ``get_descriptors``, Orb and
EquiformerV3 via a forward hook on their encoder) all need the same wrapper: for
each structure resolve a stable key (``atl_id``), collect the per-atom
descriptor array, optionally average it to a single structure-level vector, and
stack everything into one array keyed by structure. Only the "get the per-atom
array for one structure" step differs, so it is passed in as ``extract_fn``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from atlas.core import code_utils as atl_cut

if TYPE_CHECKING:
    from ase import Atoms


def assemble_structure_descriptors(
    database: list[Atoms],
    extract_fn: Callable[[Atoms], np.ndarray],
    outer_average: bool = False,
    verbose: bool = False,
    prepend: str = '',
) -> tuple[dict, np.ndarray, list[str]]:
    """Build the descriptor dict/array for a database from a per-structure hook.

    Parameters
    ----------
    database : list[Atoms]
        Structures to describe.
    extract_fn : Callable[[Atoms], np.ndarray]
        Returns the per-atom descriptor array ``(n_atoms, dim)`` for one
        structure.
    outer_average : bool
        If True, average the per-atom descriptors into a single structure-level
        vector (analogous to SOAP outer averaging).
    verbose : bool
        Show a progress bar.
    prepend : str
        Progress-bar label.

    Returns
    -------
    descriptor_dict : dict
        ``{atl_id: {'descriptors': [...], 'latent_space': []}}``.
    descriptor_arr : np.ndarray
        Vertically stacked descriptor array.
    uuid_list : list[str]
        UUIDs assigned to structures that lacked an ``atl_id``.
    """
    descriptor_dict: dict = {}
    descriptor_list: list[np.ndarray] = []
    uuid_list: list[str] = []

    iterable = (
        atl_cut.atl_show_progress(
            enumerate(database),
            total=len(database),
            interval=100,
            prepend=prepend,
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
            descriptor_dict[struct_key] = {'descriptors': [], 'latent_space': []}

        curr = np.asarray(extract_fn(struct))
        if outer_average:
            curr = np.mean(curr, axis=0, keepdims=True)

        descriptor_list.append(curr)
        descriptor_dict[struct_key]['descriptors'].append(curr)

    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr, uuid_list
