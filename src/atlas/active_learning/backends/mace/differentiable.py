"""Autograd-connected MACE descriptors.

``MLIPDescriptorProvider`` returns detached numpy arrays, which is all the
active-learning loop needs. Latent-space inversion needs the opposite: the
descriptor has to stay attached to the autograd graph so that
``d(descriptor)/d(positions)`` is available. Composing that with a trained
autoencoder encoder gives a differentiable structure-to-latent map, which can
then be inverted by gradient descent on the positions.

The model is called directly rather than through ``MACECalculator``, because
``MACECalculator.get_descriptors`` ends in ``.detach().cpu().numpy()`` and
because ``compute_force=True`` runs its own ``autograd.grad`` pass, which
conflicts with positions that already require gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class MaceBatchState:
    """One prepared MACE batch, with its leaf tensors exposed.

    Satisfies the ``DescriptorBatchState`` protocol in
    ``atlas.active_learning.backends._base``.
    """

    #: The batch dict handed to the model; its ``positions`` entry is
    #: :attr:`positions`.
    batch_dict: dict

    #: ``(n_atoms, 3)`` positions the model sees. Equal to
    #: :attr:`free_positions` when no strain was applied.
    positions: Any

    #: ``(n_atoms, 3)`` unstrained positions. Always the autograd leaf.
    free_positions: Any

    #: ``(3, 3)`` strain leaf, or None when the cell is held fixed.
    strain: Any

    #: ``(3, 3)`` lattice vectors of the (single) structure, strained when
    #: :attr:`strain` is set.
    cell: Any

    #: ``(2, n_edges)`` sender/receiver index pairs.
    edge_index: Any

    #: ``(n_edges, 3)`` Cartesian periodic-image offset per edge.
    shifts: Any


class MaceDifferentiableDescriptors:
    """Evaluate MACE descriptors with gradients w.r.t. positions and cell.

    Satisfies the ``DifferentiableDescriptorHandle`` protocol.

    The emitted tensor reproduces ``MACECalculator.get_descriptors`` exactly:
    raw ``node_feats`` is *not* the same thing in general. The calculator
    additionally applies ``extract_invariant`` and a per-layer width slice,
    which collapse to the identity only when ``l_max == 0`` (true for
    ``mace:mp-small``, false for most other models). Getting this wrong is
    silent when the widths happen to agree, so the reduction is replicated here
    rather than approximated.

    Only ``models[0]`` is used. For a committee the members share an
    architecture but not weights, so descriptors -- and therefore the latent
    map -- are defined by whichever model built that map.
    """

    def __init__(
        self,
        model_path: str | Path,
        settings: dict | None = None,
    ) -> None:
        from atlas.active_learning.backends.mace.calculator import (
            create_mace_calculator,
        )

        settings = settings or {}
        self.device = str(settings.get('device', 'cpu'))
        dtype_name = str(settings.get('dtype', 'float64'))

        self._calculator = create_mace_calculator(
            model_path=model_path,
            device=self.device,
            dtype=dtype_name,
        )
        self._model = self._calculator.models[0]
        self._model.eval()

        # The calculator may have cast the model on load (mace_mp/mace_off pick
        # their own default dtype), so follow the weights rather than the
        # requested name -- otherwise the batch and the model disagree.
        self.dtype = next(self._model.parameters()).dtype

        self._reduction = _descriptor_reduction(self._model)
        self.descriptor_dim = self._reduction.width
        self.supports_cell_gradients = True

    # -------------------------------------------------------------- batching

    def prepare(self, atoms: Atoms, strain: Any = None) -> MaceBatchState:
        """Build a batch with a freshly computed neighbour list.

        MACE fixes its neighbour list when the batch is built. Over an
        inversion the atoms move far enough that reusing a stale list silently
        corrupts the descriptor -- at ~1.5 A of displacement the per-atom error
        reaches the scale of the descriptor norm itself -- so callers must
        re-``prepare`` as the geometry evolves rather than reusing a state.
        """
        import torch

        batch_dict = self._calculator._atoms_to_batch(atoms).to_dict()
        for key, val in batch_dict.items():
            if not isinstance(val, torch.Tensor):
                continue
            if torch.is_floating_point(val):
                batch_dict[key] = val.to(dtype=self.dtype, device=self.device)
            else:
                batch_dict[key] = val.to(device=self.device)

        free_positions = batch_dict['positions'].clone().detach().requires_grad_(True)

        if strain is None:
            positions = free_positions
            shifts = batch_dict['shifts']
            cell = batch_dict.get('cell')
            if cell is not None:
                cell = cell.view(-1, 3, 3)[0]
        else:
            positions, shifts, cell = self._apply_strain(
                batch_dict, free_positions, strain
            )
            batch_dict['cell'] = cell.reshape(-1, 3)

        batch_dict['positions'] = positions
        batch_dict['shifts'] = shifts

        return MaceBatchState(
            batch_dict=batch_dict,
            positions=positions,
            free_positions=free_positions,
            strain=strain,
            cell=cell,
            edge_index=batch_dict['edge_index'],
            shifts=shifts,
        )

    def _apply_strain(self, batch_dict: dict, free_positions, strain):
        """Deform positions, cell and shifts by the symmetric part of *strain*.

        Mirrors ``mace.modules.utils.get_symmetric_displacement``, which is how
        MACE itself gets cell gradients for the stress. Only the symmetric part
        is used: the descriptor is rotation invariant, so the antisymmetric
        part is a three-dimensional gauge direction with identically zero
        gradient, and a line search left free to wander in it stalls.

        ``shifts`` is rebuilt from ``unit_shifts`` and the strained cell rather
        than scaled directly, matching MACE's own code path.
        """
        import torch

        sym = 0.5 * (strain + strain.transpose(-1, -2))

        positions = free_positions + free_positions @ sym

        cell = batch_dict['cell'].view(-1, 3, 3)
        cell = cell + torch.matmul(cell, sym)

        sender = batch_dict['edge_index'][0]
        batch_idx = batch_dict['batch']
        shifts = torch.einsum(
            'be,bec->bc', batch_dict['unit_shifts'], cell[batch_idx[sender]]
        )
        return positions, shifts, cell[0]

    # ----------------------------------------------------------- descriptors

    def descriptors(self, state: MaceBatchState):
        """Return ``(n_atoms, descriptor_dim)`` descriptors, still differentiable.

        ``compute_force=False`` matters: the force path runs its own
        ``autograd.grad`` over the positions, which conflicts with a caller
        that is already differentiating through them.
        """
        out = self._model(state.batch_dict, compute_force=False)
        return self._reduction(out['node_feats'])


@dataclass(frozen=True)
class _DescriptorReduction:
    """The ``node_feats`` -> descriptor reduction used by MACECalculator."""

    num_layers: int
    num_invariant_features: int
    l_max: int
    to_keep: int
    width: int

    def __call__(self, node_feats):
        from mace.modules.utils import extract_invariant

        # extract_invariant is slicing plus a cat, so gradients flow through it.
        invariants = extract_invariant(
            node_feats,
            num_layers=self.num_layers,
            num_features=self.num_invariant_features,
            l_max=self.l_max,
        )
        return invariants[:, : self.to_keep]


def _descriptor_reduction(model) -> _DescriptorReduction:
    """Derive the reduction parameters from a MACE model.

    Follows ``MACECalculator.get_descriptors`` with its defaults
    (``invariants_only=True``, ``num_layers=-1`` meaning all layers).
    """
    from e3nn import o3

    num_interactions = int(model.num_interactions)
    irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
    l_max = irreps_out.lmax
    num_invariant_features = irreps_out.dim // (l_max + 1) ** 2

    per_layer_features = [irreps_out.dim] * num_interactions
    # The last layer builds no equivariant features.
    per_layer_features[-1] = num_invariant_features
    to_keep = int(sum(per_layer_features))

    # extract_invariant emits num_layers * num_invariant_features columns; the
    # to_keep slice can only ever be a no-op with these defaults, but it is
    # kept so the behaviour tracks the calculator if that changes upstream.
    width = min(to_keep, num_interactions * num_invariant_features)

    return _DescriptorReduction(
        num_layers=num_interactions,
        num_invariant_features=num_invariant_features,
        l_max=l_max,
        to_keep=to_keep,
        width=width,
    )
