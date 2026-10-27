"""DFT convergence metadata helpers.

The flags are stored in ``Atoms.info`` and use a ``atl_dft_*`` prefix:

``atl_dft_scf_converged``
    Electronic SCF convergence. Always set when it can be determined.
``atl_dft_ionic_converged``
    Ionic relaxation convergence. Only set for relaxation calculations.
``atl_dft_is_relaxation``
    Whether the calculation was an ionic relaxation.
``atl_dft_converged``
    Overall usability flag. For static calculations it equals the SCF flag.
``atl_dft_convergence_source``
    Where the flags came from (currently ``vasprun``).

Missing keys mean that convergence is unknown, not that it failed.
"""

from __future__ import annotations

import tempfile
import warnings

from aiida import orm
from ase import Atoms
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning

from atlas.core.metadata import (
    DFT_CONVERGED_KEY,
    DFT_CONVERGENCE_SOURCE_KEY,
    DFT_IONIC_CONVERGED_KEY,
    DFT_IS_RELAXATION_KEY,
    DFT_SCF_CONVERGED_KEY,
)

_RELAXATION_IBRION_VALUES = {1, 2, 3, 5, 6, 7, 8}


def _as_int(value):
    """Convert a VASP parameter to an integer when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def derive_dft_convergence(vasprun) -> dict:
    """Derive ``atl_dft_*`` flags from a pymatgen ``Vasprun``.

    Static calculations (``IBRION = -1``) do not have a meaningful ionic
    convergence criterion, so ``atl_dft_ionic_converged`` is omitted and the
    overall flag is based on the SCF result only.

    Parameters
    ----------
    vasprun : pymatgen.io.vasp.Vasprun
        Parsed VASP run.

    Returns
    -------
    dict
        Flags that could be determined. Empty when SCF convergence is unknown.
    """
    parameters = getattr(vasprun, 'parameters', {}) or {}
    ibrion = _as_int(parameters.get('ibrion'))
    is_relaxation = ibrion in _RELAXATION_IBRION_VALUES

    scf_converged = getattr(vasprun, 'converged_electronic', None)
    if scf_converged is None:
        return {}
    scf_converged = bool(scf_converged)

    flags = {
        DFT_SCF_CONVERGED_KEY: scf_converged,
        DFT_IS_RELAXATION_KEY: is_relaxation,
        DFT_CONVERGENCE_SOURCE_KEY: 'vasprun',
    }

    if is_relaxation:
        ionic_converged = getattr(vasprun, 'converged_ionic', None)
        if ionic_converged is not None:
            ionic_converged = bool(ionic_converged)
            flags[DFT_IONIC_CONVERGED_KEY] = ionic_converged
            flags[DFT_CONVERGED_KEY] = scf_converged and ionic_converged
        # If ionic convergence is unavailable, leave the overall flag unset
        # instead of claiming failure.
    else:
        flags[DFT_CONVERGED_KEY] = scf_converged

    return flags


def _read_vasprun_from_node(node: orm.CalcJobNode) -> Vasprun | None:
    """Read and parse ``vasprun.xml`` from an AiiDA VASP calculation node."""
    try:
        retrieved = node.outputs.retrieved
        content = retrieved.get_object_content('vasprun.xml', 'rb')
    except Exception:
        return None

    with tempfile.NamedTemporaryFile(suffix='.xml') as temporary_file:
        temporary_file.write(content)
        temporary_file.flush()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UnconvergedVASPWarning)
                return Vasprun(
                    temporary_file.name,
                    parse_dos=False,
                    parse_eigen=False,
                    parse_potcar_file=False,
                    exception_on_bad_xml=False,
                )
        except Exception:
            return None


def extract_dft_convergence_from_node(node: orm.CalcJobNode) -> dict:
    """Extract ``atl_dft_*`` flags from a real aiida-vasp calculation node.

    Parameters
    ----------
    node : aiida.orm.CalcJobNode
        A finished VASP CalcJob. Nodes without a retrievable ``vasprun.xml``
        return an empty dictionary.

    Returns
    -------
    dict
        Flags that could be determined.
    """
    vasprun = _read_vasprun_from_node(node)
    if vasprun is None:
        return {}
    return derive_dft_convergence(vasprun)


def add_dft_convergence_flags(
    atoms: Atoms,
    node: orm.CalcJobNode | None = None,
    vasprun: Vasprun | None = None,
) -> Atoms:
    """Add ``atl_dft_*`` flags to an ASE ``Atoms`` object in place.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to annotate.
    node : aiida.orm.CalcJobNode, optional
        VASP calculation node. Used to read ``vasprun.xml`` when ``vasprun``
        is not supplied.
    vasprun : pymatgen.io.vasp.Vasprun, optional
        Already parsed VASP run.

    Returns
    -------
    ase.Atoms
        The annotated object.
    """
    if vasprun is not None:
        flags = derive_dft_convergence(vasprun)
    elif node is not None:
        flags = extract_dft_convergence_from_node(node)
    else:
        return atoms

    atoms.info.update(flags)
    return atoms
