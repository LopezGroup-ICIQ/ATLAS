"""Module containing general utilities for database creation."""

import itertools as it
import json as js
import os
import pathlib
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
from ase import Atoms, visualize
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.surface import Slab
from pymatgen.io import ase as pmg_ase
from pymatgen.io.ase import AseAtomsAdaptor

import atlas.core.initial_db as atl_indb
import atlas.core.phase_diagram as atl_pd
import atlas.core.structure as atl_struct
from atlas.core.code_utils import custom_print, get_config_path

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
CONFIG_EXAMPLE = {'API_KEY': 'XXXXX...'}


class AdsorbateAdder:
    """Class to add adsorbates to surfaces."""

    def __init__(self, cutoff=1.5, coverage=1.0):
        """
        Initialize adsorbate adder with cutoff and coverage.

        Parameters
        ----------
        cutoff : float
            Distance threshold to determine if a site is exposed.
        coverage : float
            Fraction of sites to be used for adsorption.
        """
        self.cutoff = cutoff
        self.coverage = coverage

    def is_exposed(self, pos, all_positions, z_pos, z_max):
        """Check if an atom is exposed."""
        # Mark atom as exposed if it is among the highest z-axis atoms
        if abs(z_pos - z_max) < 0.1:
            return True

        # Standard exposure check
        for other_pos in all_positions:
            if (
                other_pos[2] > z_pos + 0.1
                and np.linalg.norm(other_pos[:2] - pos[:2]) < self.cutoff
            ):
                return False
        return True


def clear_previous_print():
    print(LINE_UP, end=LINE_CLEAR)


def gather_secrets():
    """
    Gather Materials project API key from file/env var.

    The API key can be gathered from a secrets.json file that can
    be placed in the config directory or in the current working
    directory. If the file is not found, the function will check
    for an environment variable named 'MP_API_KEY'.

    Notes
    -----
    The json file should have the following structure:

    >>> {
    >>>     "API_KEY": "XXXXXX"
    >>> }


    Returns
    -------
    dict
        object containing the api key
    """
    config_path = get_config_path() / 'atl'

    if pathlib.Path('secrets.json').exists():
        with open('secrets.json') as f:
            secrets = js.load(f)

    elif pathlib.Path(config_path, 'secrets.json').exists():
        path = pathlib.Path(config_path, 'secrets.json')
        with open(path) as f:
            secrets = js.load(f)
    elif os.environ.get('MP_API_KEY'):
        secrets = {'API_KEY': os.environ.get('MP_API_KEY')}

    else:
        print()
        custom_print(
            '[bold red blink]Materials Project secrets missing: '
            "'secrets.json' not found![/]\n"
            "[bold red]Please, either run 'atl_init_setup', set the 'MP_API_KEY'"
            " environment variable,\nor add a 'secrets.json' file in the"
            f" following directory: '{config_path}',\nwith the following format:[/]\n"
            f'{CONFIG_EXAMPLE}',
            'error',
        )
        secrets = None

    return secrets


# TODO: Update or remove
def check_incorrect_ratios(df, curr_phase_diag):
    for _id, row in df.iterrows():
        if not row.base and not row.material_name.endswith('_symm'):
            strct = row.structure.get_sorted_structure()
            name = row.material_name
            phase = curr_phase_diag.get_phase(row.phase)
            offset = phase.offset
            tot_atoms = len(strct.species)
            one_at_perc = 1 / tot_atoms

            tot_cu = strct.species.count(Species('Cu')) + strct.species.count(
                Element('Cu')
            )
            tot_zn = strct.species.count(Species('Zn')) + strct.species.count(
                Element('Zn')
            )

            # Checking the total atom number
            if tot_cu + tot_zn != tot_atoms:
                raise ValueError(
                    'Total count does not match.'
                    f' tot_cu: {tot_cu}, tot_zn: {tot_zn}, total: {tot_atoms}.'
                    f' Species: {set(strct.species)}'
                )

            perc = round(tot_zn / tot_atoms, 2)

            offset_min = round(phase.base_elem_comp_min - offset, 2)
            if offset_min < 0:
                offset_min = 0

            offset_max = round(phase.base_elem_comp_max + offset, 2)
            if offset_max > 1:
                offset_max = 1

            # Checking if the current structure is between the phase ratio
            # percentages.
            if not (offset_min - 0.1 <= perc <= offset_max + 0.1):
                # If the structure could be fixed by adding or removing an atom
                if (offset_min <= perc + one_at_perc <= offset_max) or (
                    offset_min <= perc - one_at_perc <= offset_max
                ):
                    custom_print(
                        (
                            f'{name}: {perc:.2f} Zn outside of ({offset_min:.2f} -'
                            f' {offset_max:.2f}) range'
                        ),
                        'error',
                    )
                else:
                    pass


def _display_indb_dataframe(structures, data=None):
    """
    Display all of the given structures using the ase gui.

    Parameters
    ----------
    structures : list
        List of structures
    data : dict, optional
        Dictionary containing additional data, by default None
    """
    atoms_obj_list = []
    for structure in structures:
        if not isinstance(structure, Atoms):
            struct_ase = pmg_ase.AseAtomsAdaptor().get_atoms(structure)
        else:
            struct_ase = structure

        atoms_obj_list.append(struct_ase)

    visualize.view(atoms_obj_list, data=data)


def display_dataframe_ase(dataframe):
    """
    Display the structures in the given dataframe using the ase gui.

    Wrapper function for `_display_indb_dataframe.`
    """
    structures = dataframe.structure

    data_dict = {'filename': dataframe.material_name}
    _display_indb_dataframe(structures, data=data_dict)


def display_struct_list_ase(struct_list):
    """
    Display the structures in a list using the ase gui.

    Wrapper function for `_display_indb_dataframe.`
    """
    new_struct_list = []
    for strc in struct_list:
        if isinstance(strc, atl_struct.Structure):
            new_struct_list.append(strc.structure)
        else:
            new_struct_list.append(strc)

    data_dict = {}
    _display_indb_dataframe(new_struct_list, data=data_dict)


def similarity_check_list(
    db_obj, replaced_structures, r_cut=None, n_max=None, l_max=None, save_in_db=True
):
    custom_print(
        f'Checking replacements for {len(replaced_structures)} structures.', 'debug'
    )

    # Checking for similarity after replacement
    soap_kwargs = {'l_max': 2, 'n_max': 2}
    if r_cut is not None:
        soap_kwargs['r_cut'] = r_cut
    if n_max is not None:
        soap_kwargs['n_max'] = n_max
    if l_max is not None:
        soap_kwargs['l_max'] = l_max
    uuid_list = _check_repeat_struct_list(
        replaced_structures,
        alloy_set=db_obj.phase_diagram.alloy_set,
        **soap_kwargs,
    )
    print('uuid_list: ', len(uuid_list))

    # Deleting equivalent structures
    replaced_structures = _del_structure_list_by_uuid(replaced_structures, uuid_list)
    custom_print(
        f'{len(replaced_structures)} structures after duplicate check', 'debug'
    )

    if save_in_db:
        custom_print('Saving to db...', 'debug')
        for _idx, cluster in enumerate(replaced_structures):
            db_obj._save_row(structure=cluster)

    return replaced_structures


def gauss_perturb(structure: Structure, center: float = 0.04):
    ATL_STRUCT_TYPES = (
        atl_struct.Structure,
        atl_struct.Bulk,
        atl_struct.Cluster,
        atl_struct.Surface,
    )
    if isinstance(structure, ATL_STRUCT_TYPES):
        structure_obj = structure
        structure = structure.structure

    new_structure = structure.copy()
    new_structure.perturb(distance=center * 2, min_distance=center / 2)

    if isinstance(structure, ATL_STRUCT_TYPES):
        structure_obj.structure = new_structure
        new_structure = structure_obj

    return new_structure


def _check_repeat_struct_list(structure_list, alloy_set, r_cut=6, n_max=8, l_max=6):
    print('r_cut: ', r_cut)

    species_list = [el.Z for el in alloy_set]

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species_list,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sparse=False,
    )

    soap_structs = []
    pymg_structure_list = [struct.structure for struct in structure_list]
    uuid_list = [struct.unique_id for struct in structure_list]

    for pym_struct in pymg_structure_list:
        # Converting to ase structure
        ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

        # Create soap descriptors for current system and storing it
        struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=True)
        soap_structs.append(struct_soap)

    # Calculating similarity with an average kernel and a gaussan metric. The
    # result will be a full similarity matrix.
    kernel = AverageKernel(metric='rbf', gamma=1)
    simi_matrix = kernel.create(soap_structs)

    # For every structure, check if it is repeated more than once (itself)
    repeat_struct_uuid = []
    i_max, j_max = simi_matrix.shape

    # for struct_idx, row in enumerate(simi_matrix):
    for struct_idx, _struct in enumerate(structure_list):
        check_col = simi_matrix[struct_idx, struct_idx + 1 : j_max]

        # Get position in current row

        # Getting repeats
        row_repeats = np.isclose(check_col, 1)
        n_repeats = np.count_nonzero(row_repeats)

        # If structure is repeated
        if n_repeats > 0:
            len_diff = j_max - len(row_repeats)
            # For every structure, repeated or not
            for repeat_idx, repeat in enumerate(row_repeats):
                # If a structure is repeated
                if repeat:
                    # Compute the postion on the entire array column/row
                    list_idx = repeat_idx + len_diff
                    # Get the correspondng structure and add it to the list
                    repeat_struct_uuid.append(uuid_list[list_idx])

    repeat_struct_uuid = list(set(repeat_struct_uuid))

    struct_size = len(pymg_structure_list[0].species)
    custom_print(
        (
            f'Duplicate check for size {struct_size} - Total selected structures:'
            f'  {len(structure_list)}'
            f', equivalent: {len(repeat_struct_uuid)}'
            f' ({(len(repeat_struct_uuid) / len(structure_list)) * 100:.2f}%)'
        ),
        'debug',
    )
    return repeat_struct_uuid


def _del_structure_list_by_uuid(structure_list, dupl_uuid_list):
    all_uuid_set = set([stru.unique_id for stru in structure_list])
    dupl_uuid_set = set(dupl_uuid_list)
    unique_struct_set = all_uuid_set.difference(dupl_uuid_set)

    unique_structure_list = []
    for uniq_struc_uuid in unique_struct_set:
        for structure in structure_list:
            if structure.unique_id == uniq_struc_uuid:
                unique_structure_list.append(structure)

    return unique_structure_list


def apply_replacement_no_db(
    structure: Structure,
    phase,
    n_atoms: int,
    replace_elem: str | Element,
    rng=None,
):
    ATL_STRUCT_TYPES = (
        atl_struct.Structure,
        atl_struct.Bulk,
        atl_struct.Cluster,
        atl_struct.Surface,
    )

    if not rng:
        rng = np.random.default_rng()

    is_ase = False

    if isinstance(structure, ATL_STRUCT_TYPES):
        structure = structure.structure

    # Converting ase.Atoms to pymatgen structures
    elif isinstance(structure, Atoms):
        is_ase = True
        structure = AseAtomsAdaptor.get_structure(structure)

    structure_len = len(structure.species)

    # We assume that if the n_atoms is a fractional number, it must
    # represent the ratio of atoms in the structure, so we convert
    # that to a number of atoms.
    if isinstance(n_atoms, float) and n_atoms < 1:
        n_atoms = int(n_atoms * structure_len)

    # If no replacements are going to be made, this is probably due to
    # a low percentage being rounded to 0, thus we attempt to make at
    # least one replacement.
    if n_atoms == 0:
        n_atoms = 1
    other_atom_change = n_atoms

    # Choosing which species of the structure to change with the other atom.
    other_elem_choices = rng.choice(
        a=structure_len,
        size=abs(int(other_atom_change)),
        replace=False,
        shuffle=True,
    )

    # Creating a new pymatgen structure using the base one as a template
    new_structure = structure.copy(sanitize=True)
    site_props_before = structure.site_properties

    # Replacing atoms in the structures
    for ind in other_elem_choices:
        new_structure.replace(ind, replace_elem)

    # TODO: Instead of this, create a new structure
    # Copying site properties
    new_structure = new_structure.copy(sanitize=True, site_properties=site_props_before)

    if is_ase:
        new_structure = AseAtomsAdaptor.get_atoms(new_structure)

    return new_structure


def apply_central_atom_octahedral(
    db_obj: 'atl_indb.InitialDatabase',
    filter_phase_list: list[str],
    filter_struct_types: list[str],
    central_element: str | Element,
    num_repeats: int,
    max_perturbation_ang: float,
    limit_num_structures: int,
    seed: int,
):
    # Setting up the RNG
    rng = np.random.default_rng(seed=seed)

    # Filter phase and structure types
    filtered_df, _, _ = apply_filters_db(
        db_obj, phase=filter_phase_list, filters=filter_struct_types
    )

    modified_structs = []
    ids_to_remove = []
    # Iterate over all filtered structures
    for _, row in filtered_df.iterrows():
        for repeat_idx in range(num_repeats):
            # Get mdb Structure
            atl_struct_row = atl_struct.Structure().from_db_row(
                row=row, columns=filtered_df.columns
            )
            new_structure = atl_struct_row.structure.copy()

            # Identify indices of the central element
            if isinstance(central_element, Element):
                central_element = str(central_element)
            central_element_indices = new_structure.indices_from_symbol(central_element)

            # Perurbate the identified central atoms
            perturb_vec = rng.normal(loc=0, scale=max_perturbation_ang, size=(1, 3))[0]
            for site in central_element_indices:
                new_structure.translate_sites(indices=site, vector=perturb_vec)

            # Update structure attributes (targeted_modification, uuid?) and
            # add structure to db
            atl_struct_row.structure = new_structure
            atl_struct_row.targeted_modification = 'central_atom_perturbation'
            atl_struct_row.unique_id = str(uuid.uuid4())
            atl_struct_row.base = row.base
            atl_struct_row.material_name = (
                f'{atl_struct_row.material_name}_perturb_central_{repeat_idx}'
            )
            modified_structs.append(atl_struct_row)

        ids_to_remove.append(row.unique_id)

    # Limit the number of structures
    if limit_num_structures:
        limit_num_structures = min(limit_num_structures, len(modified_structs))

        modified_structs = rng.choice(
            modified_structs, limit_num_structures, replace=False
        )

    # Use uuid to remove original structures from the database
    for curr_uuid in ids_to_remove:
        db_obj.df = db_obj.df[db_obj.df['unique_id'] != str(curr_uuid)]

    # Add modified structures to the database
    for struc in modified_structs:
        struc.save_to_db(db_obj=db_obj)

    return db_obj


def apply_filters_db(
    db_obj,
    filters: list[str],
    phase: atl_pd.Phase | str | list = None,
    filter_mode: str = 'keep',
):
    """
    Apply filters based on database columns.

    This filtering function applies several filters using an OR logic, and allows to
    keep or remove the filtered structures.

    Parameters
    ----------
    db_obj : InitialDatabase
        Structure database to filter
    filters : list[str]
        Series of column names to use as filters
    phase : atl_pd.Phase | str | list, optional
        Series of phases to use as filters, by default None
    filter_mode : str, optional
        Either 'keep' or 'remove', by default 'keep'

    Returns
    -------
    pd.DataFrame
        DataFrame with/without the filtered structures.
    """
    filtered_df = db_obj

    if isinstance(db_obj, atl_indb.InitialDatabase):
        filtered_df = db_obj.df.copy()

    remaining_df = filtered_df

    custom_print(f'Applying filters: {filters}.', 'debug')

    # Applying filters. Filter lists are column names, that select all rows
    # that have a True value in that column, applied with an OR logic.
    if filters:
        if isinstance(filters, str):
            filters = [filters]
        appl_filter_db_list = []
        for filt in filters:
            appl_filt = filtered_df.loc[filtered_df[filt]]
            appl_filter_db_list.append(appl_filt)
            custom_print(
                f"Applied filter: '{filt}' - {len(appl_filt)} structs selected.",
                'debug',
            )

        filtered_df = pd.concat(appl_filter_db_list, axis=0)

    # Getting which phases to check from the user.
    phase_list = []
    if phase is not None:
        if isinstance(phase, (list, np.ndarray)):
            for curr_phase in phase:
                if isinstance(curr_phase, str):
                    curr_phase = db_obj.phase_diagram.get_phase(curr_phase)

                if curr_phase:
                    phase_list.append(curr_phase.name)

        else:
            if isinstance(phase, str):
                phase = db_obj.phase_diagram.get_phase(phase)
            custom_print(f'Using phase: {phase.name}.', 'debug')
            phase_list.append(phase.name)

        custom_print(f'phase_list: {phase_list}.', 'debug')

    # If no phase is given, getting the unique phases in the dataframe
    else:
        phase_list = filtered_df.phase.unique()
        custom_print(
            (f'No phase given. Checking on all phases: {phase_list}.'),
            'debug',
        )

    # Getting the current phase structures
    # In order to access a method from an object saved in a df, we need to do a
    # check like the following:
    idxs_list = []
    for idx, val in enumerate(filtered_df['phase']):
        name = val if isinstance(val, str) else val.name
        if name in phase_list:
            idxs_list.append(idx)

    filtered_df = filtered_df.iloc[idxs_list]

    # Getting the remaining structures after selecting the phase
    remaining_df = remaining_df.loc[remaining_df.index.difference(filtered_df.index)]

    custom_print(
        f'Number of filtered structures: {filtered_df.shape[0]}',
        'debug',
    )
    custom_print(
        f'Number of remaining unfiltered structures: {remaining_df.shape[0]}',
        'debug',
    )

    if filter_mode == 'keep':
        return filtered_df, remaining_df, phase_list
    if filter_mode == 'remove':
        return remaining_df, filtered_df, phase_list


def apply_replacement(
    structure: Structure,
    phase,
    n_target_at: int | float,
    phase_diagram: atl_pd.PhaseDiagram,
    rng=None,
):
    if not rng:
        rng = np.random.default_rng()

    if isinstance(
        structure,
        (
            atl_struct.Structure,
            atl_struct.Surface,
            atl_struct.Bulk,
        ),
    ):
        structure = structure.structure

    structure_len = len(structure.species)
    curr_comp = structure.composition

    # We assume that if the n_atoms is a fractional number, it must
    # represent the ratio of atoms in the structure, so we convert
    # that to a number of atoms.
    if isinstance(n_target_at, float) and n_target_at < 1:
        n_target_at = int(n_target_at * structure_len)

    # If no replacements are going to be made, this is probably due to
    # a low percentage being rounded to 0, thus we attempt to make at
    # least one replacement.
    if n_target_at == 0:
        n_target_at = 1

    curr_n_base_atoms = int(curr_comp[phase.base_elem])
    replacement_type = 'add' if n_target_at > curr_n_base_atoms else 'sub'

    # Getting current structure composition information
    # The current procedure assumes that all of the atom species in the structure
    # will have been replaced beforehand with the base atom,
    # although this results in more randomness.
    base_elem = phase.base_elem
    if len(phase_diagram.alloy_set) > 1:
        (other_elem,) = phase_diagram.alloy_set - {base_elem}
    else:
        other_elem = list(phase_diagram.alloy_set)[0]

    # If the structure only has one type of Element, and that is not the base
    # element, this changes with what to replace.
    # if not curr_comp.as_dict().get(base_elem.symbol):
    #     base_elem = str(phase.phase_diagram.element_list[0])
    #     if len(phase_diagram.alloy_set) > 1:
    #         (other_elem,) = phase_diagram.alloy_set - {base_elem}
    #     else:
    #         other_elem = list(phase_diagram.alloy_set)[0]

    # Adding base atoms to match the target percentage
    if replacement_type == 'add':
        n_at_diff = n_target_at - curr_n_base_atoms
        spec_to_replace = Element(other_elem)
        replacing_elem = Element(base_elem)
    # Removing base atoms to match the target percentage
    else:
        n_at_diff = curr_n_base_atoms - n_target_at
        spec_to_replace = Element(base_elem)
        replacing_elem = Element(other_elem)

    # Get atoms available to replace in the structure
    if isinstance(spec_to_replace, Element):
        repl_sites = structure.indices_from_symbol(spec_to_replace.symbol)
    else:
        repl_sites = structure.indices_from_symbol(spec_to_replace)

    try:
        # Randomly selecting indices to replace out of the available positions.
        replace_elem_choices = rng.choice(
            a=repl_sites,
            size=abs(int(n_at_diff)),
            replace=False,
            shuffle=True,
        )
    except ValueError:
        custom_print(
            (
                f"No replaceable sites for composition: '{curr_comp}'."
                "Add one of the formula's elements to the current phase"
                " 'replacements.element_list'."
            ),
            'error',
        )
        sys.exit(1)

    if isinstance(structure, (atl_struct.Surface, Slab)):
        new_structure = structure.get_sorted_structure()
    else:
        new_structure = structure.copy(sanitize=True)
    site_props_before = structure.site_properties

    # Replacing atoms in the structures
    for ind in replace_elem_choices:
        new_structure = new_structure.replace(ind, replacing_elem)

    # Copying site properties
    if isinstance(structure, (atl_struct.Surface, Slab)):
        new_structure = new_structure.get_sorted_structure()
    else:
        new_structure = new_structure.copy(
            sanitize=True, site_properties=site_props_before
        )

    return new_structure


def fit_replacements_phase(
    phase,
    structure,
    subst_base_elem_perc,
):
    if isinstance(structure, atl_struct.Structure):
        structure = structure.structure

    curr_comp = structure.composition
    base_elem = phase.base_elem
    # tot_base_at_struct = curr_comp[base_elem]
    structure_len = len(structure.species)
    offset_min = phase.base_elem_comp_min - phase.offset
    offset_max = phase.base_elem_comp_max + phase.offset

    n_at_replacement_upd = []
    for _, curr_perc in enumerate(subst_base_elem_perc):
        inPhase = phase.perc_in_phase(curr_perc)

        single_at_perc = 1 / structure_len
        perc_range = offset_max - offset_min

        # Skip this offset if changing one atom always results
        # in going over the maximum or minimum.
        if single_at_perc >= perc_range:
            inPhase = True

        while not inPhase:
            perc = curr_comp.get_atomic_fraction(base_elem)
            # perc = (tot_base_at_struct + abs(curr_perc)) / structure_len

            if perc >= offset_max:
                curr_perc -= single_at_perc
            elif perc <= offset_min:
                curr_perc += single_at_perc
            else:
                inPhase = phase.perc_in_phase(curr_perc)

        new_n_at = int(round(curr_perc * structure_len, 0))
        n_at_replacement_upd.append(new_n_at)

    return n_at_replacement_upd


def gen_base_elem_perc(phase, num_struct):
    # Computing base_elem percentages using offset
    if phase.offset and phase.offset > 0:
        # Getting offset. If not found set to 0.
        offset = phase.offset

        # Randomly generating base_elem percentages for the new structures
        max_base_elem = min((phase.base_elem_comp_max + offset), 1)
        min_base_elem = max(phase.base_elem_comp_min - offset, 0)

        subst_base_elem_perc = (min_base_elem - max_base_elem) * np.random.ranf(
            size=num_struct
        ) + max_base_elem

    # Computing base element percentages without offset.
    else:
        max_base_elem = phase.base_elem_comp_min
        min_base_elem = phase.base_elem_comp_max
        subst_base_elem_perc = (min_base_elem - max_base_elem) * np.random.ranf(
            size=num_struct
        ) + max_base_elem

    return subst_base_elem_perc


def create_symmetrical_prototype(
    structure: Structure,
    phase_diagram: atl_pd.PhaseDiagram,
    phase: atl_pd.Phase,
    structure_obj: 'atl_struct.Structure',
):
    phase = structure_obj.phase

    if isinstance(phase, str):
        phase = phase_diagram.get_phase(phase)

    # curr_phase_atom = self.phase_diagram.get_phase(phase).base_elem
    # base_atom_set = list(self.phase_diagram.alloy_set - {curr_phase_atom})

    if isinstance(structure, (atl_struct.Surface, Slab)):
        new_structure = structure.get_sorted_structure()
    else:
        new_structure = structure.copy(sanitize=True)

    # Replacing atoms in the structures
    ind = 2
    sum_ind = 0
    sum_list = (2, 1, 2, 3)

    while ind < structure.num_sites:
        # new_structure.replace(ind - 1, Species(base_atom_set[0]))
        new_structure.replace(ind - 1, Element(phase.base_elem))
        ind = ind + sum_list[sum_ind]

        if sum_ind == 3:
            sum_ind = 0
        else:
            sum_ind += 1

    material_id_prefix = str(structure_obj.material_id)

    # Generating the symmetrized structure
    new_struct_symm = atl_struct.Structure(
        material_name=f'{material_id_prefix}_{phase.name}_symm',
        material_id=material_id_prefix,
        structure=new_structure,
        temperature=structure_obj.temperature,
        bulk=structure_obj.bulk,
        surface=structure_obj.surface,
        surface_miller=structure_obj.surface_miller,
        cluster=structure_obj.cluster,
        perturb=structure_obj.perturb,
        base=structure_obj.base,
        calc_performed=structure_obj.calc_performed,
        supercell=structure_obj.supercell,
        phase=phase.name,
    )

    if structure_obj.bulk:
        final_struct = new_struct_symm.to_bulk()
    elif structure_obj.surface:
        final_struct = new_struct_symm.to_surface()
    elif structure_obj.cluster:
        final_struct = new_struct_symm.to_cluster()
    else:
        raise NotImplementedError(
            'Symmetrical prototype not implemented for'
            'implemented for current structure type.'
        )

    # self.df = final_struct.save_to_db(self.df)

    return final_struct


def find_supercell_indices(
    structure,
    get_different_supercells,
    min_atoms,
    max_atoms,
    initial_supercell_size=5,
    verbose=True,
):
    # Initial supercell size
    idx = initial_supercell_size

    # Copying structure
    try:
        new_structure = structure.copy(sanitize=True)
    except TypeError:
        new_structure = structure.copy()

    # Setting different supercell geometry for slabs and bulks.
    supercell_vec = [idx, idx, 1] if isinstance(structure, Slab) else [idx, idx, idx]

    new_structure.make_supercell(supercell_vec, to_unit_cell=False)

    # Number of atoms of the supercell
    struct_size = len(new_structure.species)
    while (struct_size > max_atoms or struct_size < min_atoms) and supercell_vec != [
        1,
        1,
        1,
    ]:
        try:
            new_structure = structure.copy(sanitize=True)
        except TypeError:
            new_structure = structure.copy()

        if isinstance(structure, Slab):
            supercell_vec = [idx, idx, 1]
        else:
            supercell_vec = [idx, idx, idx]

        new_structure.make_supercell(supercell_vec, to_unit_cell=False)
        struct_size = len(new_structure.species)
        idx -= 1

    structure_list = []
    idx_list = []
    supercell_vec_list = []
    structure_list.append(new_structure)
    idx_list.append(idx)
    supercell_vec_list.append(supercell_vec)

    if verbose:
        custom_print(
            f'Supercell generated {supercell_vec}'
            f' - total atoms: {len(new_structure.species)}',
            'debug',
        )

    if get_different_supercells:
        # Generating all possible combinations of supercells up to a given size
        possible_supercells = it.combinations_with_replacement(
            range(2, initial_supercell_size + 1), r=3
        )

        for idx_smaller in possible_supercells:
            try:
                new_structure = structure.copy(sanitize=True)
            except TypeError:
                new_structure = structure.copy()

            # Slabs must not be repeated on z axis
            if isinstance(structure, Slab):
                supercell_vec = [idx_smaller[0], idx_smaller[1], 1]

                # Removing slabs if already on the list
                if supercell_vec in supercell_vec_list:
                    continue

            # Bulks and clusters can be repeated on all axis
            else:
                supercell_vec = idx_smaller

            # Creating and adding the supercell if it is within
            # the desired size range
            if supercell_vec != [1, 1, 1]:
                new_structure.make_supercell(supercell_vec, to_unit_cell=False)
                struct_size = len(new_structure.species)
                if struct_size < max_atoms and struct_size > min_atoms:
                    structure_list.append(new_structure)
                    idx_list.append(idx_smaller)
                    supercell_vec_list.append(supercell_vec)

                    if verbose:
                        custom_print(
                            (
                                f'Supercell generated (diff.) {supercell_vec} '
                                f'- total atoms: {struct_size}'
                            ),
                            'debug',
                        )

    return structure_list, idx_list, supercell_vec_list


def _apply_perturbation_atl_struct(center, row, per_idx):
    # Applying perturbation to the structure
    new_struct_perturb = gauss_perturb(center=center, structure=row.structure)

    # Getting current row information
    row_kwargs_dict = {**row}
    for func_name in [
        'from_db_row',
        'material_name',
        'perturb',
        'from_bulk',
        'structure',
        'vacancy',
        'base',
        'from_surface',
        'targeted_modification',
        'from_cluster',
        'deformation',
        'unique_id',
        'to_ase_atoms',
        'from_ase_atoms',
    ]:
        if func_name in row_kwargs_dict:
            row_kwargs_dict.pop(func_name)

    # Creating perturbed cluster object
    mat_str = f'{row.unique_id}_{row.material_id}_perturb_gauss_{per_idx + 1}'
    perturb_struct = atl_struct.Structure(
        material_name=mat_str,
        structure=new_struct_perturb,
        perturb=True,
        deformation=False,
        vacancy=False,
        targeted_modification=False,
        base=False,
        **row_kwargs_dict,
    )
    if row.bulk:
        perturb_struct.to_bulk()
    elif row.surface:
        perturb_struct.to_surface()
    elif row.cluster:
        perturb_struct.to_cluster()
    return perturb_struct


def apply_gauss_perturb_db(
    repeat: int,
    db_obj: 'atl_indb.InitialDatabase',
    filters: list,
    phase: atl_pd.Phase,
    center: float = 0.04,
    limit_num_structures: int = None,
):
    perturbed_structs = []

    if not isinstance(db_obj, atl_indb.InitialDatabase):
        raise TypeError(
            f"'{apply_gauss_perturb_db.__name__}' expects a ATLAS "
            f'database object, not a {type(db_obj)}.'
        )

    # Filtering structures to perturb
    filtered_df, _, _ = apply_filters_db(db_obj, filters, phase)

    # Iterating over all filtered database rows to get the unperturbed surfaces
    custom_print(
        f'Perturbation will be applied to: {filtered_df.shape[0]} structures.', 'debug'
    )
    for _, row in filtered_df.iterrows():
        if row.perturb:
            continue
        for per_idx in range(repeat):
            clust_obj = _apply_perturbation_atl_struct(center, row, per_idx)

            perturbed_structs.append(clust_obj)

    custom_print(f'Total structs perturbed: {len(perturbed_structs)}', 'debug')

    if limit_num_structures:
        custom_print(
            f'Limiting number of perturbations to  {limit_num_structures}', 'debug'
        )

        limit_num_structures = np.min([limit_num_structures, len(perturbed_structs)])

        perturbed_structs = np.random.choice(
            perturbed_structs, limit_num_structures, replace=False
        )

    # Saving in database
    custom_print('Saving perturbed surfaces in dataframe...', 'debug')
    for surface in perturbed_structs:
        db_obj._save_row(structure=surface)
    custom_print(f'Dataframe shape after saving: {db_obj.df.shape}.', 'debug')

    return perturbed_structs


def limit_num_structures_phase(
    db_obj: 'atl_indb.InitialDatabase',
    phase: 'atl_pd.Phase',
    num_limit: int,
    rng_seed: int,
    stratify_by_size: bool = False,
):
    # Instantiating RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Getting the current phase structures (df_filtered)
    df_filtered = db_obj.df.loc[db_obj.df['phase'] == phase.name]

    # Gathering the base structures included in the current phase
    df_filt_base = df_filtered.loc[df_filtered.base]
    df_filt_non_base = df_filtered.loc[~df_filtered.base]

    # Getting the remaining structures after selecting
    # the phase (df_remaining)
    df_remaining = db_obj.df.loc[db_obj.df['phase'] != phase.name]

    # Reserve slots for base structures, sample the rest from non-base
    n_base = df_filt_base.shape[0]
    n_sample = max(0, min(num_limit, df_filtered.shape[0]) - n_base)
    n_sample = min(n_sample, df_filt_non_base.shape[0])

    if n_sample > 0:
        if stratify_by_size:
            df_filt_sampl = _stratified_sample_by_size(
                df_filt_non_base, n_sample, rng
            )
        else:
            sampl_idx = rng.choice(df_filt_non_base.shape[0], n_sample, replace=False)
            df_filt_sampl = df_filt_non_base.iloc[sampl_idx]
    else:
        df_filt_sampl = df_filt_non_base.iloc[:0]

    # Adding df_filt_sample to the df_remaining
    df_remaining = pd.concat([df_remaining, df_filt_sampl, df_filt_base], axis=0)
    db_obj.df = df_remaining
    return db_obj


def _stratified_sample_by_size(df, n_sample, rng):
    """Sample structures proportionally across atom-count bins."""
    atom_counts = df.apply(lambda row: len(row.structure.species), axis=1)

    n_unique = atom_counts.nunique()
    n_bins = min(n_unique, max(1, n_sample // 5))

    bin_labels = pd.cut(atom_counts, bins=n_bins, labels=False)
    groups = list(df.groupby(bin_labels))
    n_groups = len(groups)

    base_quota = n_sample // n_groups
    remainder = n_sample % n_groups

    sampled_indices = []
    for bin_idx, (_, group) in enumerate(groups):
        quota = base_quota + (1 if bin_idx < remainder else 0)
        quota = min(quota, len(group))
        if quota > 0:
            chosen = rng.choice(len(group), quota, replace=False)
            sampled_indices.extend(group.index[chosen])

    # Top up if under-filled bins left us short
    if len(sampled_indices) < n_sample:
        remaining_pool = df.index.difference(sampled_indices)
        shortfall = min(n_sample - len(sampled_indices), len(remaining_pool))
        if shortfall > 0:
            extra = rng.choice(remaining_pool.values, shortfall, replace=False)
            sampled_indices.extend(extra)

    return df.loc[sampled_indices]


def add_adsorbates(
    repeat: int,
    db_obj: 'atl_indb.InitialDatabase',
    filters: list,
    phase: atl_pd.Phase,
    adsorbate_species: list[str],
    limit_num_structures: int = None,
):
    from rich.progress import track

    # Create temporary file to store the trajectory using tmpfile
    tmp_file = tempfile.NamedTemporaryFile(suffix='.traj').name  # noqa

    adsorb_structs = []

    if not isinstance(db_obj, atl_indb.InitialDatabase):
        raise TypeError(
            f"'{apply_gauss_perturb_db.__name__}' expects a ATLAS "
            f'database object, not a {type(db_obj)}.'
        )

    # Filtering structures to perturb
    filtered_df, _, _ = apply_filters_db(db_obj, filters, phase)

    # Iterating over all filtered database rows to get desired surfaces
    custom_print(
        f'Perturbation will be applied to: {filtered_df.shape[0]} structures.', 'debug'
    )

    target_structs = filtered_df.iterrows()

    if limit_num_structures:
        limit_num_structures = np.min(
            [limit_num_structures // repeat, filtered_df.shape[0] // repeat]
        )

        custom_print(
            f'Limiting number of structures to  {limit_num_structures}', 'info'
        )

        target_structs = np.random.choice(
            range(limit_num_structures), limit_num_structures, replace=False
        )

        target_structs = [filtered_df.iloc[struct] for struct in target_structs]

    # for _, row in filtered_df.iterrows():
    adsorb_structs_l = []
    for row in track(target_structs, description='Placing adsorbates...'):
        struct = row.structure
        if not isinstance(row.structure, Atoms):
            struct = AseAtomsAdaptor().get_atoms(row.structure)

        # TODO: test miller index. Get actual miller from row
        # miller = row.surface_miller

        # Getting the layers of the structure
        # layer_num = len(np.unique(get_layers(struct, miller, tolerance=0.3)[0]))

        # cust_crystal = crystal(
        #     symbols=struct,
        #     spacegroup=phase.spacegroup,
        #     cell=struct.cell,
        # )

        # ase_surf = surface(cust_crystal, miller, layers=1, vacuum=15)
        # print('ase_surf: ', ase_surf)

        # Getting the layers of the structure
        # _, layer_levels_sf = get_layers(ase_surf, miller, tolerance=0.1)
        # layer_num_sf = len(layer_levels_sf)

        # cust_surf = CustomSurface(struct, n_layers=4)
        # cust_surf = CustomSurface(struct)
        # print('cust_surf: ', dir(cust_surf))
        # print('struct: ', struct)

        # sas = SlabAdsorptionSites(
        #     struct,
        #     surface=cust_surf,
        #     composition_effect=True,
        #     #   ignore_sites=['4fold'],
        #     label_sites=True,
        # )
        # # print("sas: ", sas)

        # # Add more height
        # print('sas: ', sas)
        # heights = {k: v + 0.5 for k, v in site_heights.items()}

        # # TODO: Change surface to the correct one
        # # for spec in adsorbate_species:

        # # print("spec: ", spec)
        # print('repeat: ', repeat)
        # gen = RPG(
        #     images=struct,
        #     adsorbate_species=adsorbate_species,
        #     min_adsorbate_distance=1.5,
        #     surface=cust_surf,
        #     heights=heights,
        #     # species_forbidden_sites={'CHOO': ['ontop','bridge']},
        #     trajectory=tmp_file,
        #     adsorption_sites=sas,
        #     append_trajectory=True,
        # )
        # gen.run(num_gen=repeat, action="add")
        # atoms = ase_read(tmp_file)
        # adsorb_structs.append(atoms)
        # visualize(atoms)

        # Make this general
        adder = AdsorbateAdder(cutoff=1.5, coverage=1)
        ir_sites, o_sites = adder.find_surface_atoms(struct)
        print(f'Found {len(ir_sites)} exposed Ir atoms')
        print(f'Found {len(o_sites)} exposed O atoms')

        structures = adder.add_adsorbates(struct, ir_sites, o_sites)
        adsorb_structs.extend(structures)

    surfaces = [tupl[2] for tupl in adsorb_structs_l]
    # REMOVE
    visualize.view(surfaces)

    # Saving in database
    custom_print('Saving surfaces with adsorbates in dataframe...', 'debug')
    for surf in surfaces:
        db_obj._save_row(structure=surf)
    custom_print(f'Dataframe shape after saving: {db_obj.df.shape}.', 'debug')

    # Deleting temporary file
    # os.remove(tmp_file)

    return adsorb_structs


def fix_bottom_layers(structure: Structure, n_layers: int) -> Structure:
    """
    Fixes the bottom n layers of a pymatgen Structure by setting the selective
    dynamics to False for the atoms in those layers.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The structure to modify.
    n_layers : int
        The number of bottom layers to fix.

    Returns
    -------
    Structure
        A new Structure object with selective dynamics updated.
    """
    sorted_indices = sorted(range(len(structure)), key=lambda i: structure[i].z)
    bottom_indices = sorted_indices[:n_layers]

    # Set selective dynamics, fixing bottom layers
    selective_dynamics = [[True, True, True]] * len(structure)
    for idx in bottom_indices:
        selective_dynamics[idx] = [False, False, False]

    structure_with_constraints = structure.copy()
    structure_with_constraints.add_site_property(
        'selective_dynamics', selective_dynamics
    )

    print('structure_with_constraints: ', structure_with_constraints.site_properties)
    return structure_with_constraints
