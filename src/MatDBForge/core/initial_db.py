"""
This script generates a pandas dataframe containing a set of
base (unperturbed) structures and a certain number of structures
with an applied perturbation with respect to the temperature.
"""

import itertools as it
import lzma
import os
import pathlib
import pickle
import re
import warnings
from io import BytesIO, TextIOWrapper

import ase.io as aseio
import numpy as np
import pandas as pd
import pymatgen.io.vasp as vasp
import rich.align as rialg
import rich.console as ricns
import rich.live as riliv
import rich.progress as riprg
from aiida import load_profile, orm
from aiida_vasp.calcs.vasp import VaspCalculation
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from slugify import slugify

import MatDBForge as mdb
import MatDBForge.core.clusters as mdb_clust
import MatDBForge.core.exceptions as mdb_exc
import MatDBForge.core.phase_diagram as mdb_pd
import MatDBForge.core.structure as mdb_struct
import MatDBForge.core.surfaces as mdb_surf
from MatDBForge.core import utils as ut

# Filtering certain warnings
warnings.filterwarnings("ignore", category=vasp.outputs.UnconvergedVASPWarning)


class InitialDatabase:
    """
    Object that creates an initial database where structures will be
    stored. Materials are downloaded using the materials project API.
    For it to work, a 'secrets.json' file should be located in the
    same directory.

    Attributes
    ----------
    df : pd.Dataframe
        Dataframe containing the structures for the initial database.
    elements : list
        List containing the elements from the material.
    main_element : str | Element
        Reference element for the material. Will be used to represent the phase diagram.
    phases_dict : dict
        Dictionary representing each of the phases of the material.
        Uses the following format:
        ```
        phases_dict = {
            {
                "name": "alpha",
                "base_elem": "Zn",
                "cluster_elem": "Cu",
                "base_elem_comp_min": 0,
                "base_elem_comp_max": 0.3895,
                "prototype": "mp-30",
                "offset": 0.03,
            },
            {
                "name": "beta",
                ...
            },
        }
        ```
    use_offset : bool, optional
        Use an offset for the phase ratios to allow them to overlap, by default True.
    database_name : str
        Orientative name for the database. Will be used for saving it into a file.
    material_name : str
        Orientative name for the material. Will be used for the PhaseDiagram definition.
    max_num_atoms: int
        Maximum number of atoms present in any structure generated.
    secrets : dict
        Object containing secrets related to the materials project database.


    Notes
    -----
        The json file should have the following structure:

        {
            "API_KEY": "XXXXXX"
        }


    """

    # Boltzmann constant in J/(Da*K)
    kB = 8.314

    # Sourced from CODATA 2018
    Bohr2Ang = 0.5291772109030
    Ang2Bohr = 1 / Bohr2Ang
    Eh2eV = 27.211386245988
    eV2Eh = 1 / Eh2eV

    def __init__(
        self,
        elements: list,
        main_element: str,
        phases_dict_list: dict,
        database_name: str,
        material_name: str,
        max_num_atoms: int = 64,
        use_offset: bool = True,
    ) -> None:
        self.db_version = mdb.__version__

        # Creating phase diagram object
        self.db_phase_diagram: mdb_pd.BinaryPhaseDiagram = self._create_phase_diagram(
            phases_dict_list, material_name
        )

        # Saving elements into a set
        self.database_elements: set = set(elements)
        self.main_element = main_element
        self.material_name: str = material_name

        # Name of the database
        self.database_name = database_name

        # Setting the maximum number of atoms of any generated structure.
        self.max_num_atoms = max_num_atoms

        # Checking if a database is already found in the cwd
        check_flag = self._check_database()

        # Create the database if it does not exists
        # Load it if otherwise.
        if not check_flag:
            self.df = self._create_database()
        else:
            self.df = self._load_database()

        # Loading materials project API key from a json file
        self.secrets = ut.gather_secrets()

        # Using the offset
        if use_offset:
            ut.custom_print(
                "Using an offset for computing the phases concentrations.", "info"
            )
            self.use_offset = use_offset

    def __repr__(self):
        # Getting the class name
        class_name = self.__class__.__name__

        # Getting the amount of entries in the database
        count = len(self.df.count(axis=1))

        repr_string = (
            f"{class_name} named '{self.database_name}' containing {count} structures."
        )

        return repr_string

    # TODO: Implement this
    def _create_phase_diagram(
        self, phases_dict_list: list, material_name: str
    ) -> mdb_pd.BinaryPhaseDiagram:
        # Converting every phase dict into a Phase object
        final_phases_list = []
        for curr_phase in phases_dict_list:
            curr_phase: dict
            obj_Phase = mdb_pd.Phase(
                name=curr_phase["name"],
                base_elem=curr_phase["base_elem"],
                cluster_elem=curr_phase["cluster_elem"],
                base_elem_comp_min=curr_phase["base_elem_comp_min"],
                base_elem_comp_max=curr_phase["base_elem_comp_max"],
                prototype=curr_phase["prototype"],
                offset=curr_phase["offset"],
            )

            final_phases_list.append(obj_Phase)

        # TODO:
        # Create a size independent PhaseDiagram object
        # or
        # Use the length of self.database_elements to get which type of PD to use

        # Generating PhaseDiagam using the phases
        db_phase_diagram = mdb_pd.BinaryPhaseDiagram(material_name, *final_phases_list)

        return db_phase_diagram

    def _adapt_old_db(self, database_old):
        print("database_old: ", database_old.columns)
        columns_to_add = {
            "bulk": bool,
            "surface": bool,
            "cluster": bool,
            "calc_performed": bool,
            "replacement": bool,
        }

        for col in columns_to_add:
            database_old[col] = None

        return database_old

    def _load_database(self) -> pd.DataFrame:
        """
        Load a database from a pickle file on the cwd or a specific path.

        Returns
        -------
        pd.DataFrame
            Dataframe containing structure data for the initial database
        """
        db_path = pathlib.PurePath(self.database_name)
        ut.custom_print(f"Loading database: '{self.database_name}'", "debug")
        self.database_name = db_path.name.replace(db_path.suffix, "")

        if len(db_path.suffixes) == 0:
            suffix = ".xz"
        else:
            if db_path.suffixes[0] == ".pkl":
                suffix = ".pkl"
                ut.custom_print(
                    "Using outdated version of database. Adding missing columns.",
                    "debug",
                )
                database = pd.read_pickle(db_path)
                database = self._adapt_old_db(database)

                return database

            elif db_path.suffixes[0] == ".xz":
                suffix = ".xz"
                with lzma.open(db_path, "rb") as f:
                    database = pickle.load(f)

                    # Setting parameters from InitialDatabase
                    self.database_name = database.database_name
                    self.max_num_atoms = database.max_num_atoms
                    self.db_version = database.db_version

                    ut.custom_print(
                        f"Using database version {self.db_version}.", "debug"
                    )

                return database.df
            # else:
            #     err_msg = (
            #         "MatDBForge InitialDatabase object not found "
            #         "in the specified location."
            #     )
            #     raise FileNotFoundError(err_msg)

        ut.custom_print(f"Loaded '{self.database_name}{suffix}'", "info")
        ut.custom_print(f"Path: {db_path}", "debug")

    def _check_database(self) -> bool:
        """
        This method check if a database with the name 'self.database_name'
        exists in the current working directory or is a path to a existing
        database.

        Returns
        -------
        bool
            True if the database exists, False if not does not.
        """
        # Checking if dataframe already exists on the cwd.
        file_exists = False
        file_check = [file for file in os.listdir() if self.database_name in file]
        if len(file_check) > 0:
            file_exists = True

        # If the database name is a path
        name_as_path = pathlib.Path(self.database_name)

        if name_as_path.exists():
            file_exists = True

        ut.custom_print(f"Database found: {file_exists}.", "debug")

        return file_exists

    def _create_database(self) -> pd.DataFrame:
        """
        Create an empty  dataframe in order to be used in the class

        Returns
        -------
        pd.DataFrame
            Empty dataframe containing the necessary columns to be used in the main
            database.
        """
        # Creating a pandas dataframe to store the structures
        df = pd.DataFrame(
            columns=[
                "material_name",
                "material_id",
                "structure",
                "phase",
                "formula",
                "symmetry",
                "base",
                "perturb",
                "unique_id",
                "supercell",
                "surface",
                "bulk",
                "cluster",
                "temperature",
                "magnetic_properties",
                "calc_energy",
                "calc_energy_per_atom",
                "calc_energy_toten",
                "calc_performed",
                "calc_type",
                "calc_output",
            ]
        )

        df.attrs["db_version"] = self.db_version
        ut.custom_print(f"Created database '{self.database_name}'.", "done")

        return df

    def _find_supercell_indices(
        self,
        structure,
        get_different_supercells,
        max_atoms,
        initial_supercell_size=5,
        verbose=True,
    ):
        # Initial supercell size
        idx = initial_supercell_size

        # Copying structure
        new_structure = structure.copy(sanitize=True)

        # Setting different supercell geometry for slabs and bulks.
        if isinstance(structure, Slab):
            supercell_vec = [idx, idx, 1]
        else:
            supercell_vec = [idx, idx, idx]

        new_structure.make_supercell(supercell_vec, to_unit_cell=False)

        # Number of atoms of the supercell
        struct_size = len(new_structure.species)
        while struct_size > max_atoms and supercell_vec != [1, 1, 1]:
            new_structure = structure.copy(sanitize=True)
            idx -= 1

            if isinstance(structure, Slab):
                supercell_vec = [idx, idx, 1]
            else:
                supercell_vec = [idx, idx, idx]

            new_structure.make_supercell(supercell_vec, to_unit_cell=False)
            struct_size = len(new_structure.species)

        structure_list = []
        idx_list = []
        supercell_vec_list = []
        structure_list.append(new_structure)
        idx_list.append(idx)
        supercell_vec_list.append(supercell_vec)

        if verbose:
            ut.custom_print(
                f"Supercell generated - total atoms: {len(new_structure.species)}",
                "debug",
            )

        if get_different_supercells:
            for idx_smaller in range(idx - 1, 0, -1):
                new_structure = structure.copy(sanitize=True)

                if isinstance(structure, Slab):
                    supercell_vec = [idx_smaller, idx_smaller, 1]
                else:
                    supercell_vec = [idx_smaller, idx_smaller, idx_smaller]

                new_structure.make_supercell(supercell_vec, to_unit_cell=False)
                structure_list.append(new_structure)
                idx_list.append(idx_smaller)
                supercell_vec_list.append(supercell_vec)

                if verbose:
                    ut.custom_print(
                        (
                            "Supercell generated - total atoms:"
                            f"{len(new_structure.species)}"
                        ),
                        "debug",
                    )

        return structure_list, idx_list, supercell_vec_list

    def _check_repeat_struct(self, curr_phase, curr_struct: Structure):
        structure_list = self.df.loc[self.df.phase == curr_phase].structure.values

        # species_list = set([a.symbol for a in curr_struct.species])
        species_list = [el.Z for el in self.database_elements]

        r_cut = 6
        n_max = 8
        l_max = 6

        soap_structs = []

        soap = SOAP(
            species=species_list,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sparse=False,
        )

        for pym_struct in structure_list:
            ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

            # Create output for multiple system in parallel
            struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=False)
            curr_feat_sum = struct_soap.sum()
            soap_structs.append(curr_feat_sum)

        # TODO: Should this be here twice?
        curr_ase_struct = AseAtomsAdaptor().get_atoms(curr_struct)
        curr_struct_soap = soap.create(curr_ase_struct, n_jobs=-1, verbose=False)
        curr_soap_sum = curr_struct_soap.sum()

        total_soap_arr = np.array(soap_structs)

        comp_arr = np.isclose(curr_soap_sum, total_soap_arr, rtol=7.5e-04, atol=5e-05)

        if np.count_nonzero(comp_arr) > 0:
            ut.custom_print("duplicate found!!", "warn")
            return True

        else:
            return False

    def find_repeat_structures(
        self,
        delete=False,
        filters: list = None,
        phase: mdb_pd.Phase = None,
    ):
        # Filtering the dataframe
        # Filters allow to select certain subsets of structures
        # from the database.
        filtered_df = self.df
        # remaining_df = self.df

        filtered_df, remaining_df, phase_list = ut.apply_filters_db(
            db_obj=filtered_df, phase=phase, filters=filters
        )

        # Getting the species from the current phase diagram
        species = self.database_elements
        species_str_list = [spec.symbol for spec in species]

        # Setting SOAP related parameters
        r_cut = 4
        n_max = 4
        l_max = 4

        ut.custom_print(
            f"Setting up SOAP with: r_cut = {r_cut}, n_max = {n_max}, l_max = {l_max}",
            "debug",
        )

        # Setting up the SOAP descriptor
        soap = SOAP(
            species=species_str_list,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sparse=False,
        )

        # The uuids of the repeated structues will be stored.
        tot_duplicate_uuid_list = []

        # Computing SOAP for every phase
        for curr_phase in phase_list:
            # This list will contain the descriptors for every structure
            soap_structs = []

            # Getting the current structures
            structure_list = filtered_df.structure.values

            # Getting the names fo the current structures
            uuid_list = filtered_df[filtered_df.phase == curr_phase]["unique_id"].values

            # Getting the total structure count
            tot_structures = len(structure_list)

            # Counter for the total equivalent structure number
            tot_equival = 0

            for pym_struct in structure_list:
                # Converting to ase structure
                ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

                # Create soap descriptors for current system and storing it
                struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=True)
                soap_structs.append(struct_soap)

            # Calculating similarity with an average kernel and a gaussan metric. The
            # result will be a full similarity matrix.
            kernel = AverageKernel(metric="rbf", gamma=1)
            simi_matrix = kernel.create(soap_structs)

            # Checking every structure in the similarity matrix.
            # Similarity goes from 0 to 1. If a structure is very close to 1, it
            # will return a True in the mask. There will be always one matching
            # structure, as the algorithm compares the structure with itself.

            for struct_idx, row in enumerate(simi_matrix):
                row_n_repeat = np.count_nonzero(np.isclose(row, 1)) - 1

                # If the structure is repeated, get its name
                if row_n_repeat > 0:
                    tot_duplicate_uuid_list.append(uuid_list[struct_idx])
                    tot_equival += 1

            ut.custom_print(
                (
                    f"Phase '{curr_phase.name}' - Total selected structures:"
                    f" {tot_structures}, equivalent: {tot_equival}"
                    f" ({(tot_equival/tot_structures)*100:.2f}%)"
                ),
                "debug",
            )

        duplicate_structure_names = tot_duplicate_uuid_list

        # If the deletion flag is set, the function will delete the duplicate stuctures.
        if delete:
            ut.custom_print(
                f"{len(duplicate_structure_names)} structures marked for deletion.",
                "debug",
            )

            # Getting the dataframe entries that match the stored uuids
            mat_name_match_mask = filtered_df.isin(
                {"unique_id": duplicate_structure_names}
            )["unique_id"]
            duplicate_structures_df = filtered_df[mat_name_match_mask]

            # Dropping the matching entries
            filtered_df = filtered_df.drop(duplicate_structures_df.index)

            init_df_after_removal = pd.concat([remaining_df, filtered_df])

            self.df = init_df_after_removal

            ut.custom_print(
                f"Deleted {len(duplicate_structures_df)} structures.",
                "warn",
            )

            ut.custom_print(f"Dataframe shape after deleting: {self.df.shape}", "debug")

        else:
            ut.custom_print(
                (
                    f"{len(duplicate_structure_names)} repeated structures found. "
                    "Database untouched as 'delete' is set to False."
                ),
                "info",
            )

    def gather_base_structures(
        self, target_structures: list[str] = None, target_elements: list[str] = None
    ):
        if target_structures:
            # Checking which materials are already on the database
            missing_mat = set(target_structures) - set(self.df["material_id"].values)

            # Querying materials project database.
            with MPRester(self.secrets["API_KEY"]) as mpr:
                query_result = mpr.summary.search(material_ids=missing_mat)
                for material in query_result:
                    for phase in self.db_phase_diagram.phases:
                        if phase.prototype == material.material_id:
                            curr_phase = phase.name
                        else:
                            curr_phase = np.nan

                    curr_struct = mdb_struct.Bulk(
                        material_id=str(material.material_id),
                        structure=material.structure,
                        temperature=np.nan,
                        perturb=False,
                        formula=material.composition_reduced,
                        symmetry=material.get_space_group_info(),
                        base=True,
                        phase=curr_phase,
                        magnetic_properties=material.total_magnetization,
                        energy_per_atom=material.energy_per_atom,
                    )

                    self.df = curr_struct.save_to_db(self.df)

        elif target_elements:
            # Querying materials project database.
            with MPRester(self.secrets["API_KEY"]) as mpr:
                query_result = mpr.summary.search(
                    elements=target_elements, num_elements=len(target_elements)
                )

                for material in query_result:
                    for phase in self.db_phase_diagram.phases:
                        if phase.prototype == material.material_id:
                            curr_phase = phase.name
                        else:
                            curr_phase = np.nan

                    print("material: ", material.structure.formula)
                    curr_struct = mdb_struct.Bulk(
                        material_id=str(material.material_id),
                        structure=material.structure,
                        temperature=np.nan,
                        formula=material.composition_reduced,
                        symmetry=material.structure.get_space_group_info(),
                        base=True,
                        replacement=False,
                        perturb=False,
                        phase=curr_phase,
                        magnetic_properties=material.total_magnetization,
                        energy_per_atom=material.energy_per_atom,
                    )

                    self.df = curr_struct.save_to_db(self.df)

        self.df.set_index("material_id", inplace=True, drop=False)

    def read_base_structures(self, path: str, target_structures=None):
        ut.custom_print("Reading relaxed structures...")

        # Getting the path where the calculations will be searched for.
        if path:
            read_path = pathlib.Path(path)
        else:
            read_path = pathlib.Path()

        if target_structures:
            selection_criteria = target_structures
        else:
            selection_criteria = self.db_phase_diagram.keys()

        folders = read_path.glob("./*")
        list_dir = [
            fold
            for fold in folders
            if pathlib.PurePath(fold).name in selection_criteria
        ]

        for calc_fold in list_dir:
            # Getting information about the current calculation
            curr_phase = pathlib.PurePath(calc_fold).name
            ut.custom_print(
                f"Loading calculation for '{curr_phase}' as a base structure.", "debug"
            )

            # Loading current calculation info
            xml_path = pathlib.Path(calc_fold, "vasprun.xml")
            curr_run = vasp.Vasprun(xml_path, parse_potcar_file=False)

            # Gathering phase information
            for phase in self.db_phase_diagram.phase_names:
                for folder in xml_path.parts:
                    if slugify(folder) == slugify(phase):
                        curr_phase = phase
                        curr_phase = self.db_phase_diagram.get_phase(phase)
                        curr_mat_id = curr_phase.prototype
                        curr_name = f"base_relax_{curr_phase.name}_MP"

            # Creating the structure object
            curr_struct = mdb_struct.Structure().from_vasprun(
                vasprun=curr_run,
                base=True,
                phase=curr_phase,
                material_name=curr_name,
                bulk=True,
                perturb=False,
                cluster=False,
                surface=False,
                material_id=curr_mat_id,
            )

            # Saving the structure to the database
            self.df = curr_struct.save_to_db(self.df)

    # def get_base_structures_mp(self, optimize=True):
    #     ut.custom_print("Getting base structures from the materials project...")

    #     for calc_fold in list_dir:
    #         # Getting information about the current calculation
    #         curr_phase = pathlib.PurePath(calc_fold).name
    #         ut.custom_print(
    #             f"Loading calculation for '{curr_phase}' as a base structure.", "debug"
    #         )

    #         # Loading current calculation info
    #         xml_path = pathlib.Path(calc_fold, "vasprun.xml")
    #         curr_run = vasp.Vasprun(xml_path, parse_potcar_file=False)

    #         # Gathering phase information
    #         for phase in self.db_phase_diagram.phase_names:
    #             for folder in xml_path.parts:
    #                 if slugify(folder) == slugify(phase):
    #                     curr_phase = phase
    #                     curr_phase = self.db_phase_diagram.get_phase(
    #                         phase
    #                     )
    #                     curr_mat_id = curr_phase.prototype
    #                     curr_name = f"base_relax_{curr_phase.name}_MP"

    #         # Creating the structure object
    #         curr_struct = mdb_struct.Structure().from_vasprun(
    #             vasprun=curr_run,
    #             base=True,
    #             phase=curr_phase,
    #             material_name=curr_name,
    #             bulk=True,
    #             perturb=False,
    #             cluster=False,
    #             surface=False,
    #             material_id=curr_mat_id,
    #         )

    #         # Saving the structure to the database
    #         self.df = curr_struct.save_to_db(self.df)

    def save_database(self, path: str = None, suffix: str = None):
        """
        Saves the database dataframe into a pkl object.

        Parameters
        ----------
        path : str, optional.
            Location where the pickle object will be saved,
            by default None, which defaults to storing the file in the CWD.
        suffix : str, optional.
            String that will be added at the end of the filename.

        """
        if suffix:
            filename = self.database_name + f"_{suffix}.xz"
        else:
            filename = self.database_name + ".xz"

        if not path:
            path = ""

        file_path = pathlib.Path(path, filename)

        with lzma.open(file_path, "wb") as f:
            pickle.dump(self, f)

        ut.custom_print(f"Database saved in {file_path}", "info")

    def _apply_user_filters(self, filters: list, target_entries: pd.DataFrame):
        # Creating a empty DataFrame with the same column dtypes but no entries.
        target_entries_filter = target_entries[0:0]

        # Iterating over every filter type and getting each related structure,
        # which will get concatenated to the empty dataframe
        for fil in filters:
            filter_entries = target_entries.loc[target_entries[fil]]
            target_entries_filter = pd.concat((target_entries_filter, filter_entries))

        return target_entries_filter

    def perturb_gauss(
        self, center: float = 0.04, repeat: int = 5, filters: list = None
    ):
        # Getting all structures which are not perturbed
        target_entries = self.df.loc[(~self.df.material_name.str.contains("_perturb"))]

        # Applying user specified filters
        if filters:
            target_entries = self._apply_user_filters(filters, target_entries)

        ut.custom_print(f"Applying filters {filters} for perturbation.")
        ut.custom_print(
            f"{len(target_entries)*repeat} perturbed entries will be added.", "debug"
        )

        # Applying displacement to all perturbed structures
        for idx, entry in target_entries.iterrows():
            # print('entry: ', entry)
            # Getting information from the current entry
            str_matid = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure

            # Generating string for naming the current structure
            extra_info = ""
            if entry.supercell:
                extra_info += f"_super-{mdb_surf.get_miller_index_str(entry.supercell)}"
            if entry.replacement:
                extra_info += (
                    f"_repl-{entry.replacement_ind[0]}-{entry.replacement_ind[1]}"
                )

            for perturb_repeat_idx in range(repeat):
                # Applying displacement
                new_struct_perturb = self._apply_gauss_perturb(
                    center=center, structure=curr_str
                )

                mat_str = f"{entry.material_name}_perturb_gauss_{perturb_repeat_idx+1}"

                # Creating a new Structure from the perturbed structure structure
                curr_struct = mdb_struct.Structure(
                    material_name=mat_str,
                    structure=new_struct_perturb,
                    material_id=str_matid,
                    phase=str_phase,
                    base=False,
                    perturb=True,
                    supercell=entry.supercell,
                    replacement=entry.replacement,
                    formula=new_struct_perturb.formula,
                    symmetry=new_struct_perturb.get_space_group_info(),
                    temperature=entry.temperature,
                    calc_performed=False,
                )

                # Converting the structure to the appropiate type
                if entry.bulk:
                    curr_struct_conv = mdb_struct.Bulk().from_mdb_structure(curr_struct)
                elif entry.surface:
                    curr_struct_conv = mdb_struct.Surface().from_mdb_structure(
                        curr_struct,
                        entry.surface_miller,
                    )
                else:
                    raise NotImplementedError(
                        "This perturbation strategy is not implemented "
                        "for the current structure type."
                    )

                # Saving the bulk to the db.
                self.df = curr_struct_conv.save_to_db(self.df)

    def _apply_gauss_perturb(self, structure: Structure, center: float = 0.04):
        new_structure = structure.copy()
        new_structure.perturb(distance=0.08, min_distance=0.02)
        return new_structure

    def perturb_min_displacement(
        self,
        frac_max: float = 0.05,
        frac_min: float = 0.01,
        repeat=5,
    ):
        # Getting all relaxed structures
        target_entries = self.df.loc[self.df.base]

        # Applying displacement to all perturbed structures
        for idx, entry in target_entries.iterrows():
            # Getting some parameters from the current perturb structure.
            str_matid = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure

            # Applying the perturbation 'repeat' times.
            for perturb_repeat_idx in range(repeat):
                # Applying displacement,
                new_struct_perturb = self._apply_min_perturbation(structure=curr_str)

                mat_str = (
                    f"{str_matid}_{str_phase.name}_perturb_min_{perturb_repeat_idx+1}"
                )

                # Creating a new Structure from the perturbed structure structure
                curr_struct = mdb_struct.Structure(
                    material_name=mat_str,
                    structure=new_struct_perturb,
                    material_id=str_matid,
                    phase=str_phase,
                    base=False,
                    perturb=True,
                    supercell=entry.supercell,
                    replacement=entry.replacement,
                    formula=entry.formula,
                    symmetry=new_struct_perturb.get_space_group_info(),
                    temperature=entry.temperature,
                    calc_performed=False,
                )

                # Converting the structure to the appropiate type
                if entry.bulk:
                    curr_struct_conv = mdb_struct.Bulk().from_mdb_structure(curr_struct)
                elif entry.surface:
                    curr_struct_conv = mdb_struct.Surface().from_mdb_structure(
                        curr_struct
                    )
                else:
                    raise NotImplementedError(
                        "This perturbation strategy is not implemented "
                        "for the current structure type."
                    )

                # Saving the bulk to the db.
                self.df = curr_struct_conv.save_to_db(self.df)

    def _apply_min_perturbation(
        self, structure: Structure, frac_max: float = 0.05, frac_min: float = 0.01
    ):
        perturb_structure = structure.copy()
        # Making a copy of the current structure lattice which can be modified
        matrix = np.copy(perturb_structure.lattice.matrix)

        # Select non-zero indices
        non_zero_mask = np.abs(matrix) > 0.01

        # Compute perturbations for all non-zero values
        fraction = (frac_max - frac_min) * np.random.ranf(size=matrix.shape) + frac_min

        # Applying perturbation as a mask
        displacements = matrix * fraction
        displacements = np.where(non_zero_mask, displacements, 0)

        # Randomly add or subtract perturbations
        signs = np.random.choice([1, -1], size=len(non_zero_mask))

        # Apply perturbations
        matrix += signs * displacements

        # Updating perturb_structure with displaced matrix
        perturb_structure.lattice = matrix

        return perturb_structure

    def _get_phase_from_id(self, idx: str) -> str:
        """
        Searches for the corresponding phase in the self.db_phase_diagram
        dict to a given a material projects id.

        Parameters
        ----------
        idx : str
            Materials project id

        Returns
        -------
        str
            A phase of the CuZn phase diagram
        """
        # Creating a list of the phase diagram phase names
        phase_list = [
            phase.name
            for phase in self.db_phase_diagram.phases
            if phase.prototype == idx
        ]

        # Returning the key if found
        if len(phase_list) > 0:
            return phase_list[0]
        else:
            return None

    def _convert_prototype_structure(
        self, structure: Structure, phase: str
    ) -> Structure:
        """
        Convert a provided prototype structure to a structure compatible
        with the CuZn alloy preparation by replacing all atoms with Cu.

        Parameters
        ----------
        structure : Structure
            Prototype structure to be modified

        Returns
        -------
        Structure
            Structure which may have all its atoms replaced by Cu if it contains
            anything other than Cu or Zn atoms.
        """
        # Checking if there are any other atoms than Cu or Zn in the structure
        if len(set(structure.symbol_set) - self.database_elements) > 0:
            # Creating a new structure using the base one as a template
            new_structure = structure.copy(sanitize=True)

            # Replacing base_elem atoms in the structures
            for ind in range(len(structure.species)):
                new_structure.replace(
                    ind,
                    # Species(self.db_phase_diagram.get_phase(phase).base_elem),
                    Element(self.db_phase_diagram.get_phase(phase).base_elem),
                )

            # Returning new structure with atoms replaced with Cu
            return new_structure

        else:
            # If everything is already either Cu or Zn, leave the structure as is.
            return structure

    def _gather_prototype_structure(
        self,
        prototype: str,
        phase: str,
        get_different_supercells: bool,
        read: bool,
    ):
        """
        Gather the structure for a prototype from the materials project database,
        while checking that the phase given for the material is correct

        Parameters
        ----------
        prototype : str
            Materials project id for the prototype structure.
        phase : str
            Name of the phase to be generated.
        get_different_supercells: bool
            Wether to get one or more supercells from the base structure.
        read: bool
            Wether to read the structure from the database or to use the MP API.


        Returns
        -------
        Structure
            Prototype structure
        emmet.core.summary.SummaryDoc
            Object containing information for the queried structure

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.db_phase_diagram dictionary.
            More phases could be added there if necessary.
        """
        # Checking for correct phase input
        phase_name = slugify(phase.name)
        if not self.db_phase_diagram.get_phase(phase_name):
            raise KeyError(
                "Wrong phase given. "
                f"Please introduce one of: {[k for k in self.db_phase_diagram.phases]}"
            )

        # Reading structure from database
        if read:
            ut.custom_print("Using structure from the db as template...", "debug")
            try:
                query_result = self.df.loc[self.df.phase == phase]
                material_id_prefix = query_result.material_id.values[0]
                structure = query_result.structure.values[0]
            except IndexError:
                raise mdb_exc.BaseStructureNotFound()

        # Querying CuZn alpha prototype structure
        else:
            ut.custom_print("Querying the MP API...", "debug")
            with MPRester(self.secrets["API_KEY"]) as mpr:
                query_result = mpr.summary.search(material_ids=[prototype])[0]
                structure = query_result.structure
                material_id_prefix = query_result.material_id

        # TODO: Add a toggle so that the user can choose to use it.
        # Converting all of the atoms from the prototype cell to
        # the base atom type if necessary
        # structure = self._convert_prototype_structure(structure=structure, phase=ph)

        # Getting conventional cell for the replaced structure
        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()

        # Create supercells for the replaced structure
        # This supercell will result in a Cu64 cell
        # This can return either 1 or more supercells of the
        # same structure, depending on the 'get_different_supercells' flag.
        structure_list, idx_list, supercells = self._find_supercell_indices(
            structure, get_different_supercells, max_atoms=self.max_num_atoms
        )

        struct_obj_list = []
        # Saving all the generated supercells as separate bulk structures
        for structure, idxs in zip(structure_list, supercells):
            # Getting the supercell vector as a string for naming
            idxs_str = "".join(map(str, idxs))

            # Creating a new bulk from the supercell
            curr_bulk = mdb_struct.Bulk(
                material_name=f"{material_id_prefix}_{phase.name}_super-{idxs_str}",
                material_id=material_id_prefix,
                structure=structure,
                temperature=query_result.temperature.values[0],
                perturb=False,
                surface=False,
                base=False,
                cluster=False,
                calc_performed=False,
                supercell=idxs,
                phase=phase,
            )

            # Saving the bulk to the db.
            self.df = curr_bulk.save_to_db(self.df)

            struct_obj_list.append(curr_bulk)

        return struct_obj_list, query_result, idx_list

    def _create_symmetrical_prototype(
        self,
        structure: Structure,
        phase: mdb_pd.Phase,
        structure_obj: mdb_struct.Structure,
    ):
        phase = structure_obj.phase
        curr_phase_atom = self.db_phase_diagram.get_phase(phase).base_elem
        base_atom_set = list(self.database_elements - {curr_phase_atom})

        new_structure = structure.copy(sanitize=True)

        # Replacing atoms in the structures
        ind = 2
        sum_ind = 0
        sum_list = (2, 1, 2, 3)

        while ind < structure.num_sites:
            # new_structure.replace(ind - 1, Species(base_atom_set[0]))
            new_structure.replace(ind - 1, Element(base_atom_set[0]))
            ind = ind + sum_list[sum_ind]

            if sum_ind == 3:
                sum_ind = 0
            else:
                sum_ind += 1

        material_id_prefix = structure_obj.material_id

        # Generating the symmetrized structure
        new_struct_symm = mdb_struct.Structure(
            material_name=f"{material_id_prefix}_{phase.name}_symm",
            material_id=material_id_prefix,
            structure=structure,
            temperature=structure_obj.temperature,
            perturb=False,
            surface=False,
            base=False,
            cluster=False,
            calc_performed=False,
            supercell=structure_obj.supercell,
            phase=phase,
        )

        if structure_obj.bulk:
            final_struct = mdb_struct.Bulk().from_mdb_structure(
                mdb_structure=new_struct_symm,
                new_structure=structure,
            )

        # TODO: Make this work for surfaces and clusters.
        else:
            raise NotImplementedError("Current function only implemented for bulks.")

        self.df = final_struct.save_to_db(self.df)

        return structure

    def _gen_base_elem_perc(self, phase, num_struct):
        # Computing base_elem percentages using offset
        if self.use_offset:
            # Getting offset. If not found set to 0.
            offset = phase.offset

            # Randomly generating base_elem percentages for the new structures
            max_base_elem = (phase.base_elem_comp_max) + offset
            if max_base_elem > 1:
                max_base_elem = 1

            min_base_elem = (phase.base_elem_comp_min) - offset
            if min_base_elem < 0:
                min_base_elem = 0

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

    def _fit_replacements_phase(
        self,
        phase,
        structure,
        subst_base_elem_perc,
    ):
        curr_comp = structure.composition
        base_elem = phase.base_elem
        (other_elem,) = self.database_elements - {base_elem}

        tot_base_at_struct = curr_comp[base_elem]

        structure_len = len(structure.species)
        offset_min = phase.base_elem_comp_min - phase.offset
        offset_max = phase.base_elem_comp_max + phase.offset

        n_at_replacement_upd = []
        for str_ind, curr_perc in enumerate(subst_base_elem_perc):
            inPhase = phase.perc_in_phase(curr_perc)

            single_at_perc = 1 / structure_len
            perc_range = offset_max - offset_min

            # Skip this offset if changing one atom always results
            # in going over the maximum or minimum.
            if single_at_perc >= perc_range:
                inPhase = True

            while not inPhase:
                perc = (tot_base_at_struct + abs(curr_perc)) / structure_len

                if perc >= offset_max:
                    curr_perc -= single_at_perc
                elif perc <= offset_min:
                    curr_perc += single_at_perc
                else:
                    inPhase = phase.perc_in_phase(curr_perc)

            new_n_at = int(round(curr_perc * structure_len, 0))
            n_at_replacement_upd.append(new_n_at)

        return n_at_replacement_upd

    def _apply_replacement(self, structure: Structure, phase, n_atoms: int, rng=None):
        if not rng:
            rng = np.random.default_rng()

        if isinstance(
            structure, (mdb_struct.Structure, mdb_struct.Surface, mdb_struct.Bulk)
        ):
            structure = structure.structure

        structure_len = len(structure.species)
        curr_comp = structure.composition

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

        # Getting current structure composition information
        # The current procedure assumes that all of the atom species in the structure
        # will have been replaced beforehand with the base atom,
        # although this results in more randomness.
        base_elem = phase.base_elem
        (other_elem,) = self.database_elements - {base_elem}

        # If the structure only has one type of Element, and that is not the base
        # element, this changes with what to replace.
        if not curr_comp.as_dict().get(base_elem.symbol):
            base_elem = structure.composition.elements[0]
            (other_elem,) = self.database_elements - {base_elem}
            other_atom_change = n_atoms

        else:
            # Getting how many base atoms must be changed in order for the
            # structure to meet the current percentage requirements.
            target_atoms_base = curr_comp[base_elem] - abs(n_atoms)

            # Getting how many atoms of the other element must be changed
            other_atom_change = int(curr_comp[other_elem] - target_atoms_base)
            # print('other_atom_change: ', other_atom_change)

        # Choosing which species of the structure to change with the other atom.
        # print('structure_len: ', structure_len)
        # print('abs(int(other_atom_change)): ', abs(int(other_atom_change)))
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
            new_structure.replace(ind, other_elem)

        # TODO: Instead of this, create a new structure
        # Copying site properties
        new_structure = new_structure.copy(
            sanitize=True, site_properties=site_props_before
        )

        return new_structure

    def generate_bulk_structures(
        self,
        prototype: str,
        phase: str,
        num_struct: int,
        num_repeats: int,
        get_different_supercells: bool,
        read: bool = True,
    ):
        """
        This method allows to create several variations of a certain phase
        structure by randomly replacing atoms in the base structure.

        Parameters
        ----------
        prototype : str
            Materials project id of the prototype structure to be used as template.
        phase : str
            Name of the phase to be used.
        num_struct : int
            Number of different atomic compositions to be generated.
        num_repeats : int
            Number of random replacements done for each atomic composition.
        get_different_supercells : bool
            Whether to store just a single supercell or several of them.
            If False, a single supercell is chosen so that the resulting structure
            has a total number of atoms under a certain threshold.
            If True, the same structure is chosen, but additionally, any possible
            structure with smaller supercells is also added.
        read: bool
            Whether to read structures from the db or use the MP API to get them, by
            default True.

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.db_phase_diagram dictionary.
            More phases could be added there if necessary.
        """
        # Instantiating RNG
        rng = np.random.default_rng()

        # Getting the prototype structure
        # First, the structure is either gathered from the MP or the initial database,
        # then all atoms are replaced with Cu. Next, the conventional cell is obtained
        # and finally a supercell is created. Depending on the setting, one or more
        # supercells can be returned.
        structure_list, query_result, idx_list = self._gather_prototype_structure(
            get_different_supercells=get_different_supercells,
            prototype=prototype,
            phase=phase,
            read=read,
        )

        for structure_obj, supr_idx in zip(structure_list, idx_list):
            curr_structure = structure_obj.structure

            # TODO: Add a toggle so that the user can choose to use it.
            # Converting all of the atoms from the prototype cell to the
            # base atom type if necessary
            structure = self._convert_prototype_structure(
                structure=curr_structure, phase=phase
            )

            # Replacing some atoms using symmetry
            structure = self._create_symmetrical_prototype(
                structure=structure, phase=phase, structure_obj=structure_obj
            )
            # Preparing an array of randomly generated base elem percentages
            # for the new structures
            subst_base_elem_perc = self._gen_base_elem_perc(phase, num_struct)

            # Choosing the amount of atoms to replace with the base element in the
            # struct which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            # n_at_replacement = [
            #     int(round(structure_len * stct, 0)) for stct in subst_base_elem_perc
            # ]

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            n_at_replacement_upd = self._fit_replacements_phase(
                phase, structure, subst_base_elem_perc
            )
            # Replacing the atoms and generate 'num_replacements'
            # structures for each percentage
            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                for repl in range(num_repeats):
                    # Applying the replacement
                    new_structure = self._apply_replacement(
                        structure, phase, n_atoms, rng
                    )

                    # Getting the supercell vector
                    supercell_vec_str = mdb_surf.get_miller_index_str(
                        structure_obj.supercell
                    )

                    # Creating a new Bulk object for the structure with replacement
                    new_struct_symm = mdb_struct.Bulk(
                        material_name=f"{prototype}_{phase.name}_super-{supercell_vec_str}-{supr_idx}_replacement-{str_ind+1}-{repl+1}",
                        material_id=prototype,
                        structure=new_structure,
                        temperature=query_result.temperature.values[0],
                        perturb=False,
                        surface=False,
                        replacement=True,
                        replacement_ind=(str_ind + 1, repl + 1),
                        base=False,
                        cluster=False,
                        calc_performed=False,
                        supercell=structure_obj.supercell,
                        phase=phase,
                    )

                    self.df = new_struct_symm.save_to_db(self.df)

    def get_base_structs_current_phase(self, phase):
        # Getting all of the base structures
        base_structs = self.df.loc[self.df.base]

        if isinstance(phase, str):
            phase = self.db_phase_diagram.get_phase(phase=phase)

        # Getting the structures corresponding to the current phase
        phase_mask = base_structs.phase == phase

        # If an old version of the phase object is being used
        if not phase_mask.any():
            phase_mask = base_structs.phase == phase.name

        base_structs = base_structs.where(phase_mask, other=pd.NA)
        base_structs.dropna(how="all", inplace=True)

        return base_structs

    def generate_surfaces_pure(
        self,
        phase: mdb_pd.Phase,
        num_diff_layer_size: int,
        max_miller_index: int = 2,
        min_slab_size: float = 3,
        max_slab_size: float = 6,
        min_vacuum_size: float = 10,
        get_supercells=False,
        get_replacements=False,
        num_replacement_structs: int = 3,
        num_replacement_repeats: int = 5,
        fixed_layers: int = 0,
        overwrite_max_num_atoms: int = None,
        limit_per_phase: int = None,
    ):
        """
        Generating a series of surfaces from the base structures using a phase
        as a template. This method must be executed in an InitialDatabase object
        that contains structures labelled as 'base' and a Phase from a PhaseDiagram
        object must be given.
        The structure generation from the CatKit library is leveraged.

        Parameters
        ----------
        phase : Phase
            Phase of the current system's phase diagram that will contain
            atomic ratio information.
        num_diff_layer_size : int
            How many different sized layers will be generated, using the maximum
            and minimum slab size.
        max_miller_index : int, optional
            Maximum index on the miller indices. The function will generate
            all miller indices starting from zero up to this maximum
            value, by default 2.
        min_slab_size : float, optional
            Minimum size of the slab in Angstrom, by default 3
        max_slab_size : float, optional
            Maximum size of the slab in Angstrom, by default 6
        min_vacuum_size : float, optional
            Minimum size of the vacuum in Angstrom, by default 10
        get_supercells : bool, optional
            Whether to generate supercells for each Slab, by default False
        get_replacements : bool, optional
            Whether to generate new Slabs with random replacements.
            This will be done to all generated Slabs, by default False
        num_replacement_structs : int, optional
            How many different random replacement percentages to generate
            for every structure, by default 3.
        num_replacement_repeats : int, optional
            How many times to repeat the random replacement of a single
            percentage in a structure, by default 5.
        fixed_layers : int, optional
            How many layers to fix at the bottom, by default 0
        overwrite_max_num_atoms : int, optional
            A parameter that overrides the max number of atoms of the
            InitialDatabase object, so larger surfaces can be created
            when generating supercells, by default None.

        Raises
        ------
        mdbex.BaseStructureNotFound
            This exception will raise if no base structures can be found for a
            certain phase.
        """
        # Getting the current phase from the phase name.
        if isinstance(phase, str):
            phase = self.db_phase_diagram.get_phase(phase)

        base_structs = self.get_base_structs_current_phase(phase)

        # Checking if there are any base structures for the current
        # phase.
        if len(base_structs) == 0:
            err_msg = (
                f"No base structure could be found for phase {phase.name}."
                "\nThe database must contain base structures before "
                "running this function."
            )

            raise mdb_exc.BaseStructureNotFound(err_msg)

        # Preparing equispaced points between initial random value and the
        # maximum thickness value.
        slab_sizes = np.linspace(min_slab_size, max_slab_size, num_diff_layer_size)

        for idx, row in base_structs.iterrows():
            # Getting the current base structure
            curr_bulk = row.structure

            # Getting the number of atoms in the conventional cell
            # of the bulk
            curr_surf_nat = len(curr_bulk.species)

            # Getting a range of maximum number of atoms using the bulk
            # atom number and the max atom number specified.
            if overwrite_max_num_atoms:
                max_atom_num_list = np.linspace(
                    curr_surf_nat,
                    overwrite_max_num_atoms,
                    int(overwrite_max_num_atoms / curr_surf_nat),
                )
            else:
                max_atom_num_list = np.linspace(
                    curr_surf_nat,
                    self.max_num_atoms,
                    int(self.max_num_atoms / curr_surf_nat),
                )
            # print("self.max_num_atoms: ", self.max_num_atoms)
            # print("curr_surf_nat: ", curr_surf_nat)
            # print(
            #     "int(self.max_num_atoms/curr_surf_nat): ",
            #     int(self.max_num_atoms / curr_surf_nat),
            # )

            # Getting an ASE Atoms object
            curr_bulk_ase = AseAtomsAdaptor().get_atoms(curr_bulk)

            # Preparing the progress bar
            text_column = riprg.TextColumn(
                "        {task.description}", table_column=riprg.Column(ratio=3)
            )
            rialg.Align(text_column.render, align="right")
            bar_column = riprg.BarColumn(table_column=riprg.Column())
            time_col = riprg.TimeElapsedColumn(table_column=riprg.Column())
            spin_col = riprg.SpinnerColumn(table_column=riprg.Column())
            remaining_col = riprg.MofNCompleteColumn(table_column=riprg.Column())
            t_remaining_col = riprg.TimeRemainingColumn(table_column=riprg.Column())
            empty_col = riprg.TextColumn("", table_column=riprg.Column())

            overall_progress = riprg.Progress(
                riprg.TextColumn(
                    " [···]  {task.description}",
                    table_column=riprg.Column(ratio=3),
                ),
                bar_column,
                time_col,
                t_remaining_col,
                remaining_col,
                # expand=True,
            )
            total_slabs_gen = list(it.product(slab_sizes, max_atom_num_list[:]))
            total_slabs = len(total_slabs_gen)
            main_task_descr = f"Generating {phase.name} slabs:"
            overall_task = overall_progress.add_task(
                main_task_descr, total=int(total_slabs)
            )

            job_progress = riprg.Progress(
                text_column,
                bar_column,
                time_col,
                spin_col,
                empty_col,
                # expand=True,
            )

            group = ricns.Group(overall_progress, job_progress)
            live = riliv.Live(group, refresh_per_second=4)

            total_slabs_generated = []
            total_slabs_generated_count = 0
            # Rich live progress bars wrapping the surface generation
            # process. A task is created for each layer+n_at number,
            # drawing a progress bar while it gets computed.
            with live:
                while not overall_progress.finished:
                    for n_layers, n_at in total_slabs_gen:
                        # sub_task = job_progress.add_task(
                        #     description=f"{int(n_layers)} layers, {int(n_at)} atoms:",
                        #     total=None,
                        # )

                        gen_slabs = mdb_surf._gen_curr_surface(
                            db_obj=self,
                            phase=phase,
                            curr_bulk_ase=curr_bulk_ase,
                            n_layers=n_layers,
                            n_at=n_at,
                            max_miller_index=max_miller_index,
                            fixed_layers=fixed_layers,
                            get_supercells=get_supercells,
                            limit_per_phase=limit_per_phase,
                        )

                        total_slabs_generated_count += len(gen_slabs)
                        total_slabs_generated.extend(gen_slabs)
                        # job_progress.update(sub_task, total=1)
                        # job_progress.advance(sub_task, advance=1)

                        overall_progress.advance(overall_task, advance=1)

            # Counter for the total number of structures
            total_slabs_generated_final_cnt = len(total_slabs_generated)

            # Applying replacements
            if get_replacements:
                ut.custom_print("Applying replacements...", "debug")
                rng = np.random.default_rng()

                replacement_list = []

                for idx, gen_slab in enumerate(total_slabs_generated):
                    # Getting current phase and structure length.
                    slab_phase = gen_slab.phase

                    # Getting the base element percentage of the current structure
                    current_perc = slab_phase.get_base_elem_perc(gen_slab.structure)

                    # Generating a list of random percentages inside the current phase
                    # range.
                    gen_percentages = mdb_surf.gen_perc_surfaces(
                        phase=slab_phase,
                        num_struct=num_replacement_structs,
                        current_perc=current_perc,
                        relative=True,
                        # db_obj=self,
                    )

                    # Going over the generated percentages
                    for str_ind, n_atoms in enumerate(gen_percentages):
                        # Repeating the replacement for each percentage, so that
                        # num_replacement_repeats structures are generated with
                        # the same ratio but different distribution.
                        for repl in range(num_replacement_repeats):
                            # Applying the replacement
                            new_structure = self._apply_replacement(
                                structure=gen_slab,
                                phase=phase,
                                n_atoms=n_atoms,
                                rng=rng,
                            )

                            # Generating name
                            if gen_slab.supercell:
                                supercell_vec_str = mdb_surf.get_miller_index_str(
                                    gen_slab.supercell
                                )
                                supercell_vec_str_name = f"super-{supercell_vec_str}_"
                            else:
                                supercell_vec_str = gen_slab.surface_miller
                                supercell_vec_str_name = supercell_vec_str

                            mat_name = (
                                f"{phase.prototype}_{phase.name}_surface"
                                f"-{supercell_vec_str_name}-{str_ind+1}"
                                f"_replacement-{repl + 1}"
                            )

                            # Creating a new Surface object for the
                            # structure with replacement
                            new_struct_symm = mdb_struct.Surface(
                                material_name=mat_name,
                                material_id=phase.prototype,
                                surface_miller=supercell_vec_str,
                                structure=new_structure,
                                temperature=gen_slab.temperature,
                                perturb=False,
                                replacement=True,
                                replacement_ind=(str_ind + 1, repl + 1),
                                base=False,
                                calc_performed=False,
                                supercell=gen_slab.supercell,
                                phase=phase,
                            )

                            replacement_list.append(new_struct_symm)

            # Limiting the number of structures per phase
            if limit_per_phase and len(replacement_list) >= limit_per_phase:
                rng = np.random.default_rng()
                slabs_selection = rng.choice(
                    len(replacement_list), size=limit_per_phase, replace=False
                )
                replacement_list = np.take(replacement_list, slabs_selection, axis=0)

            for struct in replacement_list:
                # Saving new structure into the database.
                total_slabs_generated_final_cnt += 1
                self.df = struct.save_to_db(self.df)

            ut.custom_print(
                f"Generated {total_slabs_generated_final_cnt} surfaces.", "done"
            )

    def _get_main_elem_perc(self, phase: mdb_pd.Phase, structure):
        """
        This function provides the percentage of the main element
        for a given structure

        Parameters
        ----------
        phase : Phase
            Object describing the current phase
        structure : _type_
            Structure for which the percentage is to be calculated

        Returns
        -------
        float
            Percentage of main element in the structure
        """
        main_species = phase.base_elem
        main_cnt = 0

        species_list = structure.species
        total_atoms = len(species_list)

        for element in species_list:
            if element.symbol == main_species:
                main_cnt += 1
                print("main_cnt: ", main_cnt)

        perc = main_cnt / total_atoms
        return perc

    def generate_surfaces_replacements(
        self,
        phase: mdb_pd.Phase,
        num_struct: int,
        num_repeats: int,
    ):
        if isinstance(phase, str):
            phase = self.db_phase_diagram.get_phase(phase)

        # Instantiating RNG
        rng = np.random.default_rng()

        # Getting the base structures
        slabs_df = self.df.loc[self.df.surface]
        slabs_df = slabs_df.loc[self.df.base]
        slabs_df = slabs_df.loc[self.df.phase == phase.name]
        print("slabs_df: ", slabs_df)

        for idx, row in slabs_df.iterrows():
            print("idx: ", idx)
            curr_surface = row.structure

            structure_len = len(curr_surface.species)

            # Getting current percentage of main element
            strct_perc = self._get_main_elem_perc(
                phase=phase,
                structure=curr_surface,
            )

            # Randomly generating base elem percentages for the new structures
            subst_base_elem_perc = self._gen_perc_surfaces(
                phase=phase,
                num_struct=num_struct,
                current_perc=strct_perc,
            )

            print("subst_base_elem_perc: ", subst_base_elem_perc)

            # Choosing the amount of atoms to replace with the base element in the
            # struct, which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            n_at_replacement = []
            for stct in subst_base_elem_perc:
                curr_repl = int(round(structure_len * abs(stct), 0))
                if stct < 0:
                    curr_repl *= -1
                n_at_replacement.append(curr_repl)

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            n_at_replacement_upd = self._fit_replacements_phase(
                phase, curr_surface, subst_base_elem_perc
            )

            # Adapting the replacement percentages generated to the
            # percentage of the current structure
            # n_at_replacement_final = self._adjust_replacements()

            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                print("\nreplacement: ", str_ind)
                for repl in range(num_repeats):
                    print("\nreplicate: ", repl)
                    # Replacing atoms according to the current phase from the structure.
                    new_structure = self._apply_replacement(
                        curr_surface, phase, structure_len, n_atoms, rng
                    )

                    prototype = phase.prototype
                    extra = {"surface": True}

                    # Matching the miller index from the name as it is not stored.
                    match = re.search(r"\((.*?)\)", row.material_id)
                    if match:
                        curr_miller = match.group(1)
                    else:
                        curr_miller = "???"

                    # Preparing the structure name
                    mat_id_name = f"{prototype}_{phase.name}"
                    mat_id_surf_data = f"_replacement-({curr_miller})-_{str_ind}-{repl}"
                    material_id = mat_id_name + mat_id_surf_data

                    print("new", new_structure.formula)
                    final_perc = self._get_main_elem_perc(phase, new_structure)
                    print(phase)
                    print("final_perc: ", final_perc)

                    # Saving the structure into the database.
                    self._save_row(
                        material_id=material_id,
                        phase=phase,
                        structure=new_structure,
                        extra=extra,
                    )

    def _save_row(
        self,
        structure,
        material_id=None,
        phase=None,
        extra=None,
        base=False,
    ):
        # HACK: This should probably use __dict__ instead of a manually set list...
        # Attributes not to store in a row
        unwanted_attrs = ["save_to_db", "from_vasprun", "from_mdb_structure"]

        # If given structure is a pymatgen Structure
        if isinstance(structure, Structure):
            new_row = pd.Series(
                {
                    "material_id": material_id,
                    "structure": structure,
                    "temperature": None,
                    "perturb": True,
                    "phase": phase.name,
                    "base": base,
                    "formula": structure.formula,
                    **extra,
                }
            )

        # If given structure is a MatDBForge structure.
        elif isinstance(structure, mdb_struct.Structure):
            attr_list = [
                att
                for att in dir(structure)
                if not att.startswith("_") and att not in unwanted_attrs
            ]
            att_dict = {att: getattr(structure, att) for att in attr_list}

            new_row = pd.Series(att_dict)

        new_row_df = new_row.to_frame().T
        new_row_df = new_row_df.astype(
            {
                "perturb": bool,
                "base": bool,
                "bulk": bool,
                "surface": bool,
                "cluster": bool,
                "calc_performed": bool,
                "replacement": bool,
            }
        )

        self.df = self.df.astype(
            {
                "perturb": bool,
                "base": bool,
                "bulk": bool,
                "surface": bool,
                "cluster": bool,
                "calc_performed": bool,
                "replacement": bool,
            }
        )

        # Concatenating a row with boolean columns to an empty array results
        # in a warning message. In order to avoid the warning, if the
        # dataframe is empty, it is replaced by the dataframe with the complete
        # row in this first instance only, and then the concatenation is done
        # as usual for the rest of the execution.
        if self.df.shape[0] == 0:
            self.df = new_row_df
        else:
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

    def _gather_n2p2_reqdata_from_node(self, node):
        # Getting calculation name
        name = node.label + "_aiida-uuid_" + node.uuid

        # Writing the vasprun.xml file to a buffer.
        retrieved = node.outputs.retrieved
        vasprun_f = retrieved.get_object_content("vasprun.xml", "rb")
        buffer = BytesIO(vasprun_f)

        # Reading the file from the buffer and closing it
        vasprun = aseio.read(buffer, format="vasp-xml", index="-1")
        buffer.close()

        # Getting properties from the vasprun
        pot_energy = vasprun.get_potential_energy(force_consistent=True) * self.eV2Eh

        # Getting forces
        # Reading forces from vasprun.xml, in eV/Ang and converting them to Ha/Bohr
        forces = vasprun.get_forces() * self.eV2Eh * self.Bohr2Ang

        lattice = vasprun.get_cell() * self.Ang2Bohr
        structure = vasprun.get_positions() * self.Ang2Bohr
        symbols = vasprun.get_chemical_symbols()

        # Setting charge to 0
        # charge = contcar.structure.charge
        charge = 0

        data_dict = {
            "name": name,
            "lattice": lattice,
            "positions": structure,
            "symbols": symbols,
            "pot_energy": pot_energy,
            "charge": charge,
            "forces": forces,
        }

        return data_dict

    def _add_entry_to_n2p2_input(self, buffer: TextIOWrapper, data_dict: dict):
        # Writing begin keyword and structure name
        buffer.write("begin\n")
        buffer.write(f'comment {data_dict.get("material_name","no name found")}\n')

        # Getting lattice parameters and converting them to Bohr
        lat_x = data_dict["lattice"][0]
        lat_y = data_dict["lattice"][1]
        lat_z = data_dict["lattice"][2]

        # Writing lattice parameters
        buffer.write(f"lattice {lat_x[0]:.6f} {lat_x[1]:.6f} {lat_x[2]:.6f}\n")
        buffer.write(f"lattice {lat_y[0]:.6f} {lat_y[1]:.6f} {lat_y[2]:.6f}\n")
        buffer.write(f"lattice {lat_z[0]:.6f} {lat_z[1]:.6f} {lat_z[2]:.6f}\n")

        # Writing information for every atom. Every atom line must contain:
        # atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1>
        for idx, (at, frc) in enumerate(
            zip(data_dict["positions"], data_dict["forces"])
        ):
            # Preparing and writing the line
            buffer.write(
                f"atom {at[0]:.6f} {at[1]:.6f}"
                f" {at[2]:.6f}"
                f" {data_dict['symbols'][idx]} {0:.6f} {0:.6f}"
                f" {frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n"
            )

        # writing potential energy and charge
        buffer.write(f'energy {data_dict["pot_energy"]:.8f}\n')
        buffer.write(f'charge {data_dict["charge"]:.6f}\n')

        # writing end keyword
        buffer.write("end\n")

    def generate_n2p2_input_aiida(
        self, aiida_group_list: list, filter_dict: dict, path: str = None
    ):
        # Loading aiida profile
        load_profile()

        # Handling path
        if path and isinstance(path, str):
            path = pathlib.Path(path)
        else:
            path = pathlib.Path()

        # Adding input.data filename to path
        path = path / "input.data"

        # Gathering nodes from the given group
        ut.custom_print("Getting nodes...")

        # Preparing a query in the aiida db
        qb = orm.QueryBuilder()

        for group in aiida_group_list:
            print("group: ", group)
            qb.append(orm.Group, filters={"label": group}, tag="group")
            (qb.append(orm.WorkChainNode, with_group="group", filters=filter_dict),)

        result_nodes = qb.all(flat=True)

        if len(result_nodes) == 0:
            for group in aiida_group_list:
                qb.append(VaspCalculation, with_group="group", filters=filter_dict)
            result_nodes = qb.all(flat=True)

        ut.custom_print(f"{len(result_nodes)} nodes found.", "info")

        # Writing the file
        with open(path, "w") as curr_f:
            # Checking every node
            for node in riprg.track(
                result_nodes, description=" [ ⧖ ]  Writing info..."
            ):
                # Gathering the information from each node
                data_dict = self._gather_n2p2_reqdata_from_node(node=node)

                # Writing the information to the buffer
                self._add_entry_to_n2p2_input(buffer=curr_f, data_dict=data_dict)
        ut.custom_print(f"All calculations saved in '{path}'.", "done")

    def limit_structure_number_phases(
        self,
        structure_limit: int,
        structure_types: list,
        phases_to_use: list = None,
    ):
        # Getting which phases to use
        if phases_to_use:
            num_phases = len(phases_to_use)
        else:
            phases_to_use = self.db_phase_diagram.phases
            num_phases = len(self.db_phase_diagram.phases)

        # Maximum number of structures per phase
        max_struct_phase = structure_limit // num_phases

        # Getting the current phase as a Phase object
        # If the phase is not found, omitting.
        for curr_phase in phases_to_use:
            try:
                curr_phase = self.db_phase_diagram.get_phase(curr_phase)
            except mdb_exc.PhaseNotFound:
                pass

            structure_list = self.df.loc[self.df.phase == curr_phase]

            # Getting the selected types to use.
            # TODO: Fix this, as now only keeps one structure type.
            # for structure_type in structure_types:
            # structure_list.loc[structure_list['channel'].isin(['sale','fullprice'])]
            # structure_list = structure_list.loc[structure_list[structure_type]]

            # If the number of structures for the selected types is larger than
            # the maximum allowed, reduce the number by sampling a certain amount.
            if structure_list.shape[0] > max_struct_phase:
                # Getting the all but the original surface used for this phase,
                # which should be maintained in the database.
                changed_structures = structure_list.loc[
                    (structure_list.phase == curr_phase)
                    & (
                        structure_list.perturb
                        | structure_list.supercell
                        | structure_list.replacement
                    )
                ]

                to_remove = structure_list.loc[
                    structure_list.unique_id.isin(changed_structures.unique_id)
                ]
                structure_list_base = structure_list.drop(to_remove.index)

                sample_amount = structure_limit - structure_list_base.shape[0]
                changed_structures_sample = changed_structures.sample(sample_amount)

                # These are the selected structures for the desired phase and type.
                # We want to keep these structures in the original dataframe.
                phase_structures = pd.concat(
                    [structure_list_base, changed_structures_sample]
                )

                # We remove all structures of the selected type
                # and phase from the original
                # dataframe, and add the selected ones.
                orig_removed = self.df.loc[
                    ~self.df.unique_id.isin(structure_list.unique_id)
                ]

                phase_result_df = pd.concat([orig_removed, phase_structures])
                self.df = phase_result_df

    def generate_clusters(
        self,
        size_range: list,
        get_replacements=False,
        get_perturbed=False,
        add_dimer=False,
        save_in_db=False,
        limit_per_phase: int = None,
        num_struct: int = 2,
        num_repeat: int = 2,
    ):
        # Getting default phase
        phase = self.db_phase_diagram.get_phase("alpha")

        # Generate a list of mdb_struct.Cluster
        cluster_list = []

        # Adding a dimer
        if add_dimer:
            ut.custom_print("Adding base dimers...", "debug")
            clust_obj = mdb_clust.make_clean_dimer(self, phase=phase)
            cluster_list.append(clust_obj)
            ut.custom_print("Base dimers done...", "debug")

        # Create clusters over all size range given, from smallest to largest.
        ut.custom_print(
            f"Adding base clusters with n_atoms: {size_range[0]}-{size_range[-1]}...",
            "debug",
        )
        for size in size_range:
            clust_obj = mdb_clust.make_clean_cluster(self, size=size, phase=phase)
            cluster_list.append(clust_obj)
        ut.custom_print("Base clusters done...", "debug")

        # If True, store the structures along with their information
        # into the MatDBForge InitialDatabase object
        if save_in_db:
            for idx, cluster in enumerate(cluster_list):
                self._save_row(structure=cluster)

        # Return the cluster list in case the user just wants the clusters
        # but not storing them into the database.
        ut.custom_print(f"Generated {len(cluster_list)} base clusters.", "debug")
        return cluster_list

    def display_db_ase(self):
        structures = self.df.structure
        ut._display_indb_dataframe(structures)


# class CuZnInitialDatabase(InitialDatabase):
#     """
#     Object representing a initial database for a CuZn alloy intented
#     to prepare a NNP. The database is stored as a pandas dataframe.
#     Contains methods related to gathering, preparing and modifying
#     the initial database.

#     Returns
#     -------
#     CuZnInitialDatabase
#         Object containing the database and methods.

#     Parameters
#     ----------
#     database_name : str
#         Name for the database. Will be used for internal reference and as a filename for
#         saving the dataframe.
#     use_offset : bool, optional
#         Use an offset for the phase ratios to allow them to overlap, by default True.
#     max_num_atoms : int
#         Maximum number of atoms present in any structure generated, by default 64.
#     secrets : dict
#         Object containing secrets related to the materials project database.

#     Raises
#     ------
#     KeyError
#         This error is raised when a wrong phase is given.
#     """

#     # TODO: This should be created in a function
#     # CuZn alloy phase diagram data
#     # alpha_phase = mdb_pd.Phase(
#     #     name="alpha",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0,
#     #     base_elem_comp_max=0.3895,
#     #     prototype="mp-30",
#     #     offset=0.03,
#     # )
#     # m1 = mdb_pd.Phase(
#     #     name="m1",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.3895,
#     #     base_elem_comp_max=0.45,
#     #     prototype="mp-30",
#     #     offset=0.03,
#     # )
#     # beta_prime = mdb_pd.Phase(
#     #     name="beta-prime",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.455,
#     #     base_elem_comp_max=0.507,
#     #     prototype="mp-987",
#     #     offset=0.05,
#     # )
#     # m2 = mdb_pd.Phase(
#     #     name="m2",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.51,
#     #     base_elem_comp_max=0.577,
#     #     prototype="mp-987",
#     #     offset=0.05,
#     # )
#     # gamma = mdb_pd.Phase(
#     #     name="gamma",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.577,
#     #     base_elem_comp_max=0.706,
#     #     prototype="mp-1368",
#     #     offset=0.03,
#     # )
#     # m3 = mdb_pd.Phase(
#     #     name="m3",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.706,
#     #     base_elem_comp_max=0.785,
#     #     prototype="mp-1216020",
#     #     offset=0.05,
#     # )
#     # delta = mdb_pd.Phase(
#     #     name="delta",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.7302,
#     #     base_elem_comp_max=0.765,
#     #     prototype="mp-1215518",
#     #     offset=0.05,
#     # )
#     # epsilon = mdb_pd.Phase(
#     #     name="epsilon",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.785,
#     #     base_elem_comp_max=0.883,
#     #     prototype="mp-972042",
#     #     offset=0.05,
#     # )
#     # m4 = mdb_pd.Phase(
#     #     name="m4",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.883,
#     #     base_elem_comp_max=0.9725,
#     #     prototype="mp-79",
#     #     offset=0.01,
#     # )
#     # eta = mdb_pd.Phase(
#     #     name="eta",
#     #     base_elem="Zn",
#     #     cluster_elem="Cu",
#     #     base_elem_comp_min=0.9725,
#     #     base_elem_comp_max=1,
#     #     prototype="mp-79",
#     #     offset=0.01,
#     # )
#     # db_phase_diagram = mdb_pd.BinaryPhaseDiagram(
#     #     "CuZn", alpha_phase, m1, beta_prime, m2, gamma, m3, delta, epsilon, m4, eta
#     # )
#     ####

#     # TODO: This should be passed to the original DB
#     # Which atoms are involved in the alloy.
#     # database_elements = {Element("Cu"), Element("Zn")}


# def _add_cols_old_db_version(database_old):
#     print("database_old: ", database_old)
#     quit()


# def get_database(path) -> InitialDatabase:
#     """
#     Load a database from a file on the cwd or a specific path.

#     Returns
#     -------
#     InitialDatabase
#         MatDBForge InitialDatabase containing structure data.
#     """
#     ut.custom_print("hola", "debug")
#     if isinstance(path, InitialDatabase):
#         ut.custom_print("Database already loaded!", "warning")
#         return path

#     elif isinstance(path, str):
#         db_path = pathlib.Path(path)
#         if db_path.exists():
#             if db_path.suffixes[0] == ".pkl":
#                 database = pd.read_pickle(db_path)
#                 ut.custom_print(
#                     "Old database version detected. Adding columns...", "debug"
#                 )
#                 database = _add_cols_old_db_version(database)
#             elif db_path.suffixes[0] == ".xz":
#                 with lzma.open(db_path, "rb") as f:
#                     database = pickle.load(f)
#             ut.custom_print(f"Loaded '{db_path}'", "info")
#             return database
#         else:
#             err_msg = (
#                 "MatDBForge InitialDatabase object not found "
#                 "in the specified location."
#             )
#             raise FileNotFoundError(err_msg)


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run this file! "
        "This file is intended to be used as a module and not a script."
    )
