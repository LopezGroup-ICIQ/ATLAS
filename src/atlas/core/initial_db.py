"""
Generates a pandas dataframe containing a set of
base (unperturbed) structures and a certain number of structures
with an applied perturbation with respect to the temperature.
"""

import itertools as it
import lzma
import pathlib
import pathlib as pl
import pickle
import time
import warnings
from collections import Counter
from io import BytesIO, TextIOWrapper
from os import cpu_count
from typing import Self

import ase.io as aseio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatgen.io.vasp as vasp
import rich.progress as riprg
from aiida import load_profile, orm
from aiida_vasp.calcs.vasp import VaspCalculation
from ase import Atoms
from ase import build as ase_build
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from rich import print as rprint
from rich.panel import Panel
from rich.pretty import Pretty
from slugify import slugify

import atlas as atl
import atlas.core.clusters as atl_clust
import atlas.core.code_utils as atl_cut
import atlas.core.exceptions as atl_exc
import atlas.core.phase_diagram as atl_pd
import atlas.core.structure as atl_struct
import atlas.core.surfaces as atl_surf
from atlas.core import initial_db as indb
from atlas.core import utils as ut

# from atlas.active_learning.active_learning_utils import AVAILABLE_FILTERS
from atlas.core.filtering.structure_filters import apply_struct_filters_atl_db

# Filtering certain warnings
warnings.filterwarnings('ignore', category=vasp.outputs.UnconvergedVASPWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class InitialDatabase:
    """
    Object that creates an initial database where structures will be
    stored. Materials are downloaded using the materials project API.
    Contains methods related to gathering, preparing, visualizing
    and modifying the initial database.
    The structures are stored internally as a pandas Dataframe.

    Attributes
    ----------
    df : pd.Dataframe
        Dataframe containing the structures for the initial database.
    database_name : str
        Orientative name for the database. Will be used for saving it into a file.
    database_path : str | pl.Path
        Path where the database will be saved.
    max_num_atoms : int
        Maximum number of atoms present in any structure generated, by default 64.
    secrets : dict
        Object containing the API key for the materials project database.
    use_offset : bool, optional
        Use an offset for the phase ratios to allow them to overlap, by default True.

    Returns
    -------
    InitialDatabase
        Object containing the database and methods.

    Raises
    ------
    KeyError
        This error is raised when a wrong phase is given.

    Notes
    -----
    In order to be able to gather structures from the MP API, the initial
    setup of ATLAS must have been completed, or a 'secrets.json' file
    should be located in the same directory.
    The json file containing the secrets should have the following structure:

    >>> {
    >>>        "API_KEY": "XXXXXX"
    >>> }

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
        database_name: str = 'initial_database',
        max_num_atoms: int = 64,
        load_db: bool = False,
        phase_diagram: atl_pd.BinaryPhaseDiagram = None,
        use_offset: bool = True,
        database_path: str | pl.Path = '.',
        create_db=True,
    ) -> None:
        self.db_version = atl.__version__

        # Name of the database
        self.database_name = database_name

        # Path for the database
        self.database_path = pl.Path(database_path)

        # Setting the maximum number of atoms of any generated structure.
        self.max_num_atoms = max_num_atoms

        # Setting the phase diagram
        self.phase_diagram = phase_diagram

        # Using the offset
        if use_offset:
            atl_cut.custom_print(
                'Using an offset for computing the phases concentrations.', 'info'
            )
            self.use_offset = use_offset

        # Track removal history for filtering
        self._removal_history = {'filtering': {}}

        # Create the database if it does not exists
        # Load it if otherwise.
        db_exists = self._check_database()

        if db_exists or load_db:
            self.df = self._load_database()
        elif not (db_exists or load_db) and create_db:
            self.df = self._create_database()

    def __iter__(self):
        return iter(self.df.iterrows())

    def __repr__(self):
        # Getting the class name
        class_name = self.__class__.__name__

        # Getting the amount of entries in the database
        if hasattr(self, 'df'):
            count = self.df.shape
            repr_string = (
                f"{class_name} named '{self.database_name}' containing {count} entries."
            )
        else:
            repr_string = f"Empty {class_name} named '{self.database_name}'."

        return repr_string

    def __len__(self):
        return self.df.shape[0]

    @staticmethod
    def _adapt_old_db(database_old):
        """Adapter for the old version of the database."""
        columns_to_add = {
            'bulk': bool,
            'surface': bool,
            'cluster': bool,
            'calc_performed': bool,
            'replacement': bool,
        }

        for col in columns_to_add:
            database_old[col] = None

        return database_old

    def _load_database(self) -> Self | pd.DataFrame:
        """
        Load a database from a pickle file on the cwd or a specific path.

        Returns
        -------
        pd.DataFrame
            Dataframe containing structure data for the initial database.
        """
        db_path = pl.Path(self.database_path) / pl.Path(self.database_name)
        atl_cut.custom_print(f"Loading database: '{self.database_name}'", 'debug')
        self.database_name = db_path.stem

        # If no suffixes are present, add the default one.
        if len(db_path.suffixes) == 0:
            db_path = db_path.with_suffix('.xz')

        if not db_path.exists():
            raise FileNotFoundError(f"Database '{db_path}' does not exist.")

        # Compatibility with the old version of the database
        if '.pkl' in db_path.suffixes:
            suffix = '.pkl'
            atl_cut.custom_print(
                'Using outdated version of database. Adding missing columns.',
                'debug',
            )
            database = pd.read_pickle(db_path)
            database_df = self._adapt_old_db(database)

        # Loading the database
        elif '.xz' in db_path.suffixes or suffix == '.xz':
            suffix = '.xz'
            with lzma.open(db_path, 'rb') as f:
                loaded_db_obj = pickle.load(f)

                # Setting parameters from InitialDatabase
                self.database_name = loaded_db_obj.database_name
                self.max_num_atoms = loaded_db_obj.max_num_atoms
                self.db_version = loaded_db_obj.db_version

                atl_cut.custom_print(
                    f'Using database version {self.db_version}.', 'debug'
                )

            database_df = loaded_db_obj.df

        else:
            raise atl_exc.IncompatibleDataBase(
                f"Database '{db_path}' has an unsupported format."
            )

        atl_cut.custom_print(f"Loaded '{self.database_name}{suffix}'", 'info')
        atl_cut.custom_print(f'Path: {db_path}', 'debug')

        return database_df

    @classmethod
    def load_database(cls, database_path: pl.Path | str) -> Self | pd.DataFrame:
        """
        Load a ATL database from a specific path.

        Returns
        -------
        InitialDatabase | pd.DataFrame
            Object containing structure data for the initial database.
        """
        db_path = pl.Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database '{db_path}' does not exist.")

        atl_cut.custom_print(f"Loading database from: '{db_path.resolve()}'", 'info')

        # If no suffixes are present, add the default one.
        suffix = db_path.suffix
        if not suffix:
            suffix = '.xz'
            db_path = db_path.with_suffix(suffix)

        # Compatibility with the old version of the database
        if '.pkl' in db_path.suffixes:
            suffix = '.pkl'
            atl_cut.custom_print(
                'Using outdated version of database. Adding missing columns.',
                'warn',
            )
            database = pd.read_pickle(db_path)
            database = cls._adapt_old_db(database)

        # Loading the database
        elif '.xz' in db_path.suffixes:
            suffix = '.xz'
            with lzma.open(db_path, 'rb') as f:
                database = pickle.load(f)

                atl_cut.custom_print(
                    f'Using database version {database.db_version}.', 'info'
                )

        else:
            try:
                db_ase_atoms = aseio.read(filename=db_path, index=':')
                phases = []
                species = set()

                # Instantiate the class (supports inheritance)
                database = cls()

                # Prepare structures
                if isinstance(db_ase_atoms, list):
                    atl_struct_list = []
                    for atoms in db_ase_atoms:
                        atl_struct_list.append(
                            atl_struct.Structure.from_ase_atoms(ase_atoms=atoms)
                        )
                        phases.append(atoms.info.get('phase', 'unknown'))
                        species = species.union(atoms.symbols.species())
                else:
                    atl_struct_list = [
                        atl_struct.Structure.from_ase_atoms(ase_atoms=db_ase_atoms)
                    ]
                    phases.append(db_ase_atoms.info.get('phase', 'unknown'))
                    species = species.union(db_ase_atoms.symbols.species())

                # Populate database
                for struct in atl_struct_list:
                    database = struct.save_to_db(database)

                # Create phase diagram
                base_elem = Element(Counter(phases).most_common(1)[0][0].title())
                phases_str_list = set(phases)
                phase_diagram = atl_pd.PhaseDiagram(
                    material='loaded_material',
                    base_elem=base_elem,
                    element_list=list(species),
                )

                # Adding placeholder phases
                for phase_str in phases_str_list:
                    curr_phase = atl_pd.Phase(
                        name=phase_str,
                        element_list=list(species),
                        cluster_elem=base_elem,
                        composition={base_elem.symbol: {'min': 0.0, 'max': 1.0}},
                        spacegroup='unknown',
                        symbol=base_elem.symbol,
                        prototype='unknown',
                        offset=0,
                        # replace_dict=None,
                        allow_modifications=True,
                        phase_diagram=phase_diagram,
                        use_cache=False,
                    )
                    phase_diagram.add_phase(curr_phase)

                database.phase_diagram = phase_diagram

            except Exception as e:
                print(f'#@# Error: {e}')
                raise atl_exc.IncompatibleDataBase(
                    f"Database '{db_path}' has an unsupported format."
                ) from None

        return database

    def get_db_shape(self) -> tuple:
        # Getting the amount of entries in the database
        return self.df.shape

    def gen_report(self) -> dict:
        """
        Generate a report containing the database information.

        Returns
        -------
        dict
            Dictionary containing the database information in different categories.
        """
        # Getting the amount of entries in the database
        # count = len(self.df.count(axis=1))
        count = len(self.df.index)

        # Getting the database name
        db_name = self.database_name

        # Getting the database path
        db_path = self.database_path

        # Getting the database version
        db_version = self.db_version

        # Getting the maximum number of atoms
        max_atoms = self.max_num_atoms

        struct_info_dict = {
            'structure_count': {
                'by_type': {
                    'bulk': 0,
                    'base': 0,
                    'surface': 0,
                    'cluster': 0,
                    'isolated_atom': 0,
                    'md': 0,
                },
                'by_modification': {
                    'perturb': 0,
                    'vacancy': 0,
                    'deformation': 0,
                },
            },
            'phases': {},
        }

        for phase in self.phase_diagram.phases:
            struct_info_dict['phases'][phase.name] = 0

        for struct in self.df.iterrows():
            if hasattr(struct[1], 'init_md') and struct[1].init_md:
                struct_info_dict['structure_count']['by_type']['md'] += 1
            elif hasattr(struct[1], 'phase') and struct[1].phase == 'IsolatedAtom':
                struct_info_dict['structure_count']['by_type']['isolated_atom'] += 1
            elif hasattr(struct[1], 'base') and struct[1].base:
                struct_info_dict['structure_count']['by_type']['base'] += 1
            elif hasattr(struct[1], 'bulk') and struct[1].bulk:
                struct_info_dict['structure_count']['by_type']['bulk'] += 1
            elif hasattr(struct[1], 'surface') and struct[1].surface:
                struct_info_dict['structure_count']['by_type']['surface'] += 1
            elif hasattr(struct[1], 'cluster') and struct[1].cluster:
                struct_info_dict['structure_count']['by_type']['cluster'] += 1

            if (
                hasattr(struct[1], 'targeted_modification')
                and struct[1].targeted_modification
            ):
                mod_type = struct[1].targeted_modification

                if mod_type == 'central_atom_perturbation':
                    mod_type = 'oct_perturb'

                if not struct_info_dict['structure_count']['by_modification'].get(
                    mod_type
                ):
                    struct_info_dict['structure_count']['by_modification'][mod_type] = 1
                else:
                    struct_info_dict['structure_count']['by_modification'][
                        mod_type
                    ] += 1
            elif hasattr(struct[1], 'vacancy') and struct[1].vacancy:
                struct_info_dict['structure_count']['by_modification']['vacancy'] += 1
            elif hasattr(struct[1], 'deformation') and struct[1].deformation:
                struct_info_dict['structure_count']['by_modification'][
                    'deformation'
                ] += 1
            elif hasattr(struct[1], 'perturb') and struct[1].perturb:
                struct_info_dict['structure_count']['by_modification']['perturb'] += 1

            phase_key = (
                struct[1].phase.name
                if hasattr(struct[1].phase, 'name')
                else struct[1].phase
            )
            if phase_key not in struct_info_dict['phases']:
                struct_info_dict['phases'][phase_key] = 1
            else:
                struct_info_dict['phases'][phase_key] += 1

        # Adding database info
        struct_info_dict.update(
            {
                'database_settings': {
                    'database_name': db_name,
                    'database_version': db_version,
                    'database_path': str(db_path.absolute()),
                    'total_entries': count,
                    'structure_max_atoms': max_atoms,
                },
            }
        )

        # Sorting structure count dictionary by values
        struct_info_dict['structure_count']['by_type'] = dict(
            sorted(
                struct_info_dict['structure_count']['by_type'].items(),
                key=lambda x: x[1],
            )
        )
        struct_info_dict['structure_count']['by_modification'] = dict(
            sorted(
                struct_info_dict['structure_count']['by_modification'].items(),
                key=lambda x: x[1],
            )
        )

        # Sorting phase dictionary by values
        struct_info_dict['phases'] = dict(
            sorted(struct_info_dict['phases'].items(), key=lambda x: x[1])[::-1]
        )

        # Add an additonal key to the 'by_modification' dictionary, containing
        # the total number of non-modified structures. This adds context to the
        # plot.
        total_mod_structs: int = sum(
            struct_info_dict['structure_count']['by_modification'].values()
        )
        struct_info_dict['structure_count']['by_modification']['none'] = (
            total_mod_structs
        )

        struct_info_dict['structure_count']['by_modification']['none'] += (
            struct_info_dict['structure_count']['by_modification'].pop(True, 0)
        )

        # Add removal history if available (without the UUID list)
        if hasattr(self, '_removal_history') and self._removal_history.get('filtering'):
            removal_data = dict(self._removal_history['filtering'])
            removal_data.pop('filtered_uuids', None)
            struct_info_dict['removal_history'] = removal_data

        return struct_info_dict

    def _check_database(self) -> bool:
        """
        Check if a database with the name 'self.database_name'
        exists in the current working directory or is a path to a existing
        database.

        Returns
        -------
        bool
            True if the database exists, False if not does not.
        """
        # Checking if dataframe already exists on the cwd.
        file_exists = False
        file_check = []

        db_path: pl.Path = pl.Path(self.database_path)

        # Checking for a file with correct name and suffixes.
        if db_path.is_dir():
            for file in db_path.iterdir():
                if self.database_name in file.name and set(file.suffixes) & {
                    '.xz',
                    '.pkl',
                }:
                    file_check.append(file)
        if (
            db_path.is_file()
            and self.database_name in db_path.name
            and set(db_path.suffixes) & {'.xz', '.pkl'}
        ):
            file_check.append(db_path)

        if len(file_check) > 0:
            file_exists = True

        # If the database name is a path
        name_as_path = pathlib.Path(self.database_name)

        if name_as_path.exists():
            file_exists = True

        atl_cut.custom_print(f'Database found: {file_exists}.', 'debug')

        return file_exists

    def _create_database(self) -> pd.DataFrame:
        """
        Create an empty dataframe in order to be used in the class.

        Returns
        -------
        pd.DataFrame
            Empty dataframe containing the necessary columns to be used in the main
            database.
        """
        # Creating a pandas dataframe to store the structures
        df = pd.DataFrame(
            columns=[
                'material_name',
                'material_id',
                'structure',
                'phase',
                'formula',
                'symmetry',
                'base',
                'isolated_atom',
                'perturb',
                'unique_id',
                'supercell',
                'surface',
                'bulk',
                'cluster',
                'temperature',
                'magnetic_properties',
                'calc_energy',
                'calc_energy_per_atom',
                'calc_energy_toten',
                'calc_performed',
                'calc_type',
                'calc_output',
                'replacement',
                'vacancy',
                'targeted_modification',
                'deformation',
                'init_md',
            ]
        )

        df.attrs['db_version'] = self.db_version
        atl_cut.custom_print(f"Created database '{self.database_name}'.", 'done')

        return df

    @atl_cut.deprecated('Moved to `core.utils`', since_ver='0.13.0')
    def _find_supercell_indices(
        self,
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
        if isinstance(structure, Slab):
            supercell_vec = [idx, idx, 1]
        else:
            supercell_vec = [idx, idx, idx]

        new_structure.make_supercell(supercell_vec, to_unit_cell=False)

        # Number of atoms of the supercell
        struct_size = len(new_structure.species)
        while (
            struct_size > max_atoms or struct_size < min_atoms
        ) and supercell_vec != [1, 1, 1]:
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
            atl_cut.custom_print(
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
                            atl_cut.custom_print(
                                (
                                    f'Supercell generated (diff.) {supercell_vec} '
                                    f'- total atoms: {struct_size}'
                                ),
                                'debug',
                            )

        return structure_list, idx_list, supercell_vec_list

    def _check_repeat_struct(self, curr_phase, curr_struct: Structure):
        structure_list = self.df.loc[self.df.phase == curr_phase].structure.values

        # species_list = set([a.symbol for a in curr_struct.species])
        species_list = [el.Z for el in self.phase_diagram.alloy_set]

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
            average='inner',
        )

        for pym_struct in structure_list:
            ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

            # Create output for multiple system in parallel
            struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=False)
            # curr_feat_sum = struct_soap.sum()
            soap_structs.append(struct_soap)

        # TODO: Should this be here twice?
        curr_ase_struct = AseAtomsAdaptor().get_atoms(curr_struct)
        curr_struct_soap = soap.create(curr_ase_struct, n_jobs=-1, verbose=False)
        # curr_soap_sum = curr_struct_soap.sum()

        total_soap_arr = np.array(soap_structs)

        comp_arr = np.isclose(
            curr_struct_soap, total_soap_arr, rtol=7.5e-04, atol=5e-05
        )

        if np.count_nonzero(comp_arr) > 0:
            atl_cut.custom_print('Duplicate structure found!', 'warn')
            return True

        else:
            return False

    def _get_structure_type_row(
        self, bulk: bool, surface: bool, cluster: bool, isolated_atom: bool
    ) -> str:
        """
        Determines the type of a structure based on boolean flags.

        Parameters
        ----------
        - bulk (bool): True if the structure is a bulk.
        - surface (bool): True if the structure is a surface.
        - cluster (bool): True if the structure is a cluster.
        - isolated_atom (bool): True if the structure is an isolated_atom.

        Returns
        -------
        - str: The type of the structure ("bulk", "surface", or "cluster").
        """
        if bulk:
            return 'bulk'
        elif surface:
            return 'surface'
        elif cluster:
            return 'cluster'
        elif isolated_atom:
            return 'isolated_atom'
        else:
            return 'unknown'  # handle the case where no flags are True

    def add_single_atoms(self, vacuum: float = 10):
        """
        Add structures containing a single atom of each species in the database.

        Some MLIPs such as MACE require specifically labelled structures to be used
        for their reference energies. The structures are defined as a single atom of
        one of the species present in the MLIP training database. Structures are
        centered in the cell with some vacuum around them.

        Parameters
        ----------
        vacuum : float, optional
            Amount of vacuum to add around the atom in Angstroms, by default 10.
        """
        # Get all species present in the database
        species_list = []
        for _, db_entry in self.df.iterrows():
            species_list.extend(db_entry.structure.species)
        species_set = set(species_list)

        # Create an ase structure with a single atom of each species
        # centered on the cell, with some vacuum around it
        for spec in species_set:
            # Create a single atom structure
            single_atom = Structure(
                lattice=np.eye(3) * vacuum,
                species=[spec],
                coords=[[0.5, 0.5, 0.5]],
                site_properties={'magmom': [0.0]},
            )
            single_atom = atl_clust.center_structure(single_atom)

            # Add the structure to the database
            new_struct = atl_struct.IsolatedAtom(
                material_name=f'{spec}_reference',
                structure=single_atom,
                temperature=np.nan,
                perturb=False,
                bulk=False,
                vacancy=False,
                deformation=False,
                formula=single_atom.formula,
                symmetry=None,
                base=False,
                phase='IsolatedAtom',
                isolated_atom=True,
                init_md=False,
                magnetic_properties=np.nan,
                energy_per_atom=np.nan,
            )

            self.df = new_struct.save_to_db(self.df)

    def get_structure_list(self) -> list[Atoms]:
        """
        Retrieve all structures from the dataframe and return them as a list.

        Returns
        -------
        list[ase.Atoms]
            A list of ASE Atoms objects representing the structures in the dataframe.
        """
        structure_list = []

        for _, row in self.df.iterrows():
            # Get ASE structure
            pmg_curr_struct = row['structure']
            ase_curr_struct = AseAtomsAdaptor().get_atoms(pmg_curr_struct)

            # Populate structure with information
            ase_curr_struct.info['atl_struct_type'] = self._get_structure_type_row(
                bulk=row['bulk'],
                surface=row['surface'],
                cluster=row['cluster'],
                isolated_atom=row['isolated_atom'],
            )
            ase_curr_struct.info['atl_id'] = str(row['unique_id'])
            ase_curr_struct.info['struct_name'] = row['material_name']

            structure_list.append(ase_curr_struct)

        return structure_list

    def db_struct_to_ase(self, row) -> Atoms:
        ase_curr_struct = AseAtomsAdaptor().get_atoms(row.structure)

        # Populate structure with information
        ase_curr_struct.info['atl_struct_type'] = self._get_structure_type_row(
            bulk=row['bulk'],
            surface=row['surface'],
            cluster=row['cluster'],
            isolated_atom=row['isolated_atom'],
        )

        # Add custom information to the structure for MACE reference atoms
        if ase_curr_struct.info['atl_struct_type'] == 'isolated_atom':
            ase_curr_struct.info['config_type'] = 'IsolatedAtom'

        ase_curr_struct.info['atl_id'] = str(row['unique_id'])
        ase_curr_struct.info['struct_name'] = row['material_name']
        ase_curr_struct.info['perturb'] = row['perturb']
        ase_curr_struct.info['replacement'] = row['replacement']
        ase_curr_struct.info['base'] = row['base']
        ase_curr_struct.info['bulk'] = row['bulk']
        ase_curr_struct.info['cluster'] = row['cluster']
        ase_curr_struct.info['surface'] = row['surface']
        ase_curr_struct.info['surface_miller'] = row['surface_miller']
        ase_curr_struct.info['supercell'] = row['supercell']
        ase_curr_struct.info['symmetry'] = row['symmetry']
        ase_curr_struct.info['calc_type'] = row['calc_type']
        ase_curr_struct.info['calc_performed'] = row['calc_performed']
        ase_curr_struct.info['deformation'] = row['deformation']
        ase_curr_struct.info['vacancy'] = row['vacancy']
        ase_curr_struct.info['init_md'] = row['init_md']

        ase_curr_struct.info['targeted_modification'] = row['targeted_modification']
        if row.get('phase'):
            if isinstance(row['phase'], str):
                ase_curr_struct.info['phase'] = row['phase']
            else:
                ase_curr_struct.info['phase'] = row['phase'].name

        return ase_curr_struct

    def export_db(
        self,
        out_format: str = 'extxyz',
        file_name: str = None,
        file_path: str | pl.Path = None,
    ):
        """Export the structures of the database to a file using ASE."""
        struct_list = []
        for _, row in self.df.iterrows():
            # Get ASE structure
            # pmg_curr_struct = row['structure']
            ase_curr_struct = self.db_struct_to_ase(row)

            struct_list.append(ase_curr_struct)

        # Use database path and name if not specified
        if file_name is None:
            file_name = self.database_name
        if file_path is None and self.database_path:
            file_path = self.database_path
        if file_path is None and not self.database_path:
            file_path = pl.Path()

        file_path = pl.Path(file_path) / file_name
        file_path = file_path.with_suffix(f'.{out_format}')

        struct_list = self.standardize_struct_ids(struct_list)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        aseio.write(filename=file_path, images=struct_list, format=out_format)
        atl_cut.custom_print(f"Database exported to '{file_path}'", 'done')

    def standardize_struct_ids(self, atoms_list: list[Atoms]) -> list[Atoms]:
        """
        Ensure every ASE structure has standardized ``atl_*`` info keys.

        Delegates to
        :func:`atlas.active_learning.active_learning_utils.standardize_atoms_info`,
        which migrates legacy ``mdb_*`` keys to ``atl_*`` and ensures every
        structure has an ``atl_id`` UUID.

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            The list of ASE atoms objects to process.

        Returns
        -------
        list of ase.Atoms
            The updated list with standardized keys in ``info``.
        """
        from atlas.active_learning.active_learning_utils import (
            standardize_atoms_info,
        )

        return standardize_atoms_info(atoms_list)

    def find_repeat_structures(
        self,
        delete=False,
        filters: list = None,
        phase: atl_pd.Phase = None,
    ):
        """Find and delete repeated structures in the database.

        The SOAP descriptor is used to fingerprint structures in the database.
        """
        # Filtering the dataframe
        # Filters allow to select certain subsets of structures
        # from the database.
        filtered_df = self.df
        # remaining_df = self.df

        filtered_df, remaining_df, phase_list = ut.apply_filters_db(
            db_obj=filtered_df, phase=phase, filters=filters
        )

        # Getting the species from the current phase diagram
        species = self.phase_diagram.alloy_set
        species_str_list = [spec.symbol for spec in species]

        # Setting SOAP related parameters
        r_cut = 4
        n_max = 4
        l_max = 4

        atl_cut.custom_print(
            f'Setting up SOAP with: r_cut = {r_cut}, n_max = {n_max}, l_max = {l_max}',
            'debug',
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

            # Getting the names for the current structures
            uuid_list = filtered_df[filtered_df.phase == curr_phase]['atl_id'].values

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
            kernel = AverageKernel(metric='rbf', gamma=1)
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

            atl_cut.custom_print(
                (
                    f"Phase '{curr_phase.name}' - Total selected structures:"
                    f' {tot_structures}, equivalent: {tot_equival}'
                    f' ({(tot_equival / tot_structures) * 100:.2f}%)'
                ),
                'debug',
            )

        duplicate_structure_names = tot_duplicate_uuid_list

        # If the deletion flag is set, the function will delete the duplicate stuctures.
        if delete:
            atl_cut.custom_print(
                f'{len(duplicate_structure_names)} structures marked for deletion.',
                'debug',
            )

            # Getting the dataframe entries that match the stored uuids
            mat_name_match_mask = filtered_df.isin(
                {'unique_id': duplicate_structure_names}
            )['unique_id']
            duplicate_structures_df = filtered_df[mat_name_match_mask]

            # Dropping the matching entries
            filtered_df = filtered_df.drop(duplicate_structures_df.index)

            init_df_after_removal = pd.concat([remaining_df, filtered_df])

            self.df = init_df_after_removal

            atl_cut.custom_print(
                f'Deleted {len(duplicate_structures_df)} structures.',
                'warn',
            )

            atl_cut.custom_print(
                f'Dataframe shape after deleting: {self.df.shape}', 'debug'
            )

        else:
            atl_cut.custom_print(
                (
                    f'{len(duplicate_structure_names)} repeated structures found. '
                    "Database untouched as 'delete' is set to False."
                ),
                'info',
            )

    def gather_base_structures(self, phase_diag_phases):
        # Querying materials project database.
        atl_cut.custom_print(
            'Gathering base structures by querying the MP API...', 'info'
        )

        from mp_api.client import MPRester

        report_replacements = True
        with MPRester(ut.gather_secrets()['API_KEY'], mute_progress_bars=True) as mpr:
            for phase in phase_diag_phases:
                query_materials = phase.prototype

                if isinstance(query_materials, str):
                    query_materials = [query_materials]

                query_result = mpr.summary.search(material_ids=query_materials)

                ut.custom_print(
                    (
                        f'Gathered {len(query_result)} structures for phase'
                        f" '{phase.name}' from the MP,"
                        f' using {len(query_materials)} MP ids.'
                    ),
                    'info',
                )

                for material in query_result:
                    # for phase in self.phase_diagram.phases:
                    if (
                        isinstance(phase.prototype, str)
                        and phase.prototype == material.material_id
                    ):
                        curr_phase = phase.name
                        # break
                    elif isinstance(phase.prototype, list):
                        if material.material_id in phase.prototype:
                            curr_phase = phase.name
                            # break
                        else:
                            curr_phase = np.nan
                    else:
                        curr_phase = np.nan

                    try:
                        material_symmetry = material.get_space_group_info()
                    except Exception:
                        material_symmetry = material.symmetry.symbol

                    # Replacing elements
                    if phase.replace_dict:  # noqa: SIM102
                        if phase.replace_dict.get('replace'):
                            replace_with = phase.replace_dict.get(
                                'replace_with', phase.base_elem
                            )

                            replace_dict = {}
                            for element in phase.replace_dict['element_list']:
                                # If "M", all metals should be replaced.
                                # Thus, any metals found in the structure willl
                                # be added in phase.replace_dict["element_list"]
                                if element == 'M':
                                    for spec in material.structure.species:
                                        if spec.is_metal:
                                            phase.replace_dict['element_list'].append(
                                                spec.symbol
                                            )
                                            # Removing repeated keys
                                            phase.replace_dict['element_list'] = list(
                                                set(phase.replace_dict['element_list'])
                                            )

                                else:
                                    # Do not replace element with itself
                                    if element == replace_with:
                                        continue

                                    # Add element to replace_dict
                                    replace_dict[element] = str(replace_with)

                            # Removing M key from dict
                            if 'M' in replace_dict:
                                replace_dict.pop('M')

                            if report_replacements:
                                atl_cut.custom_print(
                                    (
                                        f'Applying substitution to '
                                        f'base structures: {replace_dict}...'
                                    ),
                                    'debug',
                                )
                                report_replacements = False

                            material.structure.replace_species(
                                species_mapping=replace_dict, in_place=True
                            )

                    curr_struct = atl_struct.Bulk(
                        material_id=str(material.material_id),
                        material_name=f'base_{material.material_id}',
                        structure=material.structure,
                        temperature=np.nan,
                        perturb=False,
                        bulk=True,
                        vacancy=False,
                        deformation=False,
                        formula=material.structure.formula,
                        symmetry=material_symmetry,
                        base=True,
                        phase=curr_phase,
                        magnetic_properties=material.total_magnetization,
                        energy_per_atom=material.energy_per_atom,
                    )

                    self.df = curr_struct.save_to_db(self.df)

        self.df.set_index('material_id', inplace=True, drop=False)

    def read_base_structures(self, path: str, target_structures=None):
        """Reads base structures from a given path and stores them in the database."""
        atl_cut.custom_print('Reading relaxed structures...')

        # Getting the path where the calculations will be searched for.
        read_path = pathlib.Path(path) if path else pathlib.Path()

        selection_criteria = target_structures or self.phase_diagram.keys()

        selection_criteria = [crit.name for crit in selection_criteria]

        folders = read_path.glob('./*')
        list_dir = [
            fold
            for fold in folders
            if pathlib.PurePath(fold).name in selection_criteria
        ]

        for calc_fold in list_dir:
            # Getting information about the current calculation
            curr_phase = pathlib.PurePath(calc_fold).name
            atl_cut.custom_print(
                f"Loading calculation for '{curr_phase}' as a base structure.", 'debug'
            )

            # Loading current calculation info
            xml_path = pathlib.Path(calc_fold, 'vasprun.xml')
            curr_run = vasp.Vasprun(xml_path, parse_potcar_file=False)

            # Gathering phase information
            for phase in self.phase_diagram.phase_names:
                for folder in xml_path.parts:
                    if slugify(folder) == slugify(phase):
                        curr_phase = phase
                        curr_phase = self.phase_diagram.get_phase(phase)
                        curr_mat_id = curr_phase.prototype
                        curr_name = f'base_relax_{curr_phase.name}_MP'

            # Creating the structure object
            curr_struct = atl_struct.Structure().from_vasprun(
                vasprun=curr_run,
                base=True,
                phase=curr_phase,
                material_name=curr_name,
                bulk=True,
                perturb=False,
                deformation=False,
                vacancy=False,
                cluster=False,
                surface=False,
                material_id=curr_mat_id,
                replacement=False,
            )

            # Saving the structure to the database
            self.df = curr_struct.save_to_db(self.df)

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
            filename = self.database_name + f'_{suffix}.xz'
        else:
            filename = self.database_name + '.xz'

        if not path:
            path = ''

        file_path = pathlib.Path(path, filename)

        with lzma.open(file_path, 'wb') as f:
            pickle.dump(self, f)

        atl_cut.custom_print(f'Database saved in {file_path}', 'info')

    def _apply_user_filters(self, filters: list, target_entries: pd.DataFrame):
        # Creating a empty DataFrame with the same column dtypes but no entries.
        target_entries_filter = target_entries[0:0]

        # Iterating over every filter type and getting each related structure,
        # which will get concatenated to the empty dataframe
        for fil in filters:
            filter_entries = target_entries.loc[target_entries[fil]]
            target_entries_filter = pd.concat((target_entries_filter, filter_entries))

        return target_entries_filter

    def apply_vacancies_random(
        self,
        filters: list,
        seed: int,
        repeat=2,
        element_list: list = None,
        max_vac_perc: float = 0.75,
        min_vac_perc: float = 0.25,
        lim_num_struc: int = None,
        phase: atl_pd.Phase = None,
    ):
        """Apply random vacancies to the structures in the database."""
        # Instantiating RNG
        rng = np.random.default_rng(seed=seed)

        # Apply filters to the database
        filtered_df, _, _ = ut.apply_filters_db(
            db_obj=self, filters=filters, phase=phase
        )

        # Filtering by phase if requested
        if phase:
            if isinstance(phase, str):
                filtered_df = filtered_df.loc[filtered_df.phase == phase]
            elif isinstance(phase, atl_pd.Phase):
                filtered_df = filtered_df.loc[filtered_df.phase == phase.name]

        # Check if the number of structures is less than the limit
        lim_num_struc = min(filtered_df.shape[0], lim_num_struc)

        # Select the random subset structures that will be perturbed
        sel_idx = rng.choice(filtered_df.index, size=lim_num_struc, replace=False)

        # Getting the structures that will be perturbed
        target_entries = filtered_df.loc[sel_idx]

        # Applying deformation to all perturbed structures
        for _, entry in target_entries.iterrows():
            if isinstance(entry.phase, str):
                curr_phase = self.phase_diagram.get_phase(entry.phase)
            else:
                curr_phase = entry.phase

            # If element_list is given, get the indices from the atoms in the
            # structure that match atoms in the element_list
            available_sites = []
            if element_list:
                for symbol in element_list:
                    idxs = entry.structure.indices_from_symbol(symbol)
                    available_sites.extend(idxs)

            # Applying repeat times
            for vac_idx in range(repeat):
                # Getting random vacancy percentage
                vac_perc = rng.uniform(min_vac_perc, max_vac_perc)

                # Getting random indices for the vacancies
                vac_indices = rng.choice(
                    available_sites,
                    size=int(vac_perc * len(available_sites)),
                    replace=False,
                )

                # Removing site from the structure
                new_struct_vac = entry.structure.copy()
                new_struct_vac = new_struct_vac.remove_sites(vac_indices)

                mat_str = f'{entry.material_id}_{curr_phase.name}_vacancy_{vac_idx + 1}'

                new_struct_templ: dict = entry.to_dict()

                # Updating template
                new_struct_templ['material_name'] = mat_str
                new_struct_templ['material_id'] = entry.material_id
                new_struct_templ['structure'] = new_struct_vac
                new_struct_templ['phase'] = curr_phase.name
                new_struct_templ['surface_miller'] = entry.surface_miller
                new_struct_templ['base'] = False
                new_struct_templ['bulk'] = entry.bulk
                new_struct_templ['surface'] = entry.surface
                new_struct_templ['cluster'] = entry.cluster
                new_struct_templ['perturb'] = True
                new_struct_templ['supercell'] = entry.supercell
                new_struct_templ['replacement'] = entry.replacement
                new_struct_templ['formula'] = entry.formula
                new_struct_templ['symmetry'] = new_struct_vac.get_space_group_info()[0]
                new_struct_templ['temperature'] = entry.temperature
                new_struct_templ['calc_performed'] = False
                new_struct_templ['vacancy'] = True
                new_struct_templ['init_md'] = False
                new_struct_templ['unique_id'] = None

                # Remove unused method
                for _key in ('from_db_row', 'to_ase_atoms', 'from_ase_atoms'):
                    new_struct_templ.pop(_key, None)

                # Creating a new Structure from the perturbed structure
                curr_struct = atl_struct.Structure(
                    **new_struct_templ,
                )

                # Converting the structure to the appropiate type
                if entry.bulk:
                    curr_struct_conv = curr_struct.to_bulk()
                elif entry.surface:
                    curr_struct_conv = curr_struct.to_surface()
                elif entry.cluster:
                    curr_struct_conv = curr_struct.to_cluster()
                else:
                    raise NotImplementedError(
                        'Vacancy generation is not implemented '
                        'for the current structure type.'
                    )

                # Saving the bulk to the db.
                self.df = curr_struct_conv.save_to_db(self.df)

    def gen_init_md_frames(
        self,
        filters: list,
        seed: int,
        md_gen_dict: dict,
        lim_num_struc: int = None,
        phase: atl_pd.Phase = None,
    ):
        """Run MD simulations for some structures in the database."""
        # Instantiating RNG
        # rng = np.random.default_rng(seed=seed)

        # Apply filters to the database
        filtered_df, _, _ = ut.apply_filters_db(
            db_obj=self,
            filters=filters,
            phase=phase,
        )
        # Force add filters for perturbed structures, displaced and replaced
        # structures
        filtered_df = filtered_df.loc[(filtered_df['base'])]

        # Filtering by phase if requested
        if phase is not None:
            if isinstance(phase, str):
                filtered_df = filtered_df.loc[filtered_df.phase == phase]
            elif isinstance(phase, atl_pd.Phase):
                filtered_df = filtered_df.loc[filtered_df.phase == phase.name]

        # Get rows with unique values using the 'material_id' column
        target_entries = filtered_df.drop_duplicates(subset='material_id')

        atl_cut.custom_print(
            (f'Running MD for {len(target_entries)} base structures...'),
            'info',
        )

        # Load MD settings
        md_params = md_gen_dict.get('md', {}).get('parameters', {})
        explode_filter_dict = (
            md_gen_dict.get('md', {}).get('filters', {}).get('exploding_structures', {})
        )
        T_start = md_params['temperature_K']

        from atlas.active_learning.active_learning_utils import run_mace_md_ase

        # Run MD for all selected structures
        md_struct_list = []
        for _, entry in target_entries.iterrows():
            # Get ATL structure as ASE atoms
            init_conf_orig = atl_struct.Structure().from_db_row(
                row=entry, columns=entry.index.to_list()
            )
            init_conf_orig = init_conf_orig.to_ase_atoms()
            init_conf = init_conf_orig.copy()
            info_dict = init_conf.info

            # Generate [2,2,1] supercell
            sup_transf = np.eye(3) * [2, 2, 2]

            init_conf = ase_build.make_supercell(prim=init_conf, P=sup_transf)
            init_conf.info = info_dict

            # Adding some displacement from the center of the cell
            init_conf.positions = init_conf.positions + [0.075, -0.05, 0.1]

            md_struct_list = run_mace_md_ase(
                init_conf,
                md_params,
                T_start,
                traj_obj=None,
                prepend_path='.',
                explode_filter_dict=explode_filter_dict,
                mode='db_gen',
                md_struct_list=md_struct_list,
            )

        if lim_num_struc < len(md_struct_list):
            idxs = np.linspace(start=0, stop=len(md_struct_list) - 1, num=lim_num_struc)
            md_struct_list = [md_struct_list[int(i)] for i in idxs]

        # Add the generated structures to the database
        for _, entry in target_entries.iterrows():
            curr_id = entry.material_id
            for idx, struct in enumerate(md_struct_list):
                if curr_id in struct.info.get('material_id'):
                    # Creating a new Structure from the initial MD structure.
                    pmg_struct = AseAtomsAdaptor().get_structure(struct)
                    mat_str = f'{entry.material_id}_{entry.phase}_init_md-{idx}'
                    templ_entry = entry.to_dict()
                    templ_entry['material_name'] = mat_str
                    templ_entry['structure'] = pmg_struct
                    templ_entry['supercell'] = str(sup_transf.diagonal())
                    templ_entry['replacement'] = entry.replacement
                    templ_entry['formula'] = entry.formula
                    templ_entry['symmetry'] = pmg_struct.get_space_group_info()[0]
                    templ_entry['temperature'] = entry.temperature
                    templ_entry['calc_performed'] = False
                    templ_entry['vacancy'] = False
                    templ_entry['base'] = False
                    templ_entry['init_md'] = True
                    templ_entry['unique_id'] = None

                    curr_struct = atl_struct.Structure(
                        **templ_entry,
                    )

                    # Converting the structure to the appropiate type
                    if entry.bulk:
                        curr_struct_conv = curr_struct.to_bulk()
                    elif entry.surface:
                        curr_struct_conv = curr_struct.to_surface()
                    elif entry.cluster:
                        curr_struct_conv = curr_struct.to_cluster()
                    else:
                        raise NotImplementedError(
                            'Vacancy generation is not implemented '
                            'for the current structure type.'
                        )

                    # Saving the structure to the db.
                    self.df = curr_struct_conv.save_to_db(self.df)

    def _apply_gauss_perturb(self, structure: Structure, center: float = 0.04):
        new_structure = structure.copy()
        new_structure.perturb(distance=center * 2, min_distance=center / 2)
        return new_structure

    def apply_min_deformation(
        self,
        frac_max: float = 0.05,
        frac_min: float = 0.01,
        repeat: int = 1,
        filters: list['str'] = None,
        only_use_base: bool = True,
        use_phase: atl_pd.Phase = None,
        rng_seed: int = None,
        limit_num_structures: int = None,
    ):
        """
        Apply small deformations to the lattice parameters of relaxed structures.

        This method perturbs the lattice parameters of structures by applying small
        deformations to their lattice matrix elements.
        The perturbations are repeated a specified number of times, creating multiple
        perturbed structures for each initial structure. This helps in generating
        structures with slightly higher energies and forces, useful for generating
        training data for neural network potentials (NNP) intended for
        molecular dynamics (MD) simulations.

        Parameters
        ----------
        frac_max : float, optional
            Maximum fraction of the lattice parameter perturbation, by default 0.05.
        use_phase: atl_pd.Phase, optional
            Phase to be used for the perturbation.
        filters : list[str], optional
            List of filters to apply to the database, by default None. It can be one of
            'bulk', 'surface', 'cluster', 'vacancy' or 'perturb'.
        only_use_base: bool, optional
            If True, only the base structures will be perturbed, by default True in
            order to maintain the perturbation strategy consistent with previous
            versions.
        frac_min : float, optional
            Minimum fraction of the lattice parameter perturbation, by default 0.01.
        repeat : int, optional
            Number of times to apply the perturbation to each structure, by default 1.
        rng_seed : int, optional
            Seed for the random number generator, by default None.
        limit_num_structures : int, optional
            Limit the number of structures to be perturbed, by default None.


        Raises
        ------
        NotImplementedError
            If the perturbation strategy is applied to an unsupported structure type.

        Notes
        -----
        The perturbed structures are then converted to the appropriate type
        (Bulk or Surface) and saved to the database.

        Example
        -------
        >>> initial_db = InitialDatabase()
        >>> initial_db.apply_min_deformation(
        >>>     frac_max=0.05, frac_min=0.01, repeat=5
        >>> )

        """
        # Instantiating RNG
        if not rng_seed:
            rng_seed = np.random.randint(0, ((2**32) - 1))
        rng = np.random.default_rng(seed=rng_seed)

        # Selecting which subset of structures to use by either
        # selecting only the relaxed structures, applying filters
        # and selecting a specific phase or using all structures.
        if only_use_base:
            # Getting all relaxed structures.
            target_entries = self.df.loc[self.df.base]
        elif (not only_use_base) and (filters or use_phase):
            # Filtering structures to perturb.
            target_entries, _, _ = ut.apply_filters_db(self, filters, use_phase)
        else:
            # Using the entire database.
            target_entries = self.df

        # Limiting number of structures
        if limit_num_structures:
            limit_num_structures = min(
                limit_num_structures // repeat, target_entries.shape[0]
            )
            atl_cut.custom_print(
                f'Limiting number of deformations to  {limit_num_structures}', 'debug'
            )
            rng_idxs = rng.choice(
                target_entries.shape[0], size=limit_num_structures, replace=False
            )
            target_entries = target_entries.iloc[rng_idxs]

        # Applying deformation to all perturbed structures
        for _, entry in target_entries.iterrows():
            # Getting some parameters from the current perturb structure.
            str_matid = entry.material_id

            if isinstance(entry.phase, str):
                str_phase = self.phase_diagram.get_phase(entry.phase)
            else:
                str_phase = entry.phase
            curr_str = entry.structure

            # Applying the perturbation 'repeat' times.
            for perturb_repeat_idx in range(repeat):
                # Applying deformation,
                new_struct_perturb = self._apply_min_deformation(
                    structure=curr_str,
                    frac_max=frac_max,
                    frac_min=frac_min,
                )

                mat_str = (
                    f'{entry.unique_id}_{str_matid}_{str_phase.name}_'
                    f'perturb_min_{perturb_repeat_idx + 1}'
                )

                surface_miller = entry.surface_miller if entry.surface_miller else None

                # Creating a new Structure from the perturbed structure
                curr_struct = atl_struct.Structure(
                    material_name=mat_str,
                    structure=new_struct_perturb,
                    material_id=str_matid,
                    phase=str_phase.name,
                    base=False,
                    perturb=False,
                    deformation=True,
                    vacancy=entry.vacancy,
                    targeted_modification=entry.targeted_modification,
                    supercell=entry.supercell,
                    replacement=entry.replacement,
                    formula=entry.formula,
                    symmetry=new_struct_perturb.get_space_group_info()[0],
                    temperature=entry.temperature,
                    calc_performed=False,
                    surface_miller=surface_miller,
                )

                # Converting the structure to the appropiate type
                if entry.bulk:
                    curr_struct_conv = curr_struct.to_bulk()
                elif entry.surface:
                    curr_struct_conv = curr_struct.to_surface()
                elif entry.cluster:
                    curr_struct_conv = curr_struct.to_cluster()
                else:
                    raise NotImplementedError(
                        'This perturbation strategy is not implemented '
                        'for the current structure type.'
                    )

                # Saving the bulk to the db.
                self.df = curr_struct_conv.save_to_db(self.df)

    def _apply_min_deformation(
        self, structure: Structure, frac_max: float = 0.05, frac_min: float = 0.01
    ):
        """Apply a small deformation to the lattice matrix of a structure."""
        # Making a copy of the current structure lattice which can be modified
        perturb_structure = structure.copy()
        matrix = np.copy(perturb_structure.lattice.matrix)

        # Select non-zero indices
        non_zero_mask = np.abs(matrix) > 0.01

        # Compute deformations for all non-zero values
        fraction = (frac_max - frac_min) * np.random.ranf(size=matrix.shape) + frac_min

        # Applying deformation as a mask
        deformations = matrix * fraction
        deformations = np.where(non_zero_mask, deformations, 0)

        # Randomly add or subtract deformations
        signs = np.random.choice([1, -1], size=len(non_zero_mask))

        # Apply deformations
        matrix += signs * deformations

        # Updating perturb_structure with deformed matrix
        perturb_structure.lattice = matrix

        return perturb_structure

    def display_db_ase(self):
        """Display the the structures in the database using `ase.visualize.view`."""
        structures = self.df.structure
        ut._display_indb_dataframe(structures)

    def _get_phase_from_id(self, idx: str) -> str:
        """
        Searches for the corresponding phase in the self.phase_diagram
        dict to a given a material projects id.

        Parameters
        ----------
        idx : str
            Materials project id

        Returns
        -------
        str
            A phase of the phase diagram
        """
        # Creating a list of the phase diagram phase names
        phase_list = [
            phase.name for phase in self.phase_diagram.phases if phase.prototype == idx
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
        with the alloy preparation by replacing all atoms with the base_elem.

        Parameters
        ----------
        structure : Structure
            Prototype structure to be modified

        Returns
        -------
        Structure
            Structure which may have all its atoms replaced by base_elem if it contains
            anything other than the alloy_set atoms.
        """
        # Checking if there are any other atoms than ones in alloy_set in the structure
        alloy_symbols = {str(e) for e in self.phase_diagram.alloy_set}
        if len(set(structure.symbol_set) - alloy_symbols) > 0:
            # Creating a new structure using the base one as a template
            new_structure = structure.copy(sanitize=True)

            # Replacing base_elem atoms in the structures
            for ind in range(len(structure.species)):
                new_structure.replace(
                    ind,
                    # Species(self.phase_diagram.get_phase(phase).base_elem),
                    Element(self.phase_diagram.get_phase(phase).base_elem),
                )

            # Returning new structure with atoms replaced with base_elem
            return new_structure

        else:
            # If everything is already either atoms from alloy_set,
            # leave the structure as is.
            return structure

    def _gather_prototype_structure(
        self,
        prototype: str,
        phase: str,
        get_different_supercells: bool,
        read: bool,
        supercell_max_idx: int,
        num_min_atoms: int = 1,
    ):
        """
        Gather the structure for a prototype from the materials project database,
        while checking that the phase given for the material is correct.

        Parameters
        ----------
        prototype : str
            Materials project id for the prototype structure.
        phase : str
            Name of the phase to be generated.
        get_different_supercells: bool
            Whether to get one or more supercells from the base structure.
        num_min_atoms: int
            Minimum number of atoms in a generated supercell in order to be considered.
        read: bool
            Whether to read the structure from the database or to use the MP API.


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
            are given on the self.phase_diagram dictionary.
            More phases could be added there if necessary.
        """
        # Checking for correct phase input
        if isinstance(phase, atl_pd.Phase):
            phase_name = slugify(phase.name)
        elif isinstance(phase, str):
            phase_name = slugify(phase)

        if not self.phase_diagram.get_phase(phase_name):
            raise KeyError(
                'Wrong phase given. '
                f'Please introduce one of: {[k for k in self.phase_diagram.phases]}'
            )

        # Reading structure from database
        if read:
            atl_cut.custom_print('Using structure from the DB as template...', 'info')
            try:
                if isinstance(phase, atl_pd.Phase):
                    phase_name = phase.name
                query_result = self.df.loc[self.df.phase == phase_name]
                material_id_prefix = query_result.material_id.values[0]
                structure = query_result.structure.values[0]
            except IndexError:
                (
                    query_result,
                    material_id_prefix,
                    structure,
                ) = self.query_mp_api_prototype(prototype)

        # Querying alloy prototype structure
        else:
            query_result, material_id_prefix, structure = self.query_mp_api_prototype(
                prototype
            )

        # TODO: Add a toggle so that the user can choose to use it.
        # Converting all of the atoms from the prototype cell to
        # the base atom type if necessary
        # structure = self._convert_prototype_structure(structure=structure, phase=ph)

        if not isinstance(structure, list):
            structure = [structure]

        report_replacements = True

        for struct_idx, current_struct in enumerate(structure):
            # Getting conventional cell for the replaced structure
            sga = SpacegroupAnalyzer(current_struct)

            current_struct = sga.get_conventional_standard_structure()

            # Create supercells for the replaced structure
            # This can return either 1 or more supercells of the
            # same structure, depending on the 'get_different_supercells' flag.
            structure_list, idx_list, supercells = self._find_supercell_indices(
                current_struct,
                get_different_supercells,
                max_atoms=self.max_num_atoms,
                min_atoms=num_min_atoms,
                initial_supercell_size=supercell_max_idx,
            )

            struct_obj_list = []
            # Saving all the generated supercells as separate bulk structures
            for structure, idxs in zip(structure_list, supercells, strict=False):
                # Ignore small structures
                if len(structure) < num_min_atoms:
                    continue

                # Replacing elements
                if phase.replace_dict:  # noqa: SIM102
                    if phase.replace_dict.get('replace'):
                        replace_with = phase.replace_dict.get(
                            'replace_with', phase.base_elem
                        )

                        replace_dict = {}
                        for element in phase.replace_dict['element_list']:
                            # If "M", all metals should be replaced.
                            # Thus, any metals found in the structure willl
                            # be added in phase.replace_dict["element_list"]
                            if element == 'M':
                                for spec in structure.species:
                                    if spec.is_metal:
                                        phase.replace_dict['element_list'].append(
                                            spec.symbol
                                        )
                                        # Removing repeated keys
                                        phase.replace_dict['element_list'] = list(
                                            set(phase.replace_dict['element_list'])
                                        )

                            else:
                                # Do not replace element with itself
                                if element == replace_with:
                                    continue

                                # Add element to replace_dict
                                replace_dict[element] = str(replace_with)

                        # Removing M key from dict
                        if 'M' in replace_dict:
                            replace_dict.pop('M')

                        if report_replacements:
                            atl_cut.custom_print(
                                (
                                    'Applying substitution to'
                                    ' all base structures from phase.'
                                    f' Using dict: {replace_dict}...'
                                ),
                                'debug',
                            )
                            report_replacements = False

                        structure.replace_species(
                            species_mapping=replace_dict, in_place=True
                        )

                # Getting the supercell vector as a string for naming
                idxs_str = ''.join(map(str, idxs))

                try:
                    bulk_temp = query_result.temperature.values[0]
                except Exception:
                    bulk_temp = np.nan

                try:
                    targeted_modification = query_result.targeted_modification.values[0]
                except Exception:
                    targeted_modification = None

                if isinstance(material_id_prefix, list):
                    curr_mat_id = material_id_prefix[struct_idx]
                else:
                    curr_mat_id = material_id_prefix

                # Creating a new bulk from the supercell
                curr_bulk = atl_struct.Bulk(
                    material_name=f'{curr_mat_id}_{phase_name}_super-{idxs_str}',
                    material_id=curr_mat_id,
                    structure=structure,
                    temperature=bulk_temp,
                    perturb=False,
                    base=False,
                    vacancy=False,
                    targeted_modification=targeted_modification,
                    calc_performed=False,
                    supercell=idxs,
                    phase=phase_name,
                )

                # Saving the bulk to the db.
                self.df = curr_bulk.save_to_db(self.df)

                struct_obj_list.append(curr_bulk)

        return struct_obj_list, query_result, idx_list

    def query_mp_api_prototype(self, prototype):
        from mp_api.client import MPRester

        atl_cut.custom_print('Querying the MP API...', 'debug')
        with MPRester(ut.gather_secrets()['API_KEY'], mute_progress_bars=True) as mpr:
            if isinstance(prototype, list):
                query_result = mpr.summary.search(material_ids=prototype)
            elif isinstance(prototype, str):
                query_result = mpr.summary.search(material_ids=[prototype])[0]

            if isinstance(query_result, list):
                structure = [res.structure for res in query_result]
                material_id_prefix = [res.material_id for res in query_result]
            else:
                structure = query_result.structure
                material_id_prefix = query_result.material_id

        return query_result, material_id_prefix, structure

    @atl_cut.deprecated('Moved to `core.utils`', since_ver='0.13.0')
    def _create_symmetrical_prototype(
        self,
        structure: Structure,
        phase: atl_pd.Phase,
        structure_obj: 'atl_struct.Structure',
    ):
        phase = structure_obj.phase

        if isinstance(phase, str):
            phase = self.phase_diagram.get_phase(phase)

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
            structure=structure,
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

        self.df = final_struct.save_to_db(self.df)

        return structure

    @atl_cut.deprecated('Moved to `core.utils`', since_ver='0.13.0')
    def _gen_base_elem_perc(self, phase, num_struct):
        # Computing base_elem percentages using offset
        if self.use_offset:
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

    @atl_cut.deprecated('Moved to `core.utils`', since_ver='0.13.0')
    def _fit_replacements_phase(
        self,
        phase,
        structure,
        subst_base_elem_perc,
    ):
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

    @atl_cut.deprecated('Moved to `core.utils`', since_ver='0.13.0')
    def _apply_replacement(
        self, structure: Structure, phase, n_target_at: int | float, rng=None
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
        if len(self.phase_diagram.alloy_set) > 1:
            (other_elem,) = self.phase_diagram.alloy_set - {base_elem}
        else:
            other_elem = list(self.phase_diagram.alloy_set)[0]

        # If the structure only has one type of Element, and that is not the base
        # element, this changes with what to replace.
        # if not curr_comp.as_dict().get(base_elem.symbol):
        #     base_elem = str(phase.phase_diagram.element_list[0])
        #     if len(self.phase_diagram.alloy_set) > 1:
        #         (other_elem,) = self.phase_diagram.alloy_set - {base_elem}
        #     else:
        #         other_elem = list(self.phase_diagram.alloy_set)[0]

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
            atl_cut.custom_print(
                (
                    f"No replaceable sites for composition: '{curr_comp}'."
                    "Add one of the formula's elements to the current phase"
                    " 'replacements.element_list'."
                ),
                'error',
            )

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

    def generate_bulk_structures(
        self,
        prototype: str,
        phase: str,
        num_struct: int,
        num_repeats: int,
        get_different_supercells: bool,
        min_num_atoms: int,
        supercell_max_idx: int,
        convert_to_base: bool = True,
        read: bool = True,
        overwrite_read_from_db_list: list = None,
        seed: int = None,
    ):
        """
        Allows to create several variations of a certain phase
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
        overwrite_read_from_db_list: list
            List of structures to read from the db instead of the MP API, will
            ignore the read flag if this is not empty.
        convert_to_base: bool
            Whether to convert all atoms from the structure to the base atom, in order
            to increase randomness when replacing.

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.phase_diagram dictionary.
            More phases could be added there if necessary.
        """
        # Instantiating RNG
        if not seed:
            seed = np.random.randint(0, 2**32 - 1)

        rng = np.random.default_rng(seed=seed)
        atl_cut.custom_print(f'Bulk generation RNG seed: {str(seed)}', 'debug')

        # If the current phase is in overwrite_read_from_db_list,
        # the read flag is set to True
        if overwrite_read_from_db_list:
            curr_phase = phase.name if not isinstance(phase, str) else phase

            if curr_phase in overwrite_read_from_db_list:
                read = True

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
            num_min_atoms=min_num_atoms,
            supercell_max_idx=supercell_max_idx,
        )

        for structure_obj, supr_idx in zip(structure_list, idx_list, strict=False):
            structure = structure_obj.structure

            # Converting all of the atoms from the prototype cell to the
            # base atom type if necessary
            if convert_to_base:
                structure = self._convert_prototype_structure(
                    structure=structure, phase=phase
                )

            # Replacing some atoms using symmetry
            structure = self._create_symmetrical_prototype(
                structure=structure, phase=phase, structure_obj=structure_obj
            )
            # Preparing an array of randomly generated base elem percentages
            # for the new structures
            subst_base_elem_perc = self._gen_base_elem_perc(phase, num_struct)
            atl_cut.custom_print(
                f'Random base element % for bulk to gen: {subst_base_elem_perc * 100}',
                'debug',
            )

            # Choosing the amount of atoms to replace with the base element in the
            # struct which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            # n_at_replacement = [
            #     int(round(structure_len * stct, 0)) for stct in subst_base_elem_perc
            # ]

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            # n_at_replacement_upd is a list which contains the
            # target number of base atoms in the new structure.
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
                    supercell_vec_str = atl_surf.get_miller_index_str(
                        structure_obj.supercell
                    )

                    try:
                        bulk_temp = query_result.temperature.values[0]
                    except Exception:
                        bulk_temp = np.nan

                    mat_name = (
                        f'{structure_obj.material_id}_{phase.name}_super-'
                        f'{supercell_vec_str}-{supr_idx}_replacement-'
                        f'{str_ind + 1}-{repl + 1}'
                    )

                    # Creating a new Bulk object for the structure with replacement
                    new_struct_symm = atl_struct.Bulk(
                        material_name=mat_name,
                        material_id=structure_obj.material_id,
                        structure=new_structure,
                        temperature=bulk_temp,
                        targeted_modification=structure_obj.targeted_modification,
                        perturb=False,
                        replacement=True,
                        replacement_ind=(str_ind + 1, repl + 1),
                        base=False,
                        calc_performed=False,
                        supercell=structure_obj.supercell,
                        phase=phase.name,
                    )

                    self.df = new_struct_symm.save_to_db(self.df)

    def get_base_structs_current_phase(self, phase):
        # Getting all of the base structures
        base_structs = self.df.loc[self.df.base]

        if not isinstance(phase, str):
            phase = phase.name

        # Getting the structures corresponding to the current phase
        phase_mask = base_structs.phase == phase

        base_structs = base_structs.where(phase_mask, other=pd.NA)
        base_structs.dropna(how='all', inplace=True)

        return base_structs

    def _get_main_elem_perc(self, phase: atl_pd.Phase, structure):
        """
        Get the percentage of the main element for a given structure.

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

        perc = main_cnt / total_atoms
        return perc

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
        unwanted_attrs = [
            'save_to_db',
            'from_vasprun',
            'from_atl_structure',
            'to_bulk',
            'to_surface',
            'to_cluster',
            'from_db_row',
            'to_ase_atoms',
            'from_ase_atoms',
        ]

        # If given structure is a pymatgen Structure
        if isinstance(structure, Structure):
            new_row = pd.Series(
                {
                    'material_id': material_id,
                    'structure': structure,
                    'temperature': None,
                    'perturb': True,
                    'phase': phase.name,
                    'base': base,
                    'formula': structure.formula,
                    **extra,
                }
            )
        # If given structure is a ATLAS structure.
        elif isinstance(structure, atl_struct.Structure):
            attr_list = [
                att
                for att in dir(structure)
                if not att.startswith('_') and att not in unwanted_attrs
            ]
            att_dict = {att: getattr(structure, att) for att in attr_list}

            # Normalize phase to string for consistent DataFrame comparisons
            if 'phase' in att_dict and hasattr(att_dict['phase'], 'name'):
                att_dict['phase'] = att_dict['phase'].name

            new_row = pd.Series(att_dict)

        new_row_df = new_row.to_frame().T
        new_row_df = new_row_df.astype(
            {
                'perturb': bool,
                'base': bool,
                'bulk': bool,
                'surface': bool,
                'cluster': bool,
                'calc_performed': bool,
                'replacement': bool,
                'deformation': bool,
                'vacancy': bool,
            }
        )

        self.df = self.df.astype(
            {
                'perturb': bool,
                'base': bool,
                'bulk': bool,
                'surface': bool,
                'cluster': bool,
                'calc_performed': bool,
                'replacement': bool,
                'vacancy': bool,
                'deformation': bool,
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
        name = node.label + '_aiida-uuid_' + node.uuid

        # Writing the vasprun.xml file to a buffer.
        retrieved = node.outputs.retrieved
        vasprun_f = retrieved.get_object_content('vasprun.xml', 'rb')
        buffer = BytesIO(vasprun_f)

        # Reading the file from the buffer and closing it
        vasprun = aseio.read(buffer, format='vasp-xml', index='-1')
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
            'name': name,
            'lattice': lattice,
            'positions': structure,
            'symbols': symbols,
            'pot_energy': pot_energy,
            'charge': charge,
            'forces': forces,
        }

        return data_dict

    def _add_entry_to_n2p2_input(self, buffer: TextIOWrapper, data_dict: dict):
        # Writing begin keyword and structure name
        buffer.write('begin\n')
        buffer.write(f'comment {data_dict.get("material_name", "no name found")}\n')

        # Getting lattice parameters and converting them to Bohr
        lat_x = data_dict['lattice'][0]
        lat_y = data_dict['lattice'][1]
        lat_z = data_dict['lattice'][2]

        # Writing lattice parameters
        buffer.write(f'lattice {lat_x[0]:.6f} {lat_x[1]:.6f} {lat_x[2]:.6f}\n')
        buffer.write(f'lattice {lat_y[0]:.6f} {lat_y[1]:.6f} {lat_y[2]:.6f}\n')
        buffer.write(f'lattice {lat_z[0]:.6f} {lat_z[1]:.6f} {lat_z[2]:.6f}\n')

        # Writing information for every atom. Every atom line must contain:
        # atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1>
        for idx, (at, frc) in enumerate(
            zip(data_dict['positions'], data_dict['forces'], strict=False)
        ):
            # Preparing and writing the line
            buffer.write(
                f'atom {at[0]:.6f} {at[1]:.6f}'
                f' {at[2]:.6f}'
                f' {data_dict["symbols"][idx]} {0:.6f} {0:.6f}'
                f' {frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n'
            )

        # writing potential energy and charge
        buffer.write(f'energy {data_dict["pot_energy"]:.8f}\n')
        buffer.write(f'charge {data_dict["charge"]:.6f}\n')

        # writing end keyword
        buffer.write('end\n')

    def generate_n2p2_input_aiida(
        self, aiida_group_list: list, filter_dict: dict, path: str = None
    ):
        # Loading aiida profile
        load_profile()

        # Handling path
        path = pathlib.Path(path) if path and isinstance(path, str) else pathlib.Path()

        # Adding input.data filename to path
        path = path / 'input.data'

        # Gathering nodes from the given group
        atl_cut.custom_print('Getting nodes...')

        # Preparing a query in the aiida db
        qb = orm.QueryBuilder()

        for group in aiida_group_list:
            qb.append(orm.Group, filters={'label': group}, tag='group')
            (qb.append(orm.WorkChainNode, with_group='group', filters=filter_dict),)

        result_nodes = qb.all(flat=True)

        if len(result_nodes) == 0:
            for _group in aiida_group_list:
                qb.append(VaspCalculation, with_group='group', filters=filter_dict)
            result_nodes = qb.all(flat=True)

        atl_cut.custom_print(f'{len(result_nodes)} nodes found.', 'info')

        # Writing the file
        with open(path, 'w') as curr_f:
            # Checking every node
            for node in riprg.track(
                result_nodes, description=' [ ⧖ ]  Writing info...'
            ):
                # Gathering the information from each node
                data_dict = self._gather_n2p2_reqdata_from_node(node=node)

                # Writing the information to the buffer
                self._add_entry_to_n2p2_input(buffer=curr_f, data_dict=data_dict)
        atl_cut.custom_print(f"All calculations saved in '{path}'.", 'done')

    def remove_structs_out_of_atom_count_range(
        self, min_num_atoms: int, max_num_atoms: int, remove_base=False
    ):
        remove_count = 0
        for row, struct in self.df.iterrows():
            # Do not remove structures marked as base.
            if struct.base and not remove_base:
                continue

            # Removing structures that are outside of the size range
            if (
                len(struct.structure.species) < min_num_atoms
                or len(struct.structure.species) > max_num_atoms
            ):
                self.df.drop(row, inplace=True)
                remove_count += 1
        return remove_count

    def remove_structs_out_of_cell_size_range(
        self, min_cell_size: float, remove_base=False
    ):
        remove_count = 0
        for row, struct in self.df.iterrows():
            structure_obj = struct.structure

            # Do not remove structures marked as base.
            if struct.base and not remove_base:
                continue

            if isinstance(structure_obj, Structure):
                try:
                    structure_obj = AseAtomsAdaptor().get_atoms(
                        structure_obj, msonable=False
                    )
                except TypeError as e:
                    atl_cut.custom_print(
                        f"Row {row} resulted in TypeError: '{e}'. Removing.", 'warning'
                    )
                    self.df.drop(row, inplace=True)
                    continue

            if any(structure_obj.cell.cellpar()[:3] < min_cell_size):
                self.df.drop(row, inplace=True)
                remove_count += 1

        return remove_count

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
            phases_to_use = self.phase_diagram.phases
            num_phases = len(self.phase_diagram.phases)

        # Maximum number of structures per phase
        max_struct_phase = structure_limit // num_phases

        # Getting the current phase as a Phase object
        # If the phase is not found, omitting.
        for curr_phase in phases_to_use:
            try:
                curr_phase = self.phase_diagram.get_phase(curr_phase)
            except atl_exc.PhaseNotFound:
                continue

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
        phase: 'atl_pd.Phase' = None,
        num_struct: int = 2,
        num_repeat: int = 2,
        method: str = 'wulff',
        basin_hopping: bool = False,
        bh_totalsteps: int = 50,
        bh_fmax: float = 0.05,
        lj_sigma: float | None = None,
        lj_epsilon: float = 1.0,
    ):
        if phase is None:
            phase = self.phase_diagram.get_phase('alpha')

        # Generate a list of atl_struct.Cluster
        cluster_list = []

        # Adding a dimer
        if add_dimer:
            atl_cut.custom_print('Adding base dimers...', 'debug')
            clust_obj = atl_clust.make_clean_dimer(self, phase=phase)
            cluster_list.append(clust_obj)
            atl_cut.custom_print('Base dimers done...', 'debug')

        # Create clusters over all size range given, from smallest to largest.
        atl_cut.custom_print(
            f'Adding base clusters with n_atoms: {size_range[0]}-{size_range[-1]}...',
            'debug',
        )
        for size in riprg.track(size_range, description='Base clusters...'):
            clust_obj = atl_clust.make_clean_cluster(
                self,
                size=size,
                phase=phase,
                method=method,
                basin_hopping=basin_hopping,
                bh_totalsteps=bh_totalsteps,
                bh_fmax=bh_fmax,
                lj_sigma=lj_sigma,
                lj_epsilon=lj_epsilon,
            )
            cluster_list.append(clust_obj)
        atl_cut.custom_print('Base clusters done...', 'debug')

        # If True, store the structures along with their information
        # into the ATLAS InitialDatabase object
        if save_in_db:
            for _idx, cluster in enumerate(cluster_list):
                self._save_row(structure=cluster)

        # Return the cluster list in case the user just wants the clusters
        # but not storing them into the database.
        atl_cut.custom_print(f'Generated {len(cluster_list)} base clusters.', 'debug')
        return cluster_list

    def plot_database_composition(
        self,
        temperature_K: float = 273.0,
        rc_params: dict = None,
        fig_path: str | pl.Path = '.',
        fig_name: str = 'database_composition',
        # fig_format: str = 'png',
        max_phases_pie: int = 6,
    ):
        # Updating matplotlib rcParams
        for key, value in rc_params.items():
            mpl.rcParams[key] = value

        inner = [['bar1'], ['pie']]
        outer = [
            ['histogram', 'bar2'],
            ['main', inner],
        ]

        gridspec_kw = {'height_ratios': [1, 4], 'width_ratios': [8, 1]}
        fig, axd = plt.subplot_mosaic(
            outer,
            gridspec_kw=gridspec_kw,
            figsize=(16, 7),
        )

        hist_t_ax = axd['histogram']
        main_plot_ax = axd['main']
        bar_chart_ax = axd['bar2']
        bar_chart_ax_2 = axd['bar1']
        pie_chart_ax = axd['pie']
        # empty_axis = axd['ignore']
        # empty_axis.set_axis_off()

        # Getting base element to use in x-axis
        base_elem = self.phase_diagram.base_elem

        plot_dict = {
            'bulk': {'structs': [], 'color': '#458588', 'temperature_K': []},
            'base': {'structs': [], 'color': '#076678', 'temperature_K': []},
            'surface': {'structs': [], 'color': '#fe8019', 'temperature_K': []},
            'cluster': {'structs': [], 'color': '#d3869b', 'temperature_K': []},
            'perturb': {'structs': [], 'color': '#d79921', 'temperature_K': []},
            'vacancy': {'structs': [], 'color': '#689d6a', 'temperature_K': []},
            'md': {'structs': [], 'color': '#915ad3', 'temperature_K': []},
            'deformation': {'structs': [], 'color': '#b16286', 'temperature_K': []},
            'oct_perturb': {'structs': [], 'color': '#665c54', 'temperature_K': []},
            'isolated_atom': {'structs': [], 'color': '#365c54', 'temperature_K': []},
            'unknown': {'structs': [], 'color': '#ee0000', 'temperature_K': []},
            'none': {'structs': [], 'color': '#757779', 'temperature_K': []},
        }

        # Get base element composition for every structure in the database
        all_T = []
        for _, row in self.df.iterrows():
            curr_comp = row.structure.composition
            curr_frac = curr_comp.get_atomic_fraction(base_elem.symbol)

            curr_temp = row.structure.properties.get('md_temperature', temperature_K)

            if row.bulk:
                plot_dict['bulk']['structs'].append(curr_frac * 100)
                plot_dict['bulk']['temperature_K'].append(curr_temp)
            elif row.surface:
                plot_dict['surface']['structs'].append(curr_frac * 100)
                plot_dict['surface']['temperature_K'].append(curr_temp)
            elif row.cluster:
                plot_dict['cluster']['structs'].append(curr_frac * 100)
                plot_dict['cluster']['temperature_K'].append(curr_temp)
            else:
                plot_dict['unknown']['structs'].append(curr_frac * 100)
                plot_dict['unknown']['temperature_K'].append(curr_temp)

            all_T.append(float(curr_temp))

        plot_dict['bulk']['structs'] = np.array(plot_dict['bulk']['structs'])
        plot_dict['surface']['structs'] = np.array(plot_dict['surface']['structs'])
        plot_dict['cluster']['structs'] = np.array(plot_dict['cluster']['structs'])

        for key, type_dict in plot_dict.items():
            struct_comp_base_elem = type_dict['structs']
            temps_K = type_dict['temperature_K']

            if len(struct_comp_base_elem) == 0:
                continue
            # The y-axis is generated at a fixed T.
            # temps_K = np.ones_like(struct_comp_base_elem) + (temperature_K - 1)

            # Plotting compositions
            main_plot_ax.scatter(
                struct_comp_base_elem,
                temps_K,
                color=type_dict['color'],
                label=key,
                edgecolor='black',
                marker='o',
                s=55,
                alpha=0.7,
                linewidth=0.25,
                zorder=10,
            )

            # Top histogram
            # now determine histogram limits by hand:
            binwidth = 1
            xymax = max(np.max(np.abs(struct_comp_base_elem)), np.max(temps_K))

            lim = (int(xymax / binwidth) + 1) * binwidth
            bins = np.arange(-lim, lim + binwidth, binwidth)

            # Top histogram
            hist_t_ax.sharex(main_plot_ax)
            hist_t_ax.hist(
                struct_comp_base_elem,
                color=type_dict['color'],
                alpha=0.35,
                bins=bins,
                label=key,
            )
            hist_t_ax.tick_params(axis='x', labelbottom=False)

        delta_T = max(all_T) - min(all_T)
        if delta_T == 0:
            delta_T = 100

        main_plot_ax.set_ylim(
            bottom=min(all_T) - (delta_T * 0.10),
            top=max(all_T) + (delta_T * 0.10),
        )

        # Plotting the phase diagram of the database
        main_plot_ax = self.phase_diagram.plot_diagram(
            rc_params=rc_params,
            show_plot=False,
            ax=main_plot_ax,
            max_temp_K=max(all_T) + (delta_T * 0.10),
            min_temp_K=min(all_T) - (delta_T * 0.10),
        )

        db_report: dict = self.gen_report()

        # Removing empty keys from the pie chart
        keys_to_pop = [
            key
            for key in db_report['structure_count']
            if db_report['structure_count'][key] == 0
        ]
        for key in keys_to_pop:
            db_report['structure_count'].pop(key)

        def autopct_format(values):
            def custom_format(pct):
                total = db_report['database_settings']['total_entries']
                val = int(round(pct * total / 100.0))
                return f'{pct:.1f}%\n({val:d})'

            return custom_format

        # Plotting bar chart (top)
        top_dict = db_report['structure_count']['by_modification'].copy()
        y_pos_bar = range(len(top_dict.keys()))
        bar_chart_ax.barh(
            y=y_pos_bar,
            width=top_dict.values(),
            color=[plot_dict[key]['color'] for key in top_dict],
            alpha=0.3,
        )
        bar_chart_ax.set_yticks(y_pos_bar)
        bar_chart_ax.set_yticklabels(list(top_dict.keys()))

        # Writing labels in barchart
        label_x = max(top_dict.values()) * 0.10
        for bar, amount in zip(bar_chart_ax.patches, top_dict.values(), strict=False):
            bar_chart_ax.text(
                x=label_x,
                y=bar.get_y() + bar.get_height() / 2,
                s=amount,
                color='#282828',
                ha='left',
                va='center',
                fontsize='small',
            )

        # Plotting bar chart with modifications (bottom)
        bottom_dict = db_report['structure_count']['by_type'].copy()
        y_pos_bar = range(len(bottom_dict.keys()))
        bar_chart_ax_2.barh(
            y=y_pos_bar,
            width=bottom_dict.values(),
            color=[plot_dict[key]['color'] for key in bottom_dict],
            alpha=0.3,
        )
        bar_chart_ax_2.set_yticks(y_pos_bar)
        bar_chart_ax_2.set_yticklabels(list(bottom_dict.keys()))

        # Writing labels in barchart
        label_x = max(bottom_dict.values()) * 0.10
        for bar, amount in zip(
            bar_chart_ax_2.patches, bottom_dict.values(), strict=False
        ):
            bar_chart_ax_2.text(
                x=label_x,
                y=bar.get_y() + bar.get_height() / 2,
                s=amount,
                color='#282828',
                ha='left',
                va='center',
                fontsize='small',
            )

        # Creating a shortened version of the dict
        short_phase_dict = {}
        if len(db_report['phases']) > max_phases_pie:
            phase_count = 0
            for phase in db_report['phases']:
                # Remove Isolated atom from the pie chart
                if phase == 'IsolatedAtom':
                    continue
                elif phase_count < max_phases_pie:
                    short_phase_dict[phase] = db_report['phases'][phase]
                    phase_count += 1
                else:
                    short_phase_dict['other'] = (
                        short_phase_dict.get('other', 0) + db_report['phases'][phase]
                    )
        else:
            short_phase_dict = db_report['phases']

        # Pie chart for phases
        phase_color_list = plt.cm.viridis(np.linspace(0, 1, len(short_phase_dict)))
        pie_chart_ax.pie(
            short_phase_dict.values(),
            labels=short_phase_dict.keys(),
            colors=phase_color_list,
            autopct=autopct_format(db_report),
            startangle=90,
            wedgeprops=dict(width=0.95, alpha=0.3),
            radius=1.25,
            textprops={'size': '8'},
        )

        hist_t_ax.set_title('Database composition')
        hist_t_ax.spines['top'].set_visible(False)
        hist_t_ax.spines['right'].set_visible(False)
        if main_plot_ax:
            main_plot_ax.set_title('')
        else:
            atl_cut.custom_print(
                'Incorrect main plot axis found! '
                "Maybe there's something wrong with the phase diagram?",
                'warning',
            )
        hist_t_ax.legend()
        fig.tight_layout()

        # Saving the figure
        fig_name = 'comp_plot_' + fig_name
        chart_img_path = pl.Path(fig_path) / fig_name
        plt.savefig(chart_img_path.with_suffix('.png'), dpi=300, format='png')
        plt.savefig(chart_img_path.with_suffix('.svg'), dpi=300, format='svg')

        atl_cut.custom_print(
            f"Database composition plot saved in '{chart_img_path}'.",
            'done',
        )

        # Displaying the plot
        plt.show()

        # Clearing the plot
        plt.clf()

    def get_soap_descriptors(self, **kwargs) -> np.ndarray:
        species_list = [Element(el).Z for el in self.phase_diagram.alloy_set]

        r_cut = 6
        n_max = 8
        l_max = 6

        soap_structs = {}

        # Average of the inner SOAP vectors
        # This will return a single feature vector: (1, n_features)
        soap = SOAP(
            species=species_list,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average='inner',
            sparse=False,
        )

        for _, row in self.df.iterrows():
            pym_struct = row.structure
            unique_id = row.unique_id
            soap_structs[unique_id] = {}
            ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

            # Create output for multiple system in parallel
            struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=False)

            # Adding structure to dictionary using unique_id as key
            soap_structs[unique_id]['descriptors'] = struct_soap

        return soap_structs

    def descriptors_concave_hull(
        self,
        descriptor_type: str = 'soap',
        dimensionality_reduction_method: str = 'autoencoder',
        descriptor_settings: dict = None,
        load_autoencoder_path: str = None,
        rng_seed: int = None,
        device: str = None,
        plot_filename: str = None,
        boundary_method: str = 'concave_hull',
        morph_disk_size: int = 10,
        morph_threshold: int = 250,
        morph_dpi: int = 100,
    ):
        import torch  # noqa

        from atlas.active_learning.extrapolation import concave_hull as atl_ch
        from atlas.active_learning.extrapolation import (
            morphological_closing as atl_morph,
        )
        from atlas.active_learning.extrapolation import (
            train_autoencoder as atl_tr_ae,
        )
        from atlas.active_learning.extrapolation.autoencoder import AutoencoderSettings

        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not descriptor_settings:
            descriptor_settings = {}

        # Get the descriptors using the selected descriptor type
        match descriptor_type:
            case 'soap':
                atl_cut.custom_print('Generating descriptors using SOAP...', 'info')
                # Get the SOAP descriptors
                descriptors: dict = self.get_soap_descriptors(**descriptor_settings)
            case 'mace':
                # Get the MACE descriptors
                raise NotImplementedError(
                    'MACE descriptors are not yet implementedin database generation.'
                )
                atl_cut.custom_print('Generating descriptors using MACE...', 'info')
            case 'acsf':
                # Get the ACSF descriptors
                raise NotImplementedError(
                    'ASCF descriptors are not yet implementedin database generation'
                )
        atl_cut.custom_print('Generated descriptors.', 'done')

        # Reduce dimensionality of the descriptors
        match (load_autoencoder_path, dimensionality_reduction_method):
            case (None, 'autoencoder'):
                descriptor_arr = np.vstack(
                    [val['descriptors'] for key, val in descriptors.items()]
                )
                model = atl_tr_ae.run_training(
                    AutoencoderSettings(
                        rng_seed=rng_seed,
                        device=device,
                        dtype='float32',
                        dataset=descriptor_arr,
                        num_epochs=250,
                        model_path='./autoencoder_model.pth',
                    )
                )

            case (_, 'autoencoder'):
                model = torch.load(load_autoencoder_path, weights_only=False)

        match dimensionality_reduction_method:
            case 'pca':
                raise NotImplementedError(
                    'PCA is not yet implemented in database generation.'
                )
            case 'autoencoder':
                from atlas.active_learning.extrapolation import (
                    autoencoder as atl_ae,
                )

                descriptors = atl_ae.get_latent_space_autoencoder(
                    model=model,
                    descriptor_dict=descriptors,
                    device=device,
                )

        # Get boundary of the reduced descriptors
        latent_space = np.vstack(
            [val['latent_space'] for key, val in descriptors.items()]
        )
        unique_ids = np.array(list(descriptors.keys()), dtype=str)
        npz_path = pl.Path(plot_filename).with_name('latent_space.npz')

        match boundary_method:
            case 'concave_hull':
                atl_cut.custom_print('Getting concave hull...', 'info')
                concave_hull, alpha = atl_ch.get_concave_hull_python(latent_space)

                np.savez(
                    npz_path,
                    latent_space=latent_space,
                    unique_ids=unique_ids,
                    boundary_method=np.array(['concave_hull']),
                    hull_vertices=concave_hull,
                    alpha=np.array([alpha]),
                )
                atl_cut.custom_print(f'Saved latent space data to {npz_path}', 'done')

                atl_ch.plot_concave_hull(
                    concave_hull=concave_hull,
                    latent_space=latent_space,
                    filename=plot_filename,
                    alpha=alpha,
                )

            case 'morphological_closing':
                atl_cut.custom_print(
                    'Getting morphological closing boundary...', 'info'
                )
                results = atl_morph.process_morphological_closing(
                    latent_space[:, 0],
                    latent_space[:, 1],
                    disk_size=morph_disk_size,
                    threshold=morph_threshold,
                    dpi=morph_dpi,
                )
                boundaries = results['data_boundaries']

                if boundaries:
                    all_coords = np.vstack(boundaries)
                    counts = np.array([len(b) for b in boundaries], dtype=np.int64)
                else:
                    all_coords = np.empty((0, 2), dtype=np.float64)
                    counts = np.array([], dtype=np.int64)

                np.savez(
                    npz_path,
                    latent_space=latent_space,
                    unique_ids=unique_ids,
                    boundary_method=np.array(['morphological_closing']),
                    boundary_coords=all_coords,
                    boundary_counts=counts,
                    morph_disk_size=np.array([morph_disk_size]),
                )
                atl_cut.custom_print(f'Saved latent space data to {npz_path}', 'done')

                _plot_morphological_closing(
                    latent_space, boundaries, morph_disk_size, plot_filename
                )


def _plot_morphological_closing(latent_space, boundaries, disk_size, filename):
    """Save a boundary overlay plot for morphological closing results."""
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        latent_space[:, 0],
        latent_space[:, 1],
        s=4,
        c='steelblue',
        alpha=0.6,
        edgecolors='none',
        rasterized=True,
        zorder=1,
    )

    total_area = 0.0
    for boundary in boundaries:
        boundary = np.atleast_2d(boundary)
        if len(boundary) < 3:
            continue
        try:  # noqa
            total_area += Polygon(boundary).area
        except Exception:
            pass
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            '-',
            color='#cc241d',
            lw=2,
            zorder=2,
        )

    ax.text(
        0.03,
        0.97,
        f'Disk size: {disk_size}\nTotal area: {total_area:.2e}\n'
        f'Boundaries: {len(boundaries)}\nStructures: {len(latent_space):,}',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        zorder=4,
    )
    ax.set_xlabel('Embedded dimension 1')
    ax.set_ylabel('Embedded dimension 2')
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    atl_cut.custom_print(f'Boundary plot saved to {filename}', 'done')


# def estimate_final_struct_number(
#     selected_phases, structures, gen_dict, db_dict, config_dict, composition_dict
# ):
#     # Get the number of structures to generate
#     composition_dict = db_dict.get("size")

#     # Multiply number of selected phases by the number of structures per phase


def cli_gen_db_report(database_path: pl.Path | str):
    """
    Generate a report of the database.

    Parameters
    ----------
    database_path : str | Path
        Path to the database.

    Returns
    -------
    dict
        Report of the database.
    """
    # Get logger
    atl_cut.init_logger(source='db_report')
    atl_cut.custom_print('Generating database report...', 'info')

    # Initialize the database
    structures = indb.InitialDatabase(
        create_db=False,
    )
    structures: InitialDatabase = structures.load_database(database_path=database_path)

    # Print report on screen
    get_database_report(structures=structures)

    params = {}

    # Generating and saving the composition plot
    structures.plot_database_composition(
        temperature_K=300,
        rc_params=params,
        fig_path='.',
        fig_name=structures.database_name,
        # fig_format=db_dict['plot_db'].get('format', 'png'),
    )

    return structures.gen_report()


def cli_run_gen_initial_database(
    db_path: str | pl.Path,
    db_dict: dict,
    phase_diagram_dict: dict,
    gen_dict: dict,
    selected_phases,
    config_dict: dict,
):
    # Get timestamp for the entire run
    timestamp = int(time.time())

    # If database_path is not given, the current directory is used.
    if not db_path:
        db_dict['database_path'] = pl.Path.cwd()

    composition_dict = db_dict.get('composition')
    if not composition_dict:
        raise ValueError(
            'Composition dictionary not found. '
            'Add the `database.composition` key in the .toml file.'
        )

    ratios_dict = composition_dict.get('ratios', {})
    bulk_r = ratios_dict.get('bulk', 0)
    surf_r = ratios_dict.get('surface', 0)
    clst_r = ratios_dict.get('cluster', 0)
    ratio_sum = bulk_r + surf_r + clst_r

    if ratio_sum != 1:
        raise ValueError(
            'Sum of ratios in the composition dictionary must be 1. '
            f'Currently: {ratio_sum}. Ommitted types will be considered as 0.'
        )

    # Start logger
    log_path = pl.Path(db_path) / 'logs'
    if not log_path.exists():
        log_path.mkdir(parents=True)
    logger, _ = atl_cut.init_logger(source=pl.Path(__file__).stem, log_path=log_path)

    # Checking last version of the library
    atl_cut.check_atl_version(logger=logger)

    # Check if all required sections are present
    from atlas.core.command_line.command_line_utils import validate_config_file

    validate_config_file(config_dict=config_dict, config_type='database_generation')

    # Get timestamp for the entire run
    timestamp = int(time.time())

    overwrite_db = db_dict.get('overwrite_db', False)
    db_path_exists = pl.Path(db_dict['database_path']).exists()

    # Avoiding database overwrite
    if not overwrite_db and db_path_exists:
        # Get timestamp based on host and current time
        # db_dict["database_path"] = pl.Path(db_path) / f"gen_db_{timestamp}"
        db_dict['database_name'] = db_dict['database_name'] + f'_{timestamp}'
        atl_cut.custom_print(
            (
                'Overwriting disabled. Creating new database in'
                f" {db_dict['database_path']} named '{db_dict['database_name']}'."
            ),
            'warning',
        )

    # If db_path is not given, the current directory is used.
    if not db_path:
        db_path = pl.Path.cwd()

    atl_cut.custom_print('Starting generation of initial database...', 'info')
    print()

    # Get seed from input file or generate one
    rng_seed = int(db_dict.get('rng_seed', np.random.randint(0, 2**32 - 1)))
    atl_cut.custom_print(f"Using RNG seed: '{rng_seed}'.", 'info')

    # Assemble phase diagram
    phase_diagram = atl_pd.PhaseDiagram(
        material=phase_diagram_dict['material_name'],
        base_elem=phase_diagram_dict['base_element'],
        element_list=phase_diagram_dict['element_list'],
    )

    # Create phase diagram
    phases_list = []
    for curr_phase_name, phase_d in phase_diagram_dict['phase'].items():
        # Query the materials project for the space group number
        spacegroup = None
        symbol = None

        try:
            from mp_api.client import MPRester

            with MPRester(
                ut.gather_secrets()['API_KEY'], mute_progress_bars=True
            ) as mpr:
                query_result = mpr.summary.search(material_ids=phase_d['prototype'])[0]
                spacegroup = query_result.symmetry.number
                symbol = query_result.symmetry.symbol
        except TypeError:
            pass

        curr_phase = atl_pd.Phase(
            name=curr_phase_name,
            element_list=phase_diagram_dict['element_list'],
            cluster_elem=phase_d.get('cluster_element'),
            composition=phase_d['composition'],
            spacegroup=spacegroup,
            symbol=symbol,
            prototype=phase_d['prototype'],
            offset=float(phase_d.get('offset', 0)),
            replace_dict=phase_d.get('replacements'),
            allow_modifications=phase_d.get('allow_modifications', True),
            phase_diagram=phase_diagram,
            use_cache=phase_d.get('use_cache', False),
        )
        phases_list.append(curr_phase)

    for phase in phases_list:
        phase_diagram.add_phase(phase)

    # Initialize the database
    structures = indb.InitialDatabase(
        database_name=db_dict['database_name'],
        database_path=db_dict['database_path'],
        max_num_atoms=int(db_dict['max_num_atoms']),
        phase_diagram=phase_diagram,
        load_db=False,
    )

    read_from_db = True
    if db_dict.get('relax_struct_path'):
        # Initial structures obtained with DFT relaxation are loaded from a given path
        structures.read_base_structures(
            path=db_dict['relax_struct_path'],
            target_structures=selected_phases,
        )
    else:
        # Obtain structures from Materials Project
        structures.gather_base_structures(phase_diag_phases=phase_diagram.phases)
        read_from_db = False

    # Applying central_atom_octahedral perturbation to specific structures
    phases_read_from_db = []
    target_mod_dict = config_dict.get('targeted_modification', {})
    if target_mod_dict.get('central_atom_octahedral'):
        cen_at_oh_dict = target_mod_dict['central_atom_octahedral']
        atl_cut.custom_print(
            'Applying central atom octahedral modifications...', 'info'
        )
        ut.apply_central_atom_octahedral(
            db_obj=structures,
            filter_struct_types=cen_at_oh_dict['filter_struct_types'],
            filter_phase_list=cen_at_oh_dict['filter_phases'],
            num_repeats=int(cen_at_oh_dict['num_repeats']),
            central_element=cen_at_oh_dict['central_element'],
            limit_num_structures=int(cen_at_oh_dict['limit_max_num_modifications']),
            seed=rng_seed,
            max_perturbation_ang=float(cen_at_oh_dict.get('max_perturbation_ang', 0.2)),
        )
        phases_read_from_db.extend(cen_at_oh_dict['filter_phases'])
        output_db_status(structures)

    atl_cut.custom_print('Generating structures from initial structures...', 'debug')

    # TODO: Implement this function
    # estimate_final_struct_number(
    #     selected_phases=selected_phases,
    #     structures=structures,
    #     gen_dict=gen_dict,
    #     db_dict=db_dict,
    #     config_dict=config_dict,
    #     composition_dict=composition_dict,
    # )
    # quit()

    # Fair cluster budget distribution across phases
    _cluster_budget_remaining = None
    if 'cluster' in gen_dict and 'cluster' in gen_dict.get('generate_type', []):
        _cluster_limit_global = gen_dict['cluster'].get('limit_max_num_structures')
        if _cluster_limit_global:
            _cluster_budget_remaining = int(_cluster_limit_global)

    # Global target from database.composition.size, used to auto-derive
    # per-phase limits when no explicit limit_max_num_structures is set.
    _total_target = int(composition_dict.get('size', 0)) or None

    for phase_idx, phase in enumerate(selected_phases):
        # Line break for aesthetic purposes
        print()

        # Creating surfaces from the base structures, generating
        # different supercells and applying replacements.
        atl_cut.custom_print(
            (
                f'[bold][{phase_idx + 1}/{len(selected_phases)}] '
                f'- Current phase: {phase}[/]'
            ),
            'info',
        )

        # Getting phase object
        phase = phase_diagram.get_phase(phase)

        # If modifications are not allowed, don't do anything for the
        # current phase.
        if not phase.allow_modifications:
            atl_cut.custom_print(
                f"Modifications are not allowed for phase: '{phase.name}'. Skipping.",
                'warn',
            )
            continue

        if 'bulk' in gen_dict and 'bulk' in gen_dict.get('generate_type', []):
            step_name = 'bulk generation'
            atl_cut.custom_print(
                f"Step '{step_name}' - Generating bulk structures...", 'info'
            )
            ini_n_structs = len(structures)

            # Generating bulk structures.
            structures.generate_bulk_structures(
                prototype=phase.prototype,
                phase=phase,
                num_struct=int(gen_dict['bulk']['num_struct']),
                num_repeats=int(gen_dict['bulk']['num_repeat']),
                get_different_supercells=True,
                min_num_atoms=int(db_dict['min_num_atoms']),
                supercell_max_idx=int(gen_dict['bulk']['supercell_max_idx']),
                read=read_from_db,
                overwrite_read_from_db_list=phases_read_from_db,
                convert_to_base=False,
                seed=rng_seed,
            )

            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)

        if 'surface' in gen_dict and 'surface' in gen_dict.get('generate_type', []):
            step_name = 'surface generation'
            atl_cut.custom_print(
                f'Step {step_name} - Generating surface structures...', 'info'
            )
            ini_n_structs = len(structures)

            # Generating surface structures.
            atl_surf.gen_surfaces_diff_miller_parallel(
                db_obj=structures,
                phase=phase,
                min_num_atoms=int(db_dict['min_num_atoms']),
                overwrite_max_num_atoms=int(db_dict['max_num_atoms']),
                min_miller_index=int(gen_dict['surface']['min_miller_index']),
                max_miller_index=int(gen_dict['surface']['max_miller_index']),
                min_slab_size=float(gen_dict['surface']['min_slab_size_ang']),
                min_vacuum_size=float(gen_dict['surface']['min_vacuum_size_ang']),
                get_supercells=gen_dict['surface']['get_supercells'],
                fixed_layers=int(gen_dict['surface']['fixed_layers']),
                num_replacements=int(gen_dict['surface']['num_replacements']),
                num_repeat_replace=int(gen_dict['surface']['num_repeat_replace']),
                limit_total_num_struct=int(
                    gen_dict['surface']['max_number_supercells']
                ),
                frac_slabs_save=gen_dict['surface'].get('frac_slabs_save', 0.1),
                frac_supercells_save=gen_dict['surface'].get(
                    'frac_supercells_save', 0.1
                ),
                save_in_db=gen_dict['surface']['save_in_db'],
                max_slab_num=int(gen_dict['surface']['max_slab_num']),
                n_workers=int(
                    gen_dict['surface'].get('n_workers', max(cpu_count() - 1, 1))
                ),
            )
            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)
            # output_db_status(structures)

        if 'cluster' in gen_dict and 'cluster' in gen_dict.get('generate_type', []):
            step_name = 'cluster generation'
            atl_cut.custom_print(
                f'Step {step_name} - Generating cluster structures...', 'info'
            )
            ini_n_structs = len(structures)

            # Per-phase cluster budget: divide remaining budget among remaining phases
            _phase_cluster_start = len(structures)
            if _cluster_budget_remaining is not None:
                _num_remaining = len(selected_phases) - phase_idx
                _per_phase_ceiling = max(1, _cluster_budget_remaining // _num_remaining)
                cluster_limit = _phase_cluster_start + _per_phase_ceiling
            else:
                cluster_limit = None

            # Generating base clusters
            structures.generate_clusters(
                size_range=list(int(s) for s in gen_dict['cluster']['size_range']),
                add_dimer=gen_dict['cluster'].get('add_dimer', False),
                save_in_db=gen_dict['cluster'].get('save_in_db', True),
                get_replacements=False,
                get_perturbed=False,
                phase=phase,
                num_struct=int(gen_dict['cluster'].get('num_struct', 2)),
                num_repeat=int(gen_dict['cluster'].get('num_repeat', 2)),
                method=gen_dict['cluster'].get('cluster_method', 'wulff'),
                basin_hopping=gen_dict['cluster'].get('basin_hopping', False),
                bh_totalsteps=int(gen_dict['cluster'].get('bh_totalsteps', 50)),
                bh_fmax=float(gen_dict['cluster'].get('bh_fmax', 0.05)),
                lj_sigma=float(gen_dict['cluster']['lj_sigma'])
                if 'lj_sigma' in gen_dict['cluster']
                else None,
                lj_epsilon=float(gen_dict['cluster'].get('lj_epsilon', 1.0)),
            )

            # Check limit after base cluster generation
            if cluster_limit and len(structures) >= cluster_limit:
                atl_cut.custom_print(
                    f'Cluster limit reached ({len(structures)} >= {cluster_limit}). '
                    'Skipping replacements and perturbations.',
                    'warning',
                )
            else:
                # Generating replacement clusters
                if gen_dict['cluster'].get('get_replacements', False):
                    atl_cut.custom_print('Generating cluster replacements...', 'info')
                    _remaining = (
                        (cluster_limit - len(structures)) if cluster_limit else None
                    )
                    atl_clust.apply_replacement_cluster_db(
                        db_obj=structures,
                        phase=phase,
                        num_struct=int(gen_dict['cluster'].get('num_struct', 2)),
                        num_repeat=int(gen_dict['cluster'].get('num_repeat', 2)),
                        similarity_check=True,
                        save_in_db=gen_dict['cluster'].get('save_in_db', True),
                        max_structures=_remaining,
                    )

                # Check limit after replacements
                if cluster_limit and len(structures) >= cluster_limit:
                    atl_cut.custom_print(
                        f'Cluster limit reached ({len(structures)} >= {cluster_limit}).'
                        ' Skipping perturbations.',
                        'warning',
                    )
                else:
                    # Generating perturbed clusters
                    if gen_dict['cluster'].get('get_perturbed', False):
                        atl_cut.custom_print('Generating perturbed clusters...', 'info')
                        _remaining = (
                            (cluster_limit - len(structures)) if cluster_limit else None
                        )
                        atl_clust.apply_gauss_perturb_db(
                            repeat=int(gen_dict['cluster'].get('num_repeat', 2)),
                            db_obj=structures,
                            center=float(
                                gen_dict['cluster'].get('perturbation_ang', 0.04)
                            ),
                            max_structures=_remaining,
                            phase=phase,
                        )

            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)

            # Update remaining cluster budget
            if _cluster_budget_remaining is not None:
                _phase_cluster_used = len(structures) - _phase_cluster_start
                _cluster_budget_remaining = max(
                    0, _cluster_budget_remaining - _phase_cluster_used
                )

        # Filter small and large structures
        remove_count = structures.remove_structs_out_of_atom_count_range(
            min_num_atoms=int(db_dict['min_num_atoms']),
            max_num_atoms=int(db_dict['max_num_atoms']),
        )
        atl_cut.custom_print(
            f'Removed {remove_count} structures out of atom count range.', 'info'
        )
        output_db_status(structures)

        # Lattice deformation
        if 'deformation' in config_dict:
            displ_dict = config_dict['deformation']
            step_name = 'lattice deformation'
            atl_cut.custom_print(
                f'Step {step_name} - Applying deformations to lattices.', 'info'
            )
            ini_n_structs = len(structures)

            structures.apply_min_deformation(
                frac_max=float(displ_dict['lattice_frac_deform_max']),
                frac_min=float(displ_dict['lattice_frac_deform_min']),
                repeat=int(displ_dict['num_repeats']),
                use_phase=phase,
                only_use_base=False,
                limit_num_structures=int(displ_dict['limit_max_num_deformations']),
                filters=displ_dict.get('filter_struct_types'),
                rng_seed=rng_seed,
            )

            remove_count = structures.remove_structs_out_of_cell_size_range(
                min_cell_size=float(db_dict['min_cell_size'])
            )
            atl_cut.custom_print(
                f'Removed {remove_count} structures out of cell size range.', 'info'
            )
            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)

        if 'perturbation' in config_dict:
            step_name = 'random perturbation'
            perturb_dict = config_dict['perturbation']
            atl_cut.custom_print(
                f'Step {step_name} - Applying a random perturbation'
                ' to the structures...',
                'info',
            )
            ini_n_structs = len(structures)

            ut.apply_gauss_perturb_db(
                db_obj=structures,
                repeat=int(perturb_dict.get('num_repeats', 1)),
                filters=perturb_dict.get('filter_struct_types', []),
                phase=phase,
                center=perturb_dict.get('perturbation_ang', 0.04),
                limit_num_structures=int(
                    perturb_dict.get('limit_max_num_perturbs', 100)
                ),
            )

            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)

        # Applying vacancies to a random subset of structures
        if 'vacancies' in config_dict:
            step_name = 'vacancy generation'
            vacancies_dict = config_dict['vacancies']
            atl_cut.custom_print(
                (
                    f'Step - {step_name} - Applying vacancies to a random subset of '
                    f'{vacancies_dict["limit_max_num_vacancies"]} structures...'
                ),
                'info',
            )
            ini_n_structs = len(structures)

            structures.apply_vacancies_random(
                max_vac_perc=vacancies_dict['max_vacancy_percentage'],
                min_vac_perc=vacancies_dict['min_vacancy_percentage'],
                filters=vacancies_dict['filter_struct_types'],
                lim_num_struc=int(vacancies_dict['limit_max_num_vacancies']),
                repeat=int(vacancies_dict['num_repeats']),
                seed=rng_seed,
                element_list=vacancies_dict['element_list'],
                phase=phase,
            )
            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)

        # Limiting structures for current phase
        lim_phas_structs = phase_diagram_dict['phase'][phase.original_name].get(
            'limit_max_num_structures'
        )
        if not lim_phas_structs and _total_target:
            n_phases = len(selected_phases)
            lim_phas_structs = max(1, _total_target // n_phases)
        if lim_phas_structs:
            step_name = 'limiting structures'
            atl_cut.custom_print(
                (
                    f'Step - {step_name} - Limiting number of structures '
                    f"from phase '{phase.name}' to {lim_phas_structs}."
                ),
                'info',
            )
            ini_n_structs = len(structures)
            _stratify = phase_diagram_dict['phase'][phase.original_name].get(
                'stratify_by_size', False
            )
            structures = ut.limit_num_structures_phase(
                structures,
                phase,
                lim_phas_structs,
                rng_seed,
                stratify_by_size=_stratify,
            )
            added_structs = len(structures) - ini_n_structs
            report_completed_step(phase, structures, added_structs, step_name=step_name)

    print()
    atl_cut.custom_print(
        'Finishing populating structures from every phase.',
        'done',
    )
    print()

    if 'adsorbates' in config_dict:
        step_name = 'adsorbate generation'
        atl_cut.custom_print(
            f'Step {step_name} - Adding adsorbates on top of the structures...',
            'info',
        )
        ini_n_structs = len(structures)
        adsorb_dict = config_dict['adsorbates']

        ut.add_adsorbates(
            db_obj=structures,
            repeat=int(adsorb_dict['num_repeats']),
            filters=adsorb_dict['filter_struct_types'],
            phase=structures.df.phase.unique(),
            limit_num_structures=int(adsorb_dict['limit_max_num_perturbs']),
            adsorbate_species=adsorb_dict['adsorbate_species'],
        )

        added_structs = len(structures) - ini_n_structs
        report_completed_step(phase, structures, added_structs, step_name=step_name)

    # Run short MD simulations for some unperturbed structures
    # to get some MD information.
    if 'md_gen' in config_dict:
        atl_cut.custom_print(
            f'Step {step_name} - Running MD on base structures...',
            'info',
        )
        step_name = 'MD generation'
        md_gen_dict: dict = config_dict['md_gen']
        ini_n_structs = len(structures)

        structures.gen_init_md_frames(
            phase=structures.df.phase.unique(),
            filters=md_gen_dict.get('filter_struct_types'),
            lim_num_struc=int(md_gen_dict['max_num_frames']),
            seed=rng_seed,
            md_gen_dict=md_gen_dict,
        )

        added_structs = len(structures) - ini_n_structs
        report_completed_step(phase, structures, added_structs, step_name=step_name)

    print()
    if 'struct_filters' in config_dict:
        step_name = 'applying filters'
        atl_cut.custom_print(
            f'Step {step_name} - Applying user-defined filters to structures...',
            'info',
        )
        ini_n_structs = len(structures)

        # Apply user-defined filters to the structures
        filter_results = apply_struct_filters_atl_db(structures, config_dict)
        total_removed = len(filter_results['filtered_uuids'])
        atl_cut.custom_print(
            f'Removed {total_removed} structures based on user-defined filters.',
            'info',
        )
        if total_removed > 0:
            bp = dict(filter_results['by_phase'])
            bt = dict(filter_results['by_type'])
            bm = dict(filter_results['by_modification'])
            bf = dict(filter_results['by_filter'])
            atl_cut.custom_print(f'  By phase: {bp}', 'info')
            atl_cut.custom_print(f'  By type: {bt}', 'info')
            atl_cut.custom_print(f'  By modification: {bm}', 'info')
            atl_cut.custom_print(f'  By filter: {bf}', 'info')
            structures._removal_history['filtering'].update(filter_results)
        added_structs = len(structures) - ini_n_structs
        report_completed_step(phase, structures, added_structs, step_name=step_name)
    # Add function to check all atom type in database and
    # # create a structure for each atom type that contains a single atom
    # # in a vaccuum, and it is labelled with `config_type=IsolatedAtom`.
    structures.add_single_atoms()

    atl_cut.custom_print('Database generation complete!', 'done')
    print()

    structures.save_database(
        path=db_dict['database_path'],
    )

    # Generating report with database composition
    # Print using rich
    get_database_report(structures)

    # Plot the database if requested
    if db_dict.get('plot_db', {}).get('show'):
        atl_cut.custom_print('Plotting database composition...', 'info')

        # Selecting plot style settings for the plot
        params = db_dict['plot_db'].get('rc_params')

        # Generating and saving the composition plot
        structures.plot_database_composition(
            temperature_K=400,
            rc_params=params,
            fig_path=db_dict['database_path'],
            fig_name=db_dict['database_name'],
            # fig_format=db_dict['plot_db'].get('format', 'png'),
        )

    # Getting the concave hull if requested
    if config_dict.get('concave_hull', {}).get('gen_concave_hull', False):
        concave_dict = config_dict['concave_hull']
        boundary_method = concave_dict.get('boundary_method', 'concave_hull')
        print()
        atl_cut.custom_print(
            f'Generating boundary ({boundary_method})...',
            'info',
        )
        structures.descriptors_concave_hull(
            descriptor_type=concave_dict.get('descriptor', 'soap').lower(),
            dimensionality_reduction_method=concave_dict.get(
                'dim_reduction', 'autoencoder'
            ),
            plot_filename=concave_dict.get(
                'plot_filename',
                f'{pl.Path(db_path) / "descriptors_concave_hull.png"}',
            ),
            device=concave_dict.get('device', 'cpu'),
            rng_seed=rng_seed,
            boundary_method=boundary_method,
            morph_disk_size=concave_dict.get('morph_disk_size', 10),
            morph_threshold=concave_dict.get('morph_threshold', 250),
            morph_dpi=concave_dict.get('morph_dpi', 100),
        )
        atl_cut.custom_print('Boundary generated!', 'done')
        print()

    # Export the database if requested
    if db_dict.get('export', {}).get('export'):
        export_path = db_dict['export'].get('file_path', db_dict['database_path'])
        file_name = db_dict['export'].get('file_name', db_dict['database_name'])

        if not file_name:
            file_name = db_dict['database_name']
        if not export_path:
            export_path = db_dict['database_path']

        out_format = db_dict['export'].get('format')

        atl_cut.custom_print(f"Exporting database as '{out_format}'...", 'info')

        structures.export_db(
            out_format=out_format,
            file_path=export_path,
            file_name=file_name + f'_structures_{timestamp}',
        )

    # Display the database if requested
    if db_dict.get('show_db_ase', {}).get('show'):
        atl_cut.custom_print('Displaying database in ASE...', 'info')
        structures.display_db_ase()


def report_completed_step(
    phase: atl_pd.Phase,
    structures: InitialDatabase,
    added_structs: int,
    step_name: str = 'unknown',
):
    """
    Report the completion of a step in the database generation process.

    Parameters
    ----------
    phase : atl_pd.Phase
        The phase for which the step was completed.
    structures : InitialDatabase
        The database object containing the structures.
    added_structs : int
        The number of structures added in this step.
    step_name : str, optional
        The name of the step that was completed, by default 'unknown'.
    """
    atl_cut.custom_print(
        f"Step {step_name} complete for phase '{phase.name}'.", 'debug'
    )
    atl_cut.custom_print(f'Structures added: {added_structs}', 'debug')
    atl_cut.custom_print(f'Current DB total: {len(structures)}', 'debug')
    db_report = structures.gen_report()
    atl_cut.custom_print(
        f'Current DB breakdown: {db_report["structure_count"]["by_type"]}', 'debug'
    )
    atl_cut.custom_print(
        f'Modifications applied in DB (cumulative count): '
        f'{db_report["structure_count"]["by_modification"]}',
        'debug',
    )


def get_database_report(structures: InitialDatabase):
    report = structures.gen_report()
    pretty = Pretty(report)
    panel = Panel(pretty, title='Database report')
    rprint(panel)
    atl_cut.custom_print(f'Database report generated:\n{report}', 'debug')
    return report


def output_db_status(database: InitialDatabase):
    shape = database.get_db_shape()
    atl_cut.custom_print(f'DB Info: {shape[0]} entries, {shape[1]} fields', 'info')


def get_struct_idx_from_uuid(
    database: InitialDatabase | Atoms | list[Atoms], uuid: str
) -> int | None:
    match_idx = None
    if isinstance(database, InitialDatabase):
        match_idx = None
    else:
        for idx, struct in enumerate(database):
            if struct.info.get('atl_id') == uuid:
                return idx
    return match_idx


def get_step_to_structure_mapping(
    database: Atoms | list[Atoms] | InitialDatabase,
) -> dict:
    if isinstance(database, Atoms):
        database = [database]

    structure_mapping = {}
    for struct in database:
        curr_step = struct.info.get('atl_al_step')
        if curr_step not in structure_mapping:
            structure_mapping[curr_step] = []

        structure_mapping[curr_step].append(struct.info.get('atl_id'))

    return structure_mapping
