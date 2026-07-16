"""Module for the Structure class."""

import inspect  # noqa
import uuid

import pandas as pd
import pymatgen.io.vasp as vasp
from pymatgen.core.units import Energy
from pymatgen.core.structure import Structure as pmg_struct
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import pathlib as pl
import warnings

from atlas.core import initial_db as atl_indb


class Structure:
    """
    Wrapper around a pymatgen structure object with extra fields for ATLForge.

    Parameters
    ----------
    structure : pymatgen.core.Structure, optional
        A pymatgen Structure object representing the atomic structure, by default None
    material_name : str, optional
        The name of the material, by default None
    material_id : str or int, optional
        The unique identifier for the material, by default None
    phase : str, optional
        The phase of the material (e.g., solid, liquid, gas), by default None
    base : bool, optional
        Flag indicating if this is a base structure, by default False
    isolated_atom : bool, optional
        Flag indicating if this is an isolated_atom structure, by default False
    perturb : bool, optional
        Flag indicating if the structure should be perturbed, by default False
    supercell : tuple of int, optional
        The dimensions of the supercell, specified as a tuple of integers
        (e.g., (2, 2, 2)), by default None
    surface : bool, optional
        Flag indicating if this is a surface structure, by default False
    bulk : bool, optional
        Flag indicating if this is a bulk structure, by default False
    cluster : bool, optional
        Flag indicating if this is a cluster structure, by default False
    vacancy: bool, optional
        Flag indicating if the structure contains a vacancy, by default False
    formula : str, optional
        The chemical formula of the material, by default None
    replacement : bool, optional
        Flag indicating if an atomic replacement should be performed, by default False
    replacement_ind : int, optional
        The index of the atom to be replaced, by default None
    symmetry : str, optional
        The symmetry information of the structure, by default None
    energy_per_atom : float, optional
        The energy per atom of the structure, by default None
    temperature : float, optional
        The temperature at which the properties are calculated, by default None
    magnetic_properties : dict, optional
        A dictionary containing magnetic properties of the structure, by default None
    calc_energy_per_atom : float, optional
        The calculated energy per atom from a computational method, by default None
    calc_energy_toten : float, optional
        The total energy from a computational method, by default None
    calc_energy : float, optional
        The calculated energy of the structure, by default None
    calc_performed : bool, optional
        Flag indicating if a calculation has been performed, by default False
    calc_type : str, optional
        The type of calculation performed (e.g., DFT, MD), by default None
    calc_output : dict, optional
        The output of the calculation, by default None
    surface_miller : tuple of int, optional
        The Miller indices of the surface, specified as a tuple of integers,
        by default None
    """

    def __init__(
        self,
        structure=None,
        material_name: str = None,
        material_id=None,
        phase=None,
        base: bool = None,
        isolated_atom: bool = False,
        perturb: bool = None,
        deformation: bool = None,
        supercell=None,
        surface: bool = False,
        bulk: bool = False,
        cluster: bool = False,
        formula=None,
        replacement: bool = False,
        replacement_ind=None,
        vacancy: bool = False,
        symmetry=None,
        energy_per_atom=None,
        temperature: float = None,
        magnetic_properties=None,
        calc_energy_per_atom=None,
        calc_energy_toten=None,
        calc_energy=None,
        calc_performed=False,
        calc_type=None,
        calc_output=None,
        surface_miller=None,
        targeted_modification: str = None,
        al_loop_step: int = 0,
        unique_id=None,
        init_md: bool = False,
    ):
        if unique_id:
            self.unique_id = unique_id
        else:
            self.unique_id = str(uuid.uuid4())
        self.material_name = material_name
        if isinstance(structure, (pmg_struct, type(None))):
            self.structure = structure
        else:
            raise TypeError(
                'Given structure is not a pymatgen structure, '
                f'it is {type(structure)} instead'
            )
        self.material_id = material_id
        self.phase = phase
        self.base = base
        self.isolated_atom = isolated_atom
        self.perturb = perturb
        self.supercell = supercell
        self.deformation = deformation
        self.surface = surface
        self.surface_miller = surface_miller
        self.bulk = bulk
        self.replacement = replacement
        self.replacement_ind = replacement_ind
        self.cluster = cluster

        # This should be the material's project db energies
        self.energy_per_atom = energy_per_atom

        if not formula and structure:
            formula = structure.formula
        self.formula = formula

        self.symmetry = symmetry
        self.temperature = temperature
        self.magnetic_properties = magnetic_properties

        # Everything prefixed with calc_ should come from a DFT
        # calculation, and not the materials project db
        self.calc_energy_per_atom = calc_energy_per_atom
        self.calc_energy_toten = calc_energy_toten
        self.calc_energy = calc_energy
        self.calc_performed = calc_performed
        self.calc_type = calc_type
        self.calc_output = calc_output
        self.vacancy = vacancy
        self.targeted_modification = targeted_modification
        self.al_loop_step = al_loop_step
        self.init_md = init_md

    def to_bulk(self):
        """Create a Bulk instance by passing the current Structure attributes."""
        attributes = {
            name: value
            for name, value in inspect.getmembers(self)
            if not inspect.isroutine(value)
            and not name.startswith('__')
            and name not in ['to_surface', 'to_bulk', 'to_cluster', 'to_ase_atoms']
        }
        return Bulk(**attributes)

    def to_surface(self):
        """Create a surface instance by passing the current Structure attributes."""
        attributes = {
            name: value
            for name, value in inspect.getmembers(self)
            if not inspect.isroutine(value)
            and not name.startswith('__')
            and name not in ['to_surface', 'to_bulk', 'to_cluster', 'to_ase_atoms']
        }
        return Surface(**attributes)

    def to_cluster(self):
        """Create a cluster instance by passing the current Structure attributes."""
        attributes = {
            name: value
            for name, value in inspect.getmembers(self)
            if not inspect.isroutine(value)
            and not name.startswith('__')
            and name not in ['to_surface', 'to_bulk', 'to_cluster', 'to_ase_atoms']
        }
        return Cluster(**attributes)

    @classmethod
    def from_ase_atoms(cls, ase_atoms: Atoms):
        """
        Creates a Structure object from an ASE Atoms object.

        Parameters
        ----------
        ase_atom : ase.Atoms
            The ASE Atoms object to convert (must have .info dictionary populated).

        Returns
        -------
        Structure
            The resulting Structure object.
        """
        # Convert ASE atoms to Pymatgen structure
        # We use the same adaptor class as used in db_struct_to_ase
        pmg_structure = AseAtomsAdaptor().get_structure(ase_atoms)

        # Extract info dictionary for easier access
        info = ase_atoms.info

        # Determine isolated_atom flag from the stored structure type
        is_isolated = info.get('atl_struct_type') == 'isolated_atom'

        return cls(
            structure=pmg_structure,
            unique_id=info.get('atl_id'),
            material_name=info.get('struct_name', 'unknown'),
            phase=info.get('phase'),
            perturb=info.get('perturb', False),
            replacement=info.get('replacement', False),
            base=info.get('base', False),
            bulk=info.get('bulk', False),
            cluster=info.get('cluster', False),
            surface=info.get('surface', False),
            surface_miller=info.get('surface_miller'),
            supercell=info.get('supercell'),
            symmetry=info.get('symmetry'),
            calc_type=info.get('calc_type'),
            calc_performed=info.get('calc_performed', False),
            deformation=info.get('deformation', False),
            vacancy=info.get('vacancy', False),
            init_md=info.get('init_md', False),
            targeted_modification=info.get('targeted_modification', False),
            isolated_atom=is_isolated,
        )

    def from_vasprun(
        self,
        vasprun: vasp.Vasprun | str | pl.Path,
        **kwargs,
    ):
        """
        Create a Structure object by parsing a `vasprun.xml` file.

        The structure will contain the energy information and the structure
        information from the vasprun fil

        Parameters
        ----------
        vasprun : vasp.Vasprun | str | pl.Path
            The vasprun file containing information for a DFT calculation.

        Returns
        -------
        Structure
            Structure object containing the energy and structure information.
        """
        # Load the vasprun file if it is a string or Path object
        if isinstance(vasprun, (pl.Path, str)):
            vasprun = vasp.Vasprun(vasprun)

        # Getting the structure
        structure = vasprun.structures[-1]

        # Getting energy information
        energy = vasprun.final_energy
        energy_toten = Energy(
            float(vasprun.ionic_steps[-1]['e_fr_energy']),
            'eV',
        )
        energy_per_atom = energy / len(structure.species)

        # Getting the temperature from the vasp parameters
        # If the temperature is not set, the vasprun shows 0.0001K as T.
        # I round to the third decimal place so this value then equals to 0.
        if float(vasprun.parameters['TEBEG']) <= 1e-4:
            temperature = 0
        else:
            temperature = float(vasprun.parameters['TEBEG'])

        generated_structure = Structure(
            energy_per_atom=None,
            material_name=kwargs.get('material_name'),
            structure=structure,
            material_id=kwargs.get('material_id'),
            phase=kwargs.get('phase'),
            base=kwargs.get('base'),
            isolated_atom=kwargs.get('isolated_atom'),
            perturb=kwargs.get('perturb'),
            supercell=kwargs.get('supercell'),
            surface=kwargs.get('surface'),
            bulk=kwargs.get('bulk'),
            cluster=kwargs.get('cluster'),
            replacement=kwargs.get('replacement'),
            formula=structure.formula,
            symmetry=structure.get_space_group_info(),
            temperature=temperature,
            magnetic_properties=vasprun.projected_magnetisation,
            calc_energy=energy,
            calc_energy_per_atom=energy_per_atom,
            calc_energy_toten=energy_toten,
            calc_performed=True,
            calc_type=vasprun.run_type,
            calc_output=vasprun,
            init_md=False,
        )

        if not generated_structure.replacement:
            generated_structure.replacement = False

        return generated_structure

    def __repr__(self):
        repr_str = ''
        spc = ' ' * 2

        # Gathering name information
        if self.material_name:
            repr_str += f'ATLAS {self.__class__.__name__}: {self.material_name}\n'
        else:
            repr_str += f'ATLAS {self.__class__.__name__}: (no name)\n'

        repr_str += f'{spc}ID: {self.unique_id}\n'

        # Gathing formula and phase
        if self.formula:
            repr_str += f'{spc}Formula: {self.formula}\n'
        if self.phase:
            repr_str += f'{spc}{self.phase}\n'
        else:
            repr_str += f'{spc}Phase: unknown phase\n'

        repr_str += f'{spc}Status flags: '
        # Gathering if the structure is a base or structure phase
        props = []
        # Gathering the type of structure
        if self.bulk:
            props.append('bulk')
        elif self.surface:
            props.append('surface')
        elif self.cluster:
            props.append('cluster')

        # Gathering extra properties
        if self.base:
            props.append('+base')
        if self.replacement:
            props.append('+replacements')
        if self.supercell:
            props.append('+supercell')
        if self.perturb:
            props.append('+atom_positions_perturbed')
        if self.deformation:
            props.append('+lattice_displaced')
        if self.vacancy:
            props.append('+vacancies')
        if self.targeted_modification:
            props.append(f'+{self.targeted_modification}')
        if self.init_md:
            props.append('+initial_md')

        repr_str += ' '.join(props)

        # Gathering DFT data
        if self.calc_performed:
            repr_str += '\n'
            repr_str += f'{spc}Obtained with DFT {self.calc_type} calculation:\n'
            repr_str += f'\t - Energy {self.calc_energy}\n'
            repr_str += f'\t - Free energy {self.calc_energy_toten}\n'
            repr_str += f'\t - Energy per atom: {self.calc_energy_per_atom}'
        else:
            repr_str += '\n'
            repr_str += f'{spc}No calc performed. Energy data unavailable.'

        return repr_str

    # def __dict__(self):
    #     dict_obj = {}
    #     att_names = [att for att in dir(self) if not att.startswith('__')]
    #     for att in att_names:
    #         dict_obj[att] = self.getattr(att)
    #     return dict_obj

    def save_to_db(self, db_obj):
        if isinstance(self.phase, str):
            phase = self.phase
        elif self.phase is not None:
            phase = self.phase.name
        else:
            phase = None
        new_row = pd.Series(
            {
                'material_id': str(self.material_id),
                'structure': self.structure,
                'temperature': self.temperature,
                'perturb': self.perturb,
                'formula': self.formula,
                'symmetry': self.symmetry,
                'base': self.base,
                'isolated_atom': self.isolated_atom,
                'surface': self.surface,
                'surface_miller': self.surface_miller,
                'phase': phase,
                'magnetic_properties': self.magnetic_properties,
                'energy_per_atom': None,
                'unique_id': self.unique_id,
                'material_name': self.material_name,
                'replacement': self.replacement,
                'replacement_ind': self.replacement_ind,
                'supercell': self.supercell,
                'bulk': self.bulk,
                'cluster': self.cluster,
                'calc_energy': self.calc_energy,
                'calc_energy_per_atom': self.calc_energy_per_atom,
                'calc_energy_toten': self.calc_energy_toten,
                'calc_performed': self.calc_performed,
                'calc_type': self.calc_type,
                'calc_output': self.calc_output,
                'vacancy': self.vacancy,
                'targeted_modification': self.targeted_modification,
                'deformation': self.deformation,
                'al_loop_step': self.al_loop_step,
                'init_md': self.init_md,
            }
        )
        bool_columns = {
            'perturb': bool,
            'deformation': bool,
            'base': bool,
            'isolated_atom': bool,
            'bulk': bool,
            'surface': bool,
            'cluster': bool,
            'calc_performed': bool,
            'replacement': bool,
            'vacancy': bool,
            'init_md': bool,
        }
        new_row = new_row.to_frame().T.astype(bool_columns)

        is_InitialDatabase = isinstance(db_obj, atl_indb.InitialDatabase)

        struct_df = db_obj.df if is_InitialDatabase else db_obj

        # Adding a new row to the database results in a FutureWarning
        # if some columns are empty. This is to be expected in this version
        # of the code, so we suppress the warning until it is gone.
        with warnings.catch_warnings(category=FutureWarning):
            warnings.simplefilter('ignore')

            # Adding the new row to the database
            if struct_df.shape[0] == 0 or struct_df.shape[1] == 0:
                struct_df = new_row
            else:
                with pd.option_context('future.no_silent_downcasting', True):
                    struct_df = struct_df.fillna(value=False).infer_objects(copy=False)
                struct_df = struct_df.astype(bool_columns)
                struct_df = pd.concat([struct_df, new_row], ignore_index=True)

        if is_InitialDatabase:
            db_obj.df = struct_df
        else:
            db_obj = struct_df

        return db_obj

    def from_db_row(self, row: pd.Series, columns: list):
        for col_idx, col in enumerate(columns):
            setattr(self, col, row.iloc[col_idx])
        return self

    def to_ase_atoms(self):
        # Convert pymatgen to ase
        ase_struct = AseAtomsAdaptor().get_atoms(self.structure)
        ase_struct.info = self.__dict__

        return ase_struct


class Bulk(Structure):
    """Class for bulk structures."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setting the bulk property as True.
        self.bulk = True
        self.surface = False
        self.cluster = False


class Surface(Structure):
    """Class for slab structures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting the surface property as True.
        self.surface = True
        self.cluster = False
        self.bulk = False

        # Converting surface miller to a list of integers
        if isinstance(self.surface_miller, str):
            self.surface_miller = [int(idx) for idx in self.surface_miller.split()]


class Cluster(Structure):
    """Class for cluster structures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting the cluster property as True.
        self.cluster = True
        self.surface = False
        self.bulk = False


class IsolatedAtom(Structure):
    """Class for cluster structures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting the isolated atom property as True.
        self.cluster = False
        self.surface = False
        self.bulk = False
        self.isolated_atom = True
