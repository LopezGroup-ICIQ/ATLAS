import uuid

import pandas as pd
import pymatgen.io.vasp as vasp
import numpy as np
from pymatgen.core.units import Energy

from MatDBForge.core import initial_db as mdb_indb


class Structure:
    def __init__(
        self,
        structure=None,
        material_name: str = None,
        material_id=None,
        phase=None,
        base: bool = None,
        perturb: bool = None,
        supercell=None,
        surface: bool = False,
        bulk: bool = False,
        cluster: bool = False,
        formula=None,
        replacement: bool = False,
        replacement_ind=None,
        symmetry=None,
        # This should be the material's project db energies
        energy_per_atom=None,
        temperature: float = None,
        magnetic_properties=None,
        # Everything prefixed with calc_ should come from a DFT
        # calculation, and not the materials project db
        calc_energy_per_atom=None,
        calc_energy_toten=None,
        calc_energy=None,
        calc_performed=False,
        calc_type=None,
        calc_output=None,
        surface_miller=None,
    ):
        self.unique_id = uuid.uuid4()
        self.material_name = material_name
        self.structure = structure
        self.material_id = material_id
        self.phase = phase
        self.base = base
        self.perturb = perturb
        self.supercell = supercell
        self.surface = surface
        self.surface_miller = surface_miller
        self.bulk = bulk
        self.replacement = replacement
        self.replacement_ind = replacement_ind
        self.cluster = cluster
        self.energy_per_atom = energy_per_atom

        if not formula and structure:
            formula = structure.formula
        self.formula = formula

        self.symmetry = symmetry
        self.temperature = temperature
        self.magnetic_properties = magnetic_properties
        self.calc_energy_per_atom = calc_energy_per_atom
        self.calc_energy_toten = calc_energy_toten
        self.calc_energy = calc_energy
        self.calc_performed = calc_performed
        self.calc_type = calc_type
        self.calc_output = calc_output

    def from_vasprun(
        self,
        vasprun: vasp.Vasprun,
        **kwargs,
    ):
        # Getting the structure
        structure = vasprun.structures[-1]

        # Getting energy information
        energy = vasprun.final_energy
        energy_toten = Energy(
            float(vasprun.ionic_steps[-1]["e_fr_energy"]),
            "eV",
        )
        energy_per_atom = energy / len(structure.species)

        # Getting the temperature from the vasp parameters
        # If the temperature is not set, the vasprun shows 0.0001K as T.
        # I round to the third decimal place so this value then equals to 0.
        if float(vasprun.parameters["TEBEG"]) <= 1e-4:
            temperature = 0
        else:
            temperature = float(vasprun.parameters["TEBEG"])

        generated_structure = Structure(
            energy_per_atom=None,
            material_name=kwargs.get("material_name"),
            structure=structure,
            material_id=kwargs.get("material_id"),
            phase=kwargs.get("phase"),
            base=kwargs.get("base"),
            perturb=kwargs.get("perturb"),
            supercell=kwargs.get("supercell"),
            surface=kwargs.get("surface"),
            bulk=kwargs.get("bulk"),
            cluster=kwargs.get("cluster"),
            replacement=kwargs.get("replacement"),
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
        )

        if not generated_structure.replacement:
            generated_structure.replacement = False

        return generated_structure

    def __repr__(self):
        repr_str = ""

        # Gathering name information
        if self.material_name:
            repr_str += f"MatDBForge {self.__class__.__name__}"
            repr_str += " - "
            repr_str += f"{self.material_name}\n"
            repr_str += f"{self.unique_id}\n"
        else:
            repr_str += f"MatDBForge Structure - {self.unique_id} (no name)\n"

        # Gathing formula and phase
        if self.formula:
            repr_str += f"{self.formula} - "
        if self.phase:
            repr_str += f"{self.phase}\n"
        else:
            repr_str += "unknown phase\n"

        # Gathering if the structure is a base or structure phase
        props = []
        if self.base:
            props.append("Base")
        elif self.perturb:
            props.append("Perturbed")

        # Gathering the type of structure
        if self.bulk:
            props.append("Bulk")
        elif self.surface:
            props.append("Surface")
        elif self.cluster:
            props.append("Cluster")

        repr_str += " ".join(props)

        repr_str += "\n"

        # Gathering DFT data
        if self.calc_performed:
            repr_str += f"Obtained with DFT {self.calc_type} calculation:\n"
            repr_str += f"\t - Energy {self.calc_energy}\n"
            repr_str += f"\t - Free energy {self.calc_energy_toten}\n"
            repr_str += f"\t - Energy per atom: {self.calc_energy_per_atom}"

        return repr_str

    # def __dict__(self):
    #     dict_obj = {}
    #     att_names = [att for att in dir(self) if not att.startswith('__')]
    #     for att in att_names:
    #         dict_obj[att] = self.getattr(att)
    #     return dict_obj

    def save_to_db(self, db_obj):
        bool_columns = {
            # "replacement": np.bool_,
            "perturb": np.bool_,
            "base": np.bool_,
            "bulk": np.bool_,
            "surface": np.bool_,
            "cluster": np.bool_,
            "calc_performed": np.bool_,
        }

        data_dict = {
            "material_id": str(self.material_id),
            "structure": self.structure,
            "temperature": self.temperature,
            "perturb": self.perturb,
            "formula": self.formula,
            "symmetry": self.symmetry,
            "base": self.base,
            "surface": self.surface,
            "surface_miller": self.surface_miller,
            "phase": self.phase,
            "magnetic_properties": self.magnetic_properties,
            "energy_per_atom": None,
            "unique_id": self.unique_id,
            "material_name": self.material_name,
            "replacement": self.replacement,
            "replacement_ind": self.replacement_ind,
            "supercell": self.supercell,
            "bulk": self.bulk,
            "cluster": self.cluster,
            "calc_energy": self.calc_energy,
            "calc_energy_per_atom": self.calc_energy_per_atom,
            "calc_energy_toten": self.calc_energy_toten,
            "calc_performed": self.calc_performed,
            "calc_type": self.calc_type,
            "calc_output": self.calc_output,
        }

        new_row = pd.DataFrame.from_dict(data_dict, orient='index').T
        new_row = new_row.astype(bool_columns)

        is_InitialDatabase = isinstance(db_obj, mdb_indb.InitialDatabase)

        if is_InitialDatabase:
            struct_df = db_obj.df
        else:
            struct_df = db_obj

        struct_df.fillna(value=False, inplace=True)
        struct_df = struct_df.astype(bool_columns)

        if struct_df.shape[0] == 0:
            struct_df = new_row
        else:
            struct_df = pd.concat([struct_df, new_row], ignore_index=True)

        if is_InitialDatabase:
            db_obj.df = struct_df
        else:
            db_obj = struct_df

        return db_obj


class Bulk(Structure):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setting the bulk property as True.
        self.bulk = True

    def from_mdb_structure(
        self,
        mdb_structure,
        new_structure=None,
        material_name=None,
    ):
        if material_name:
            self.material_name = material_name
        else:
            self.material_name = mdb_structure.material_name

        if new_structure:
            self.structure = new_structure
        else:
            self.structure = mdb_structure.structure

        self.material_id = mdb_structure.material_id
        self.phase = mdb_structure.phase
        self.base = mdb_structure.base
        self.perturb = mdb_structure.perturb
        self.supercell = mdb_structure.supercell
        self.surface = mdb_structure.surface
        self.cluster = mdb_structure.cluster
        self.formula = mdb_structure.formula
        self.symmetry = mdb_structure.symmetry
        self.replacement = mdb_structure.replacement
        self.replacement_ind = mdb_structure.replacement_ind
        self.temperature = mdb_structure.temperature
        self.magnetic_properties = mdb_structure.magnetic_properties
        self.calc_energy = mdb_structure.calc_energy
        self.energy_per_atom = mdb_structure.energy_per_atom
        self.calc_energy_per_atom = mdb_structure.calc_energy_per_atom
        self.calc_energy_toten = mdb_structure.calc_energy_toten
        self.calc_performed = mdb_structure.calc_performed
        self.calc_type = mdb_structure.calc_type
        self.calc_output = mdb_structure.calc_output

        return self


class Surface(Structure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface = True

    def from_mdb_structure(
        self, mdb_structure, surface_miller, new_structure=None, material_name=None
    ):
        if material_name:
            self.material_name = material_name
        else:
            self.material_name = mdb_structure.material_name

        if new_structure:
            self.structure = new_structure
        else:
            self.structure = mdb_structure.structure

        self.material_id = mdb_structure.material_id
        self.phase = mdb_structure.phase
        self.base = mdb_structure.base
        self.perturb = mdb_structure.perturb
        self.supercell = mdb_structure.supercell
        self.surface = mdb_structure.surface
        self.surface_miller = surface_miller
        self.energy_per_atom = mdb_structure.energy_per_atom
        self.bulk = mdb_structure.surface
        self.cluster = mdb_structure.cluster
        self.formula = mdb_structure.formula
        self.symmetry = mdb_structure.symmetry
        self.temperature = mdb_structure.temperature
        self.replacement = mdb_structure.replacement
        self.replacement_ind = mdb_structure.replacement_ind
        self.magnetic_properties = mdb_structure.magnetic_properties
        self.calc_energy = mdb_structure.calc_energy
        self.calc_energy_per_atom = mdb_structure.calc_energy_per_atom
        self.calc_energy_toten = mdb_structure.calc_energy_toten
        self.calc_performed = mdb_structure.calc_performed
        self.calc_type = mdb_structure.calc_type
        self.calc_output = mdb_structure.calc_output

        return self


class Cluster(Structure):
    def __init__(self, cluster=True, **kwargs):
        super().__init__(**kwargs)
        self.cluster = cluster
