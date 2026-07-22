"""Utility functions for running several through aiida for active learning loops."""

import copy
import os
import pathlib as pl

# import queue
# import threading
import time
from enum import Enum

import ase.data as ad
import numpy as np
import pymatgen.core.structure as pymg_struct
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.orm import Bool, Dict, Group, List, Str, StructureData
from aiida.orm.nodes.data.array.kpoints import KpointsData
from aiida.plugins import WorkflowFactory
from ase import Atoms
from ase.io import write as ase_write
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

import atlas.workflows.aiida_utils as atl_aut
from atlas.active_learning import conversion as atl_conv
from atlas.core import ATL_DATA_DIR
from atlas.core import code_utils as atl_cut
from atlas.core import initial_db as atl_indb
from atlas.core.clusters import center_structure

VDW_DATA_PATH = pl.Path(ATL_DATA_DIR / 'vdw-data')

# Loading default aiida user profile
try:
    load_profile()
except Exception as e:
    # Print error message using custom print function
    tmp_logger = atl_cut.custom_print(f"Error loading aiida profile: '{e}'", 'error')

    # Disable added handlers to avoid duplicate messages, since custom_print
    # instantiates the mdb logger and adds a handle if the logger argument is
    # not passed.
    tmp_logger.handlers = []


def _hash_source_directory(filepath_files: pl.Path) -> str:
    """Return a combined SHA-256 hash of all `*.py` files in a directory.

    Files are visited in sorted order with their relative paths included in
    the hash, so additions, deletions, renames, and content edits are all
    detected. `__pycache__` and non-Python files are ignored.
    """
    import hashlib

    filepath_files = pl.Path(filepath_files)
    hasher = hashlib.sha256()
    for py_file in sorted(filepath_files.rglob('*.py')):
        rel = py_file.relative_to(filepath_files)
        hasher.update(str(rel).encode())
        hasher.update(py_file.read_bytes())
    return hasher.hexdigest()


def get_or_create_portable_code(
    label: str,
    filepath_files,
    filepath_executable: str,
    prepend_text: str = '',
) -> orm.PortableCode:
    """Return a PortableCode for *label*, recreating it if source has changed.

    Queries for the most recent `PortableCode` with the given label. If one
    is found, its bundled `*.py` files are compared (via a SHA-256 hash
    stored as the `source_hash_v2` extra) against the current source
    directory. When the hash differs, i.e. the source files were edited
    since the code was created, a new PortableCode is created instead of
    reusing the stale one. This prevents situations where an edited
    PortableCode script (e.g. `atl_process_structure.py`) silently keeps
    running the old version.

    Environment setup (module loads, venv activation, `OMP_NUM_THREADS`)
    is not baked into the PortableCode. Callers should pass it dynamically
    via `builder.metadata.options.prepend_text` instead, which AiiDA
    appends after the code's (empty) prepend_text in the final submission
    script. This way changing the TOML settings takes effect immediately
    without recreating the code node.

    The `prepend_text` parameter is accepted for backward compatibility but
    is ignored; callers should set it on the builder.

    The old node remains in the database for provenance of past calculations
    but is not reused for new submissions.
    """
    filepath_files = pl.Path(filepath_files)
    current_hash = _hash_source_directory(filepath_files)

    qb = orm.QueryBuilder()
    qb.append(
        orm.PortableCode,
        filters={'label': label},
        project='*',
    )
    qb.order_by({orm.PortableCode: {'ctime': 'desc'}})
    qb.limit(1)
    results = qb.all(flat=True)

    if results:
        existing = results[0]
        stored_hash = existing.base.extras.all.get('source_hash_v2')
        if stored_hash == current_hash:
            return existing
        # Source has changed (or the code predates source hashing): fall
        # through to create a fresh code rather than reusing the stale one.

    new_code = orm.PortableCode(
        label=label,
        filepath_files=str(filepath_files),
        filepath_executable=filepath_executable,
    )
    new_code.base.extras.set('source_hash_v2', current_hash)
    return new_code


def get_or_create_containerized_code(
    label: str,
    computer: orm.Computer,
    image_name: str,
    engine_command: str,
    filepath_executable: str,
) -> orm.ContainerizedCode:
    """Return a cached ContainerizedCode, or create one if none exists.

    Matches on `(label, computer, image_name, engine_command,
    filepath_executable)`. `prepend_text` is never baked into the code;
    callers should pass it dynamically via
    `builder.metadata.options.prepend_text`.

    Unlike :func:`get_or_create_portable_code`, no source hash is needed:
    a ContainerizedCode stores only the image path, not the image contents,
    so rebuilding the container image at the same path is automatically
    picked up at runtime. Old codes remain in the database for provenance
    but are not reused when the construction parameters change.
    """
    qb = orm.QueryBuilder()
    qb.append(orm.Computer, filters={'uuid': computer.uuid}, tag='computer')
    qb.append(
        orm.ContainerizedCode,
        with_computer='computer',
        filters={
            'label': label,
            'attributes.image_name': image_name,
            'attributes.engine_command': engine_command,
            'attributes.filepath_executable': filepath_executable,
        },
        project='*',
    )
    qb.order_by({orm.ContainerizedCode: {'ctime': 'desc'}})
    qb.limit(1)
    results = qb.all(flat=True)

    if results:
        return results[0]

    new_code = orm.ContainerizedCode(
        label=label,
        computer=computer,
        image_name=image_name,
        engine_command=engine_command,
        filepath_executable=filepath_executable,
    )
    return new_code


PARSER_DICT = {
    'parser_settings': {
        'add_misc': [
            'notifications',
            'run_status',
            'run_stats',
            'version',
        ],
        'add_kpoints': False,
        'add_structure': False,
        'add_poscar-structure': False,
        'add_trajectory': False,
        'add_forces': False,
        'add_stress': False,
        'add_bands': False,
        'add_dos': False,
        'add_energies': False,
        'add_projectors': False,
        'add_born_charges': False,
        'add_dielectrics': False,
        'add_hessian': False,
        'add_dynmat': False,
        'add_charge_density': False,
        'add_wavecar': False,
        'add_site_magnetization': False,
        'add_maximum_force': False,
        'add_maximum_stress': False,
        'add_total_energies': False,
        'critical_notifications': {
            'add_edddav_zhegv': True,
            'add_eddrmm_zhegv': True,
        },
        'critical_notification_errors': [
            'add_edddav_zhegv',
            'add_eddrmm_zhegv',
        ],
    }
}

# DB_NAME = "aiida/test_database/test.db"
# DB_NAME = f"{os.getcwd()}/small_fragment_adsorption.db"
SEL_DYNAMICS = None

# INCAR equivalent
# Set input parameters
# TODO: Move to /data
INCAR_SP = {
    # 'incar': {
    # general:
    'istart': 0,
    'icharg': 2,
    'gga': 'Pe',
    'ispin': 1,
    # "lorbit": 11,
    # electronic steps:
    'encut': 450,
    'ediff': 1e-6,
    'ismear': 0,
    'sigma': 0.03,
    'algo': 'Fast',
    'lreal': 'Auto',
    'nelm': 60,
    # ionic steps:
    'ibrion': -1,
    'nsw': 1,
    'ediffg': -0.03,
    'isif': 2,
    'potim': 0.3,
    # files to write:
    'lwave': False,
    'lcharg': False,
    # parallelization:
    'ncore': 4,
    # "kpar": 4,
    # dipole correction
    'lelf': False,
    # van der Waals:
    'ivdw': 11,
    # surface
    'idipol': 3,
    'ldipol': True,
    # }
}


INCAR_RELAX = {
    'incar': {
        ## general:
        'istart': 0,
        'icharg': 2,
        'gga': 'Pe',
        'ispin': 1,
        # "lorbit": 11,
        ## electronic steps:
        'encut': 450,
        'ediff': 1e-6,
        'ismear': 0,
        'sigma': 0.03,
        'algo': 'Fast',
        'lreal': 'Auto',
        'nelm': 60,
        ## ionic steps:
        'ibrion': 2,
        'nsw': 350,
        'ediffg': -0.03,
        'isif': 3,
        'potim': 0.3,
        ## files to write:
        'lwave': False,
        'lcharg': False,
        ## parallelization:
        'ncore': 4,
        # "kpar": 4,
        ## dipole correction
        'lelf': False,
        ## van der Waals:
        'ivdw': 11,
    }
}


# Default k-spacing values for every phase to be
# included in the INCAR.
# Ideally would be overwritten by the user.
# INCAR units (the value includes 2pi, but 2pi not in units)
KSPACING_DEFAULT = {
    'alpha': 0.133203528512207,
    'm1': 0.100530964914873,
    'beta-prime': 0.100530964914873,
    'm2': 0.100530964914873,
    'gamma': 0.141371669411541,
    'm3': 0.166504410640259,
    'epsilon': 0.153309721495182,
    'eta': 0.0948760981384118,
    'm4': 0.0948760981384118,
}


# TODO: Convert the subtypes into actual clases
class CalcType(Enum):
    """
    Class representing the available calculation types for DFT.
    This allows multiple aliases for the same calculation type.
    """

    SP_BULK = 'single_point_bulk'
    SP_SURFACE = 'single_point_surface'
    SP_CLUSTER = 'single_point_cluster'
    RELAX_BULK = 'relaxation_bulk'
    RELAX_SURFACE = 'relaxation_surface'
    RELAX_CLUSTER = 'relaxation_cluster'
    SP_ISOLATEDATOM = 'single_point_isolatedatom'
    DEFAULT = SP_BULK

    @classmethod
    def from_string(cls, value):
        """Get the CalcType corresponding to a string, considering aliases."""
        aliases = {
            'sp_bulk': cls.SP_BULK,
            'sp_surface': cls.SP_SURFACE,
            'sp_cluster': cls.SP_CLUSTER,
            # 'relax': cls.RELAX,
            # 'relaxation': cls.RELAX,
            'relaxation_bulk': cls.RELAX_BULK,
            'relaxation_cluster': cls.RELAX_CLUSTER,
            'relax_bulk': cls.RELAX_BULK,
            'relaxation_surface': cls.RELAX_SURFACE,
            'relax_surface': cls.RELAX_SURFACE,
            'relax_cluster': cls.RELAX_CLUSTER,
            'single_point_bulk': cls.SP_BULK,
            'static': cls.SP_BULK,
            'static_bulk': cls.SP_BULK,
            'static_surface': cls.SP_SURFACE,
            'static_cluster': cls.SP_CLUSTER,
            'single_point_surface': cls.SP_SURFACE,
            'single_point_cluster': cls.SP_CLUSTER,
            'standalone': cls.SP_ISOLATEDATOM,
            'single': cls.SP_ISOLATEDATOM,
            'singleatom': cls.SP_ISOLATEDATOM,
            'single_atom': cls.SP_ISOLATEDATOM,
            'SingleAtom': cls.SP_ISOLATEDATOM,
            'static_singleatom': cls.SP_ISOLATEDATOM,
            'static_isolated_atom': cls.SP_ISOLATEDATOM,
            'static_isolatedatom': cls.SP_ISOLATEDATOM,
            'isolatedatom': cls.SP_ISOLATEDATOM,
            'sp_isolatedatom': cls.SP_ISOLATEDATOM,
            'IsolatedAtom': cls.SP_ISOLATEDATOM,
            'isolated_atom': cls.SP_ISOLATEDATOM,
            'single_point_isolatedatom': cls.SP_ISOLATEDATOM,
        }
        return_value = aliases.get(value)

        # Return default if no match found
        if return_value is None:
            return_value = cls.DEFAULT
            atl_cut.custom_print(
                f"Unknown calculation type '{value}'. Using default '{cls.DEFAULT}'.",
                'error',
            )
        return return_value


def choose_queue(node_type: int, tot_procs: int = None):
    """
    Choose the scheduler and aiida's computer and code options.

    Parameters
    ----------
    node_type : int
        Whether to use 12, 24 or 28 core nodes in tekla.
    tot_procs : int
        How many cores to be used. Using more than the total number of cores in
        a single node will make use of more nodes.

    Returns
    -------
    dict
        The OPTIONS dict is aiida jobfile equivalent
    str
        The CODE_STRING str is aiida code identifier.
    """
    # Jobfile equivalent
    # In OPTIONS, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    OPTIONS = {}
    OPTIONS['account'] = ''
    OPTIONS['qos'] = ''
    OPTIONS['max_wallclock_seconds'] = 117280000
    OPTIONS['max_memory_kb'] = 102400000

    if not tot_procs:
        # If tot_procs is not given, use the number of cores
        # on a node, in order to use a single node.
        tot_procs = node_type

    # Code_string is chosen among the list given by 'verdi code list'
    if node_type == 28:
        CODE_STRING = 'vasp-5.4.4_28core@tekla2'
        OPTIONS['resources'] = {
            'parallel_env': 'c28m128ib_mpi',
            'tot_num_mpiprocs': tot_procs,
        }

    elif node_type == 24:
        CODE_STRING = 'vasp-5.4.4_24core@tekla2'
        OPTIONS['resources'] = {
            'parallel_env': 'c24m128ib_mpi',
            'tot_num_mpiprocs': tot_procs,
        }

    elif node_type == 12:
        CODE_STRING = 'vasp-5.4.4@tekla2'
        OPTIONS['resources'] = {
            'parallel_env': 'c12m48ib_mpi',
            'tot_num_mpiprocs': tot_procs,
        }

    return OPTIONS, CODE_STRING


def choose_queue_from_struct(queue_data: dict, computer: orm.Computer):
    """
    Choose scheduler options for aiida.

    Choose the scheduler and aiida's computer and code options according
    to the size of the structure.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure
    queue_data : dict
        Dictionary specifying which queue gets assigned to every atom
        interval.

    Returns
    -------
    dict
        The OPTIONS dict is aiida jobfile equivalent
    str
        The CODE_STRING str is aiida code identifier.
    int
        The number of nodes that will be used.
    """
    # Jobfile equivalent
    # In OPTIONS, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    options = {}
    options['account'] = queue_data.get('account')
    options['qos'] = queue_data.get('qos')
    options['max_wallclock_seconds'] = queue_data.get('max_wallclock_seconds')
    options['max_memory_kb'] = queue_data.get('max_memory_kb')
    options['custom_scheduler_commands'] = queue_data.get('custom_scheduler_commands')
    options['prepend_text'] = queue_data.get('prepend_text')

    # Getting code string
    code_string = queue_data['code_string']
    options['resources'] = queue_data['options_resources']

    # Getting options
    node_cpus = queue_data.get(
        'node_cpus', queue_data.get('options_resources', {}).get('tot_num_mpiprocs')
    )
    mult_nodes = queue_data.get('multiple', False)

    # Specific settings for slurm scheduler
    if computer.scheduler_type == 'core.slurm':
        options['resources']['num_cores_per_machine'] = node_cpus
        options['resources']['num_cores_per_mpiproc'] = 1
        options['resources']['num_mpiprocs_per_machine'] = node_cpus
    options['resources']['tot_num_mpiprocs'] = node_cpus

    return options, code_string, mult_nodes


def validate_kspacing(kspacing_dict: dict, phases: list[str]) -> None:
    """Validate that kspacing values exist for all phases.

    Checks each phase against the kspacing dict. Missing phases fall back to
    `ATL_DEFAULT` (or the legacy `MDB_DEFAULT`). If neither default exists
    and some phases are missing, raises `ValueError` listing the missing
    phases and suggesting how to fix the TOML.

    Parameters
    ----------
    kspacing_dict : dict
        The kspacing dictionary from the TOML `[dft.vasp.kspacing]` section.
        Modified in-place: if `MDB_DEFAULT` is present but `ATL_DEFAULT` is
        not, `MDB_DEFAULT` is copied to `ATL_DEFAULT`.
    phases : list of str
        Phase names present in the structures to be submitted for DFT.
    """
    missing = sorted(
        p for p in set(phases) if p not in kspacing_dict and p != 'IsolatedAtom'
    )

    if not missing:
        return

    default_kspacing = kspacing_dict.get('ATL_DEFAULT')

    if default_kspacing is None and 'MDB_DEFAULT' in kspacing_dict:
        atl_cut.custom_print(
            "WARNING: 'MDB_DEFAULT' is a deprecated kspacing key. "
            "Please rename it to 'ATL_DEFAULT' in your TOML "
            '[dft.vasp.kspacing] section. '
            "'MDB_DEFAULT' may be removed in a future version.",
            'warn',
        )
        kspacing_dict['ATL_DEFAULT'] = kspacing_dict['MDB_DEFAULT']
        default_kspacing = kspacing_dict['ATL_DEFAULT']

    if default_kspacing is not None:
        atl_cut.custom_print(
            f'No specific kspacing found for phase(s): {missing}. '
            f'Using ATL_DEFAULT = {default_kspacing} for these phases.',
            'warn',
        )
        return

    raise ValueError(
        f'No kspacing found for phase(s): {missing}, and no ATL_DEFAULT '
        'fallback is set. Please either:\n'
        '  (1) Add per-phase kspacing values in the [dft.vasp.kspacing] '
        'section of your TOML, or\n'
        '  (2) Add an ATL_DEFAULT = <value> entry to use as a fallback '
        'for all unspecified phases.\n'
        f'Currently available kspacing keys: {list(kspacing_dict.keys())}'
    )


def submit_aiida_vasp_calculation(
    index,
    target_structure,
    phase,
    material_name,
    unique_id,
    kspacing_dict,
    incar_settings_dict,
    calc_type,
    queue_dict,
    potential_family,
    potential_mapping,
    dry_run,
    return_builder,
    group,
    aiida_vasp_settings=None,
):
    atl_cut.custom_print(f"Row {index} - '{unique_id}'.", 'debug')

    if isinstance(target_structure, Atoms):
        target_structure = AseAtomsAdaptor.get_structure(target_structure)

    struct_formula = target_structure.formula.replace(' ', '')

    # Converting phase name to string to avoid all number structure
    phase = str(phase)
    kspacing = kspacing_dict.get(phase)

    # Using default kspacing if not found and ATL_DEFAULT is set
    if not kspacing and kspacing_dict.get('ATL_DEFAULT'):
        kspacing = kspacing_dict.get('ATL_DEFAULT')

    # Legacy fallback: MDB_DEFAULT (deprecated, use ATL_DEFAULT instead)
    elif not kspacing and kspacing_dict.get('MDB_DEFAULT'):
        atl_cut.custom_print(
            "WARNING: 'MDB_DEFAULT' is a deprecated kspacing key. "
            "Please rename it to 'ATL_DEFAULT' in your TOML "
            '[dft.vasp.kspacing] section. '
            "'MDB_DEFAULT' may be removed in a future version.",
            'warn',
        )
        kspacing = kspacing_dict.get('MDB_DEFAULT')

    # Raising error for missing kspacing entries if no default is set
    elif not kspacing and phase != 'IsolatedAtom':
        raise ValueError(
            f'No kspacing found for phase {phase} '
            f'(struct id: {unique_id}).'
            f'\nAvailable kspacings: {kspacing_dict.keys()}'
        )
    # IsolatedAtom shouldn't be in the database, but also shouldnt raise an error
    # Use the highest kspacing for isolated atoms, as the calculation is very simple
    elif phase == 'IsolatedAtom':
        # Sort kspacing dict by value
        sorted_kspacing = sorted(
            kspacing_dict.items(), key=lambda item: item[1], reverse=True
        )
        kspacing = sorted_kspacing[0][1]
        atl_cut.custom_print(
            f'Using highest kspacing {kspacing} for isolated atom.', 'debug'
        )
    if incar_settings_dict:
        incar = incar_settings_dict
        kspacing_vec = select_kspacing(
            target_structure, incar, phase, kspacing_dict, calc_type
        )
    else:
        # Generate INCAR with correct kspacing
        incar, kspacing_vec = generate_incar(
            structure=target_structure,
            phase=phase,
            calc_type=calc_type,
            kspacing=kspacing_dict,
        )

    # Changing all keys to uppercase to match the VASP INCAR format
    upper_incar = {'incar': {}}
    for key, value in incar.items():
        upper_incar['incar'][key.upper()] = value

    if not isinstance(calc_type, str):
        calc_type_str = calc_type.name.lower()
    else:
        calc_type_str = calc_type.lower()

    # Dictionary containing metadata for the calculation
    struct_name = f'{material_name}-{struct_formula}-{index}_{calc_type_str}'
    metadata_dict = {
        'label': struct_name,
        'description': (
            f'{calc_type_str.title()} calculation for {struct_formula}.'
            # "options": {"dry_run": True},
        ),
    }

    # Slab structures cannot be converted to aiida StructureData
    # directly, they must be converted to pymatgen structures
    # first.
    if isinstance(target_structure, Slab):
        target_structure = _convert_Slab_to_Structure(target_structure)

    # Centering the structure if it is a cluster
    if 'cluster' in calc_type_str:
        target_structure = center_structure(target_structure)

    # Getting structure as an aiida structure from pymatgen.
    structure = StructureData(pymatgen=target_structure)

    atl_cut.custom_print(f'Calculation type: {calc_type}', 'debug')

    # Get kpoints for aiida
    kpoints_data = generate_kpoints_data(
        structure=structure,
        kspacing=kspacing,
        kspacing_vec=kspacing_vec,
        calc_type=calc_type,
    )

    # Get selective dynamics
    selective_dynamics = None
    if target_structure.site_properties.get('selective_dynamics'):
        dynamics_list = target_structure.site_properties.get('selective_dynamics')
        selective_dynamics = {'positions_dof': List(dynamics_list)}

    # Setting code
    code_string = queue_dict['code_string']

    code = orm.load_code(code_string)

    # Getting comptuer from code
    curr_computer = code.computer

    # Jobfile equivalent
    # In options, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    options, code_string, mult = choose_queue_from_struct(
        # structure=target_structure, assign_dict=queue_dict
        queue_data=queue_dict,
        computer=curr_computer,
    )

    if not options.get('custom_scheduler_commands'):
        options['custom_scheduler_commands'] = ''

    # Defining the vasp.relax workchain object
    if 'single_point' in calc_type.value.lower():
        workchain = WorkflowFactory('vasp.v2.vasp')
    elif 'relaxation' in calc_type.value.lower():
        workchain = WorkflowFactory('vasp.v2.relax')

    # Preparing a builder object to be able to submit the workchain
    # and pass inputs to it
    builder = workchain.get_builder()

    # Passing the all inputs to the builder object

    if selective_dynamics:
        builder['dynamics'] = selective_dynamics

    # Passing structure
    builder['structure'] = structure

    # Assembling the builder.vasp object
    # Entries available for the options dict:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/calculations/usage.html
    # Two different ways to set the code, depending on the workchain.
    if workchain.get_name() == 'VaspRelaxWorkChain':
        builder.vasp.code = code
        builder.vasp['options'] = Dict(options)
        # builder.vasp["parameters"] = Dict(incar)
        builder.vasp['parameters'] = Dict(upper_incar)
        builder.vasp['potential_family'] = Str(potential_family)
        builder.vasp['potential_mapping'] = Dict(potential_mapping)
        builder.vasp['metadata'] = metadata_dict
        # builder.vasp['max_iterations'] = Int(2)
        builder.vasp['verbose'] = Bool(True)
        builder.vasp['kpoints'] = kpoints_data

        # Setting parser options
        builder.vasp['settings'] = Dict(PARSER_DICT)

        # Setting custom settings for the critical notifications
        custom_parser_settings = aiida_vasp_settings.get('parser_settings', {})
        for key, value in custom_parser_settings.items():
            builder.vasp['settings']['parser_settings'][key] = value

        critical_notifications = custom_parser_settings.get(
            'critical_notifications', {}
        )
        for key, value in critical_notifications.items():
            builder.vasp['settings']['parser_settings']['critical_notifications'][
                key
            ] = value

            if value:
                builder.vasp['settings']['parser_settings'][
                    'critical_notification_errors'
                ].append(key)

    elif workchain.get_name() == 'VaspWorkChain':
        builder.code = code
        builder['options'] = Dict(options)
        # builder["parameters"] = Dict(incar)
        builder['parameters'] = Dict(upper_incar)
        builder['potential_family'] = Str(potential_family)
        builder['potential_mapping'] = Dict(potential_mapping)
        builder['metadata'] = metadata_dict
        # builder['max_iterations'] = Int(2)
        builder['verbose'] = Bool(True)
        builder['kpoints'] = kpoints_data

        # Setting parser options
        builder['settings'] = Dict(PARSER_DICT)

        # Setting custom settings for the critical notifications
        custom_parser_settings = aiida_vasp_settings.get('parser_settings', {})
        for key, value in custom_parser_settings.items():
            builder['settings']['parser_settings'][key] = value

        critical_notifications = custom_parser_settings.get(
            'critical_notifications', {}
        )
        for key, value in critical_notifications.items():
            builder['settings']['parser_settings']['critical_notifications'][key] = (
                value
            )

    if dry_run:
        options['metadata'] = {}
        options['metadata']['dry_run'] = True

    # Preparing relaxation settings
    # builder.relax_settings = Dict()
    if 'single_point' in calc_type.value.lower():
        builder['settings']['parser_settings']['critical_notifications'][
            'ionic_convergence_not_reached'
        ] = False

        for error_name in [
            'ionic_convergence_not_reached',
            'electronic_convergence_not_reached',
        ]:
            builder['settings']['parser_settings'][
                'critical_notification_errors'
            ].append(error_name)

        builder['settings']['parser_settings']['check_errors'] = False
        builder.handler_overrides = Dict(dict={'check_ionic_converged': False})
        builder.handler_overrides = Dict(dict={'check_electronic_converged': False})
        builder['settings']['CHECK_IONIC_CONVERGENCE'] = False
        builder['settings']['CHECK_ELECTRONIC_CONVERGENCE'] = False
    #     builder.relax_settings['perform_static'] = True
    #     builder.relax_settings['perform'] = False
    #     builder.relax_settings['convergence_on'] = False

    #     # TODO: If this works, add a specific setting that allows partial relaxations
    #     builder.relax_settings['convergence_max_iterations'] = Int(1)
    #     builder.vasp.max_iterations = Int(1)

    if 'relaxation' in calc_type.value.lower():
        builder.relax_settings = Dict()
        builder.relax_settings['perform_static'] = True
        builder.relax_settings['perform'] = True
        # builder.relax_settings['convergence_max_iterations'] = Int(1)

    if dry_run:
        # TODO: Generate a fake node and return it
        atl_cut.custom_print('Dry run: nothing generated.', 'debug')

    if return_builder:
        atl_cut.custom_print('Returning builder.', 'debug')
        return builder
    else:
        # Submitting the calculation.
        # Aiida should handle the scheduler, ssh connection and result
        # retrieval if everything is configured correctly
        node = submit(builder)
        atl_cut.custom_print(
            (
                f'Launched workchain for structure {index}:'
                f" '{struct_formula}' ({phase}) - node pk: {node.pk}"
            ),
            'debug',
        )

        # Storing each calculation's unique_id (uuid) with the node, so once the
        # calculation is done, it can be traced back to its database entry by
        # getting the hex of the uuid in the database and searching which
        # RelaxWorkChain has the same hex, and then, the calculation results
        # of its last children node will be gathered and added to the DB.
        # node.base.extras.set("atl_calc_uuid", target_row.unique_id.hex)
        node.base.extras.set('atl_calc_uuid', unique_id)
        node.base.extras.set('atl_struct_type', calc_type.value.lower())
        node.base.extras.set('struct_name', struct_name)

        if group:
            group.add_nodes(node)
    return node


def run_dataframe_vasp_simulations_aiida(
    sel_struct_df,
    group_name: str,
    calc_type: CalcType,
    kspacing_dict: dict,
    max_batch: int,
    start_on: int,
    potential_mapping: dict,
    potential_family: str,
    initial_db: 'atl_indb.InitialDatabase',
    queue_dict: dict,
    incar_dict: dict = None,
    dry_run: bool = False,
):
    # Getting current time
    ctime = time.strftime('%Y%m%dT%H%M%S')

    # Print for dry runs
    if dry_run:
        atl_cut.custom_print(
            (
                'This is a dry run: calculations will not be submitted. '
                'No aiida group will be created'
            ),
            'warning',
        )
    else:
        # Creating a new aiida group for the calculations.
        # It will provide an ID for the entire batch
        group_label = f'{group_name}_{calc_type}_batch_{ctime}'
        group = Group(label=group_label)
        group.store()
        atl_cut.custom_print(f'Group identifier: "{group.uuid}"', 'info')
        atl_cut.custom_print(f'Group label: {group_label}', 'info')

    # Starting calculation index

    # Splitting the initial database in chunks of size
    # max_batch
    num_chunks = len(sel_struct_df) // max_batch
    if num_chunks == 0:
        num_chunks = 1

    atl_cut.custom_print(
        (
            f'Splitting database with {len(sel_struct_df)}'
            f' entries into {num_chunks} chunks.'
        ),
        'info',
    )

    # Iterating over every chunk.
    for chunk_id, chunk in enumerate(np.array_split(sel_struct_df, num_chunks)):
        # Skipping unwanted chunks
        if chunk_id < start_on:
            continue

        atl_cut.custom_print(f'Working on chunk {chunk_id}...', 'info')

        # Sorting chunk so smallest structures are run first
        sort_chunk_size(chunk)

        # Creating list for storing the nodes once submitted
        chunk_node_list = []

        # Iterating over the chunk and launching a separate
        # vasp workchain for every structure contained.
        for it, target_row in chunk.iterrows():
            # Gathering calculation information and
            # submitting the calculation through AiiDA.
            (
                curr_structure,
                curr_material_name,
                curr_unique_id,
                curr_phase,
            ) = gather_calc_data_from_row(target_row)

            node = submit_aiida_vasp_calculation(
                index=it,
                target_structure=curr_structure,
                phase=curr_phase,
                material_name=curr_material_name,
                unique_id=curr_unique_id,
                initial_db=initial_db,
                kspacing_dict=kspacing_dict,
                incar_dict=incar_dict,
                calc_type=calc_type,
                queue_dict=queue_dict,
                potential_family=potential_family,
                potential_mapping=potential_mapping,
                dry_run=dry_run,
                group=group,
            )

            chunk_node_list.append(node)

        # Waiting until the current chunk's calculations are done.
        chunk_finished = False

        # Initialization of chunk node list as all False.
        node_status_list = np.full(len(chunk_node_list), False)

        while not chunk_finished:
            # Skipping the wait if dry run is selected
            if dry_run:
                chunk_finished = True

            atl_cut.custom_print(
                (
                    f'({time.strftime("%H:%M:%S")})'
                    f' - {np.count_nonzero(node_status_list)}'
                    f'/{len(node_status_list)} - Waiting for calculations'
                    f' from chunk {chunk_id} to be finished...'
                ),
                'info',
            )
            node_status_list = []
            for nod in chunk_node_list:
                # Some interesting options related to the node status:
                # 'is_excepted', 'is_failed', 'is_finished',
                # 'is_finished_ok', 'is_killed', 'is_sealed', 'is_stored',
                # 'process_status', exception, 'is_terminated',
                node_status_list.append(nod.is_finished)

            if all(node_status_list):
                chunk_finished = True

                for nod in chunk_node_list:
                    unique_id = nod.base.all.extras.get('atl_calc_uuid')
                    atl_cut.custom_print(
                        (
                            f"VaspCalculation {nod.pk} ('{unique_id}') finished: "
                            f'{nod.exit_status} - {nod.exit_message}'
                        ),
                        'debug',
                    )

            else:
                time.sleep(500)

        atl_cut.custom_print(f'Chunk {chunk_id} done!', 'done')

    atl_cut.custom_print('All calculations finished!', 'done')
    atl_cut.custom_print("Check 'verdi process list' for more information", 'info.')


def gather_calc_data_from_row(target_row, curr_structure=None):
    # Gathering calculation information and
    # submitting the calculation through AiiDA.
    if not curr_structure:
        curr_structure = target_row.structure.get_sorted_structure()

    if not isinstance(curr_structure, Atoms):
        curr_structure = curr_structure.get_sorted_structure()
        curr_material_name = target_row.material_name
        curr_unique_id = target_row.unique_id

        try:
            curr_phase = target_row.phase.name
        except AttributeError:
            curr_phase = target_row.phase
    else:
        curr_structure = AseAtomsAdaptor().get_structure(curr_structure)
        curr_material_name = target_row.get('struct_name')
        curr_unique_id = target_row.get('aiida_uuid')
        curr_phase = target_row.get('phase', 'unknown')

    return curr_structure, curr_material_name, curr_unique_id, curr_phase


def can_submit_calculation(
    code: str, limit: int, computer: orm.Computer = None
) -> bool:
    """
    Check if a calculation can be submitted to a given computer's queue.

    This function checks if the number of jobs currently running
    is smaller than the limit given. If smaller, the calculation
    can be submitted.

    In some supercomputer clusters there is a limit on the number
    of jobs that can be submitted at the same time, and going over
    this limit will result in the job not entering the queue.
    AiiDA will reattempt to submit after the time given by
    `exponential_backoff_retry` is elapsed and for `task_maximum_attempts`
    attempts, and if the number of attempts is exceeded, the calculation
    will be paused, holding the loop.

    Parameters
    ----------
    code : str
        AiiDA code label. Will be used to get the computer.
    computer : orm.Computer
        AiiDA computer.
    limit : int
        Maximum number of calculations that can be submitted.
        The limit will be set by the supercomputer's scheduler,
        in the case of SLURM, it can be checked with `sacctmgr show qos XXXX`
        under the `MaxSubmitPU` column.
    user : str, optional
        Username for the calculation. If not given, the default user
        will be used.

    Returns
    -------
    bool
        Whether the calculation can be submitted or not.
    """
    # Getting computer
    if not computer:
        computer = orm.load_code(label=code).computer

    # Getting default user if not specified
    user = orm.User.collection.get_default()

    # Getting the scheduler and transport
    # Transport must be set in order for the
    # scheduler to be able to check the jobs.
    authinfo = computer.get_authinfo(user)
    scheduler = computer.get_scheduler()
    transport = authinfo.get_transport()
    scheduler.set_transport(transport)

    n_jobs = len(scheduler.get_jobs())
    # Checking if the number of jobs is below the limit
    below_limit = False
    try:
        if n_jobs < limit:
            below_limit = True
    except EOFError:
        # If the scheduler is not available, the calculation
        # cannot be submitted.
        below_limit = False

    return below_limit, n_jobs


def update_db_with_dft_results(sel_struct_db, queue):
    """
    Update the original database with the results from the calculations.

    Parameters
    ----------
    sel_struct_db : list[Atoms]
        List of structures to be updated.
    queue : list[aiida.orm.nodes.process.workflow.workchain.WorkChainNode]
        List of AiiDA workchain nodes with the results of the calculations.
    """
    from atlas.active_learning import active_learning_utils as atl_al_ut

    num_error = 0
    num_correct = 0

    running_calcs = []

    # Going over every calculation in the queue
    for node in queue:
        # Getting the unique_id of the calculation
        unique_id = node.base.extras.all.get('atl_calc_uuid')

        # Getting the calcjob node(s) from the workchain descendants
        calcjob_nodes = [
            chld
            for chld in node.called_descendants
            if isinstance(chld, orm.CalcJobNode)
        ]

        # Skipping if the calculation is not finished
        if not node.is_finished_ok:
            if node.is_terminated:
                status = node.exit_status or node.process_state.value
                if calcjob_nodes:
                    last_calcjob = calcjob_nodes[-1]
                    atl_cut.custom_print(
                        f"Skipping calc. {node.process_label}<{node.pk}> "
                        f"{node.process_state.value} [{status}]\n"
                        f"    └── {last_calcjob.process_label}<{last_calcjob.pk}> "
                        f"{last_calcjob.process_state.value} "
                        f"[{last_calcjob.exit_status}]"
                        f" ('struct_id: {unique_id}')",
                        'warning',
                    )
                else:
                    atl_cut.custom_print(
                        f"Skipping calc. {node.process_label}<{node.pk}> "
                        f"{node.process_state.value} [{status}]"
                        f" ('struct_id: {unique_id}')",
                        'warning',
                    )
                num_error += 1
            else:
                atl_cut.custom_print(
                    f"Calculation {node.process_label}<{node.pk}> running...",
                    'debug',
                )
                running_calcs.append(node.pk)
            continue

        # Getting the index of the calculation in the database
        last_calcjob = calcjob_nodes[-1].pk

        # Getting index of the matching structure
        # There should be only one.
        # Find a corner to cry if there is more than one.
        idx = [
            str_tupl[0]
            for str_tupl in enumerate(sel_struct_db)
            if str_tupl[1].info['atl_id'] == unique_id
        ][0]

        # Getting the calculation results
        results_dict = atl_al_ut.gather_dft_calcs_vasp([last_calcjob])

        # Updating the database with the results
        new_struct = sel_struct_db[idx].copy()
        new_struct.info['calc_type'] = 'vasp_dft'
        new_struct.info['calc_performed'] = True
        try:
            new_struct.info['REF_energy'] = results_dict[0]['info']['REF_energy']

            # Getting stress key from results and adding it to the structure
            parse_stress_from_structure(results_dict, new_struct)

            new_struct.arrays['REF_forces'] = np.array(results_dict[0]['REF_forces'])
            new_struct.arrays['positions'] = np.array(results_dict[0]['positions'])
        except IndexError:
            new_struct.info['REF_energy'] = results_dict['info']['energy']

            # Getting stress key from results and adding it to the structure
            parse_stress_from_structure(results_dict, new_struct)

            new_struct.arrays['REF_forces'] = np.array(results_dict['REF_forces'])
            new_struct.info['REF_energy'] = results_dict['info']['energy']
            new_struct.arrays['REF_forces'] = np.array(results_dict['REF_forces'])
            new_struct.arrays['positions'] = np.array(results_dict['positions'])

        sel_struct_db[idx] = new_struct

        atl_cut.custom_print(
            f"Calculation {node.process_label}<{node.pk}> completed!",
            'done',
        )
        num_correct += 1

    if len(running_calcs) > 0:
        all_descendants = []
        for node in running_calcs:
            descendants = orm.load_node(node).called_descendants
            all_descendants.extend(descendants)

        calcjob_ids = [
            child.pk for child in all_descendants if isinstance(child, orm.CalcJobNode)
        ]
        atl_cut.custom_print(
            f'Currently running {len(running_calcs)} calculations: {calcjob_ids}...',
            'info',
        )
    return num_correct, num_error


def parse_stress_from_structure(results_dict: dict, new_struct: Atoms) -> None:
    """
    Parse stress from results dictionary and add it to
    the structure info, updating it in-place.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing the results from a VASP DFT calculation.
    new_struct : Atoms
        Structure to update.
    """
    stress = None
    stress = results_dict[0]['info'].get('stress')
    if not stress:
        stress = results_dict[0]['info'].get('REF_stress')

    new_struct.info['REF_stress'] = stress


def run_dataframe_vasp_aiida_queue(
    initial_db: 'atl_indb.InitialDatabase | list[Atoms]',
    config_dict: dict,
    log_file_path: str,
):
    group_name: str = config_dict.get('general', {}).get('aiida_group_name')
    results_path: pl.Path = pl.Path(
        config_dict.get('general', {}).get('result_file_path', 'dft_results')
    )

    # Getting current time
    ctime = time.strftime('%Y%m%dT%H%M%S')

    # Adding the current time to the results path
    stem = f'atl_dft_{ctime}_{results_path.stem}'
    results_path = results_path.with_stem(stem)

    # Adding xyz suffix if not present
    if results_path.suffix != '.xyz':
        results_path = results_path.with_suffix('.xyz')

    kspacing_dict: dict | float = config_dict.get('kpoints', {}).get('kspacing')
    dry_run: bool = config_dict.get('general', {}).get('dry_run', False)
    max_batch: int = config_dict.get('general', {}).get('max_batch', 1)
    sel_structures_type = config_dict.get('general', {}).get('selected_structure_type')
    if isinstance(sel_structures_type, str):
        sel_structures_type = [sel_structures_type]
    start_on_struct_idx: int = config_dict.get('general', {}).get(
        'start_on_struct_idx', 0
    )
    queue_check_interval: int = config_dict.get('general', {}).get(
        'queue_check_interval_seconds', 240
    )
    aiida_potential_family: str = config_dict.get('calculation', {}).get(
        'aiida_potential_family'
    )

    potential_mapping: dict = config_dict.get('calculation', {}).get(
        'potential_mapping', {}
    )
    queue_dict: dict = config_dict.get('queue')
    aiida_vasp_settings: dict = config_dict.get('aiida_vasp', {})

    # Print for dry runs
    if dry_run:
        atl_cut.custom_print(
            (
                'This is a dry run: calculations will not be submitted. '
                'No aiida group will be created'
            ),
            'warning',
        )
        group = None
    else:
        # Creating a new aiida group for the calculations.
        # It will provide an ID for the entire batch
        group_label = f'{group_name}_batch_{ctime}'
        group = Group(label=group_label)
        group.store()
        atl_cut.custom_print(f'Group identifier: "{group.uuid}"', 'info')
        atl_cut.custom_print(f'Group label: {group_label}', 'info')

    # Starting calculation queue
    # calc_queue = queue.Queue(maxsize=max_batch)

    # Check if every structure has an unique uuid
    if isinstance(initial_db, list):
        unique_uuids = set([struct.info.get('atl_id') for struct in initial_db])
        if len(unique_uuids) != len(initial_db):
            raise ValueError(
                'Some structures in the initial database lack unique IDs. '
                'Please ensure all structures have a unique "atl_id" in their info '
                'or remove all atl_id keys and re-run to auto-generate them.'
            )

    # Getting the selected structures as a list of ASE Atoms
    if isinstance(initial_db, atl_indb.InitialDatabase):
        if sel_structures_type:
            mask = initial_db.df[sel_structures_type].any(axis=1)
            filtered_df = initial_db.df[mask]
        else:
            filtered_df = initial_db.df
        sel_struct_db = [
            initial_db.db_struct_to_ase(row) for _, row in filtered_df.iterrows()
        ]
    elif isinstance(initial_db, list):
        if sel_structures_type:
            sel_struct_db = [
                struct
                for struct in initial_db
                if any(struct.info.get(st) for st in sel_structures_type)
            ]
        else:
            sel_struct_db = initial_db
    else:
        raise ValueError('Initial database type not recognized.')

    atl_cut.custom_print(
        (
            'Starting queue for running database with'
            f' {len(sel_struct_db)} structures...'
        ),
        'info',
    )

    max_batch = min(max_batch, len(sel_struct_db))

    current_row_index = start_on_struct_idx
    calcs_remaining = True
    first_step = True

    queue = []
    total_loops = 1
    correct_calcs = 0
    error_calcs = 0

    # The potential mapping selects which potential to use
    # This could for instance be {'Si': 'Si_GW'} to use the GW ready potential
    # We use a specific function which just uses the default potential for
    # every atom.
    potential_mapping_dict: dict = atl_aut.generate_potential_mapping(potential_mapping)

    atl_cut.custom_print(
        (
            'Starting submission loop. '
            f"Check the log in '{log_file_path}' for more detailed information."
        ),
        'info',
    )

    # Repeat while there are calculations to finish
    while calcs_remaining:
        atl_cut.custom_print(
            (
                f'[bold blue]Step {total_loops}[/] - '
                f'Elapsed Time: {queue_check_interval * (total_loops - 1)} s '
                '- Checking items in queue...'
            ),
            'info',
        )
        # Check all calculations and remove any that are finished.
        # chunk_status contains True for finished calculations
        chunk_status = []
        for nod in queue:
            chunk_status.append(nod.is_finished)
        chunk_status_arr = np.array(chunk_status)

        # Removing the finished ones from the queue and current chunk
        if not first_step:
            # Return results database with the dft results
            num_performed_calcs = sum(
                [True for cal in sel_struct_db if cal.info.get('calc_performed', False)]
            )
            atl_cut.custom_print(
                f'num_performed_calcs before updating: {num_performed_calcs}.', 'debug'
            )

            # Updating the database with the results and counting the number of
            # correct and error calculations
            n_correct, n_error = update_db_with_dft_results(sel_struct_db, queue)
            correct_calcs += n_correct
            error_calcs += n_error
            atl_cut.custom_print(
                f'Total count of calculations finished correctly: {correct_calcs}.',
                'info',
            )
            atl_cut.custom_print(
                f'Total count of calculations finished with errors: {error_calcs}.',
                'info',
            )

            # Writing the results to the results file
            ase_write(filename=results_path, images=sel_struct_db, format='extxyz')
            num_performed_calcs = sum(
                [True for cal in sel_struct_db if cal.info.get('calc_performed', False)]
            )
            atl_cut.custom_print(
                f'num_performed_calcs after updating: {num_performed_calcs}.', 'debug'
            )

            if len(chunk_status_arr) == 0:
                print()
                atl_cut.custom_print(
                    f'len(chunk_status_arr) == 0: {len(chunk_status_arr) == 0}', 'debug'
                )
                calcs_remaining = False
                break

            # Removing the finished calculations from the queue
            queue = list(np.array(queue)[~chunk_status_arr])
            chunk_status_arr = chunk_status_arr[~chunk_status_arr]

        # Update the queue while the length of the queue is smaller
        # than the maximum allowed number of calculations.
        while (
            len(queue) < max_batch
            and total_loops < len(sel_struct_db)
            and current_row_index < len(sel_struct_db)
        ):
            # Choosing current structure and gathering information
            target_row = sel_struct_db[current_row_index]
            curr_structure = target_row

            curr_material_name = target_row.info['struct_name']
            curr_unique_id = target_row.info['atl_id']
            curr_phase = target_row.info.get('phase')
            if curr_phase is None:
                curr_phase = target_row.info.get('atl_struct_type', 'bulk')
            struct_type = target_row.info.get('atl_struct_type', 'bulk')

            # Skipping structures that have already been calculated
            structure_already_calculated = target_row.info.get('calc_performed', False)
            if structure_already_calculated:
                atl_cut.custom_print(
                    (
                        f'Skipping structure {current_row_index} as it has already '
                        'been calculated.'
                    ),
                    'info',
                )
                current_row_index += 1
                continue

            calc_type_str = (
                config_dict.get('calculation', {}).get('calc_type') + '_' + struct_type
            )
            calc_type = atl_aut.CalcType.from_string(calc_type_str)

            incar_sections = config_dict.get('incar', {})
            incar_dict = incar_sections.get(struct_type, None)
            if not incar_dict:
                incar_dict = incar_sections.get('bulk', INCAR_SP)
                if struct_type != 'bulk':
                    atl_cut.custom_print(
                        (
                            f'No incar.{struct_type} section in config for '
                            f'structure {current_row_index}. '
                            "Using 'bulk' INCAR settings as fallback."
                        ),
                        'warning',
                    )

            # Iterating over the chunk and launching a separate
            # vasp workchain for every structure contained.
            # for it, target_row in chunk.iterrows():
            # Gathering calculation information and
            # submitting the calculation through AiiDA.
            node = submit_aiida_vasp_calculation(
                index=current_row_index,
                target_structure=curr_structure,
                phase=curr_phase,
                material_name=curr_material_name,
                unique_id=curr_unique_id,
                # initial_db=initial_db,
                return_builder=False,
                kspacing_dict=kspacing_dict,
                incar_settings_dict=incar_dict,
                calc_type=calc_type,
                queue_dict=queue_dict,
                potential_family=aiida_potential_family,
                potential_mapping=potential_mapping_dict,
                aiida_vasp_settings=aiida_vasp_settings,
                dry_run=dry_run,
                group=group,
            )

            current_row_index += 1
            queue.append(node)
            atl_cut.custom_print(
                (
                    f"Submitted calculation '{node.pk}'. "
                    f'Queue length: {len(queue)}/{max_batch}'
                ),
                'debug',
            )

        time.sleep(queue_check_interval)

        first_step = False
        total_loops += 1

        atl_cut.custom_print(f'current_row_index: {current_row_index}', 'debug')
        atl_cut.custom_print(f'len(sel_struct_db): {len(sel_struct_db)}', 'debug')
        atl_cut.custom_print(f'len(queue): {len(queue)}', 'debug')
        atl_cut.custom_print(f'total_loops: {total_loops}', 'debug')
        if current_row_index >= len(sel_struct_db) and len(queue) == 0:
            atl_cut.custom_print('Setting calcs_remaining to False', 'debug')
            calcs_remaining = False

    atl_cut.custom_print('All calculations finished!', 'done')
    atl_cut.custom_print("Check 'verdi process list' for more information", 'info.')


def determine_vacuum_direction(structure):
    """
    A heuristic function that determines the vacuum direction by
    examining the spread of atomic positions in fractional coordinates.

    Returns the index (0, 1, or 2) corresponding to the direction
    with the largest vacuum gap.
    """
    # Get fractional coordinates
    frac_coords = structure.frac_coords

    gaps = []
    for ax in range(3):
        # Sort the fractional coordinates along the i-th direction
        coords = np.sort(frac_coords[:, ax])

        # Compute differences between successive atoms
        diffs = np.diff(coords)

        # Also consider the gap from the last back to the first (periodic boundary)
        gap_end = 1 - coords[-1] + coords[0]
        max_gap = max(np.max(diffs), gap_end)
        gaps.append(max_gap)

    # The direction with the largest gap is assumed to be the vacuum direction
    return np.argmax(gaps)


def kpoint_mesh_from_density(structure, kspacing):
    """Return kpoint mesh (3x3) from kpoint array,
    intended for surfaces.
    """
    # Read POSCAR
    poscar = Poscar(structure)

    kpt_dens_arr = np.repeat(kspacing, 3)

    # Getting lattice vectors
    l_mat = poscar.structure.lattice.matrix

    # Getting volume of the reciprocal cell
    v_mat = np.dot(np.cross(l_mat[0, :], l_mat[1, :]), l_mat[2, :])

    # Computing kpoint values for each axis
    a_rcpr = np.linalg.norm((np.cross(l_mat[1, :], l_mat[2, :])) / v_mat)
    b_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[2, :])) / v_mat)
    c_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[1, :])) / v_mat)
    arr_kpt_run = 1 / (kpt_dens_arr / np.array((a_rcpr, b_rcpr, c_rcpr)))

    # Round kpoints to nearest integer
    arr_kpt_run = np.around(arr_kpt_run)

    # Setting the number of kpoints to 1 in the axis normal to the surface
    normal_dir = determine_vacuum_direction(poscar.structure)
    arr_kpt_run[normal_dir] = 1

    return arr_kpt_run


def select_kspacing(
    curr_structure, incar: dict, phase: str, kspacing_dict: dict, calc_type
):
    calc_type_str = calc_type.value if not isinstance(calc_type, str) else calc_type

    kspacing_get = kspacing_dict.get(phase)

    # Using default kspacing if not found and ATL_DEFAULT is set
    if not kspacing_get and kspacing_dict.get('ATL_DEFAULT'):
        kspacing_get = kspacing_dict.get('ATL_DEFAULT')

    elif not kspacing_get and phase != 'IsolatedAtom':
        raise ValueError('No kspacing found for the current phase.')

    # IsolatedAtom shouldn't be in the database, but also shouldnt raise an error
    # Use the highest kspacing for isolated atoms, as the calculation is very simple
    elif phase == 'IsolatedAtom':
        kspacing_get = [1, 1, 1]

    if 'surface' in calc_type_str:
        kspacing_calc = kpoint_mesh_from_density(
            structure=curr_structure, kspacing=kspacing_get
        )
    else:
        kspacing_calc = kspacing_get

    return kspacing_calc


def _convert_Slab_to_Structure(target_structure):
    conv_structure = pymg_struct.Structure(
        species=target_structure.species,
        coords=target_structure.cart_coords,
        coords_are_cartesian=True,
        lattice=target_structure.lattice,
    )
    return conv_structure


def sort_chunk_size(chunk):
    # Creating list for storing the number of atoms
    size_list = []

    # Gathering the number of atoms for every structure
    # and adding it to the size_list
    for _it, row in chunk.iterrows():
        size_list.append(len(row.structure.sites))

    # Creating a new column for the atom number
    chunk['num_atoms'] = size_list

    # Sorting the chunk using the atom number, from
    # small to large.
    chunk.sort_values(by=['num_atoms'], inplace=True)

    return chunk


def generate_incar(
    structure, phase: str, calc_type: str, kspacing: dict = KSPACING_DEFAULT
):
    """
    Generate an incar file using depending on the calculation type.

    This incar includes a kspacing variable that depends on the phase.

    Parameters
    ----------
    structure: pymatgen.core.structure.Structure
        Current structure
    phase : str
        Phase of the current structure
    calc_type : str
        Calculation type, can be either 'relax' for relaxation
        or 'sp' for single point

    Returns
    -------
    dict
        dictionary representation of the INCAR
    """
    if 'relax' in calc_type.value:
        atl_cut.custom_print('Selecting relaxation INCAR...', 'debug')
        incar = INCAR_RELAX
    elif calc_type.value in [
        'single_point_surface',
        'single_point_cluster',
        'single_point_bulk',
        'single_point',
    ]:
        atl_cut.custom_print('Selecting single point INCAR...', 'debug')
        incar = INCAR_SP

    kspacing = select_kspacing(structure, incar, phase, kspacing, calc_type)

    if 'cluster' in calc_type.value:
        # Setting the center of the cell in direct lattice coordinates
        # with respect to which the total dipole-moment in the cell
        # is calculated.
        incar['incar']['DIPOL'] = [0.5, 0.5, 0.5]
        incar['incar']['IDIPOL'] = 4

    if 'singleatom' in calc_type.value:
        # Setting the center of the cell in direct lattice coordinates
        incar['incar']['DIPOL'] = [0.5, 0.5, 0.5]
        incar['incar']['IDIPOL'] = 4
        if incar.get('incar', {}).get('NSW', 0) < 300:
            incar['incar']['NSW'] = 300

    return incar, kspacing


def generate_kpoints_data(structure, calc_type, kspacing=None, kspacing_vec=None):
    # TODO: Check if this is necessary after all.
    # if kspacing is not None and kspacing_vec is not None:
    #     raise ValueError('Both kspacing and kspacing_vec cannot be given.')

    # Get kpoints for aiida:
    # kpoints_data = DataFactory("core.array.kpoints")
    kpoints_data = KpointsData()

    if isinstance(structure, Slab):
        structure = _convert_Slab_to_Structure(target_structure=structure)
        structure = StructureData(pymatgen=structure)
    elif not isinstance(structure, StructureData):
        structure = StructureData(pymatgen=structure)

    kpoints_data.set_cell_from_structure(structuredata=structure)

    calc_name = calc_type.name if not isinstance(calc_type, str) else calc_type
    # Bulks
    if 'bulk' in calc_name.lower():
        kpoints_data.set_kpoints_mesh_from_density(distance=kspacing)
        atl_cut.custom_print(
            f'Generated kpoints for bulk: {kpoints_data.get_kpoints_mesh()}', 'debug'
        )

    # Surfaces
    elif 'surface' in calc_name.lower():
        kpoints_data.set_kpoints_mesh_from_density(distance=kspacing)
        kpoint_mesh = kpoints_data.get_kpoints_mesh()[0]

        # This will return a tuple with the kmesh and displacement
        # As the surfaces will be slabs with a long z axis,
        # 1 kpoint on the z-axis wll be employed
        kpoint_mesh[2] = 1
        kpoints_data.set_kpoints_mesh(mesh=kpoint_mesh)

        atl_cut.custom_print(
            (
                f'Generated kpoints for surface using kspacing ({kspacing:.4f}): '
                f'{kpoints_data.get_kpoints_mesh()[0]} (z-axis forced to 1)'
            ),
            'debug',
        )

    # Clusters and reference strucures
    check_list = (
        item in calc_name.lower()
        for item in [
            'cluster',
            'isolatedatom',
            'sp_isolatedatom',
            'sp_cluster',
        ]
    )
    if any(check_list):
        kpoints_data.set_cell_from_structure(structuredata=structure)
        kpoints_data.set_kpoints_mesh([1, 1, 1])
        atl_cut.custom_print(
            f'Generated kpoints for {calc_name.lower()}: '
            f'{kpoints_data.get_kpoints_mesh()}',
            'debug',
        )
    return kpoints_data


def add_aiida_group_to_db(db_obj: str, group_identifier, copy=False):
    # Loading InitialDatabase from a given path
    database = atl_indb.get_database(db_obj)
    database_df = database.df

    db_uuids = database_df.unique_id
    db_uuids_hex = [uuid_str.hex for uuid_str in db_uuids]

    # Loading aiida calculation group from a group identifier
    group = orm.load_group(identifier=group_identifier)

    for node in group.nodes:
        node_calc_uuid = node.base.extras.all.get('atl_calc_uuid', None)

        called_nodes = []
        for called in node.called_descendants:
            if (
                called.class_node_type == 'process.calculation.calcjob.CalcJobNode.'
                and called.exit_status == 0
            ):
                called_nodes.append(called)

        if node_calc_uuid and len(called_nodes) > 0:
            # Finding the position of the current index
            # in the database
            hex_index = db_uuids_hex.index(node_calc_uuid)

            # Getting the database entry using the position
            # matching_db_entry = database_df.iloc[hex_index]

            # Getting calculation data from the node
            data_dict = atl_conv.gather_calc_data_from_node(called_nodes[-1])

            # Inserting information into the database entry
            vasprun = data_dict['atoms_obj']
            database_df.at[hex_index, 'calc_performed'] = True
            database_df.at[hex_index, 'calc_type'] = vasprun.calc
            database_df.at[hex_index, 'calc_energy'] = vasprun.get_potential_energy()
            database_df.at[hex_index, 'calc_energy_toten'] = vasprun.get_total_energy()
            database_df.at[hex_index, 'calc_energy_per_atom'] = (
                vasprun.get_potential_energy() / len(vasprun)
            )
            database_df.at[hex_index, 'calc_output'] = vasprun
        else:
            pass

    # Overwriting database with the new information
    database.df = database_df

    if copy:
        if isinstance(db_obj, str):
            db_path = pl.Path(db_obj)
            database.save_database(path=db_path.parent, suffix='copy')
        else:
            raise NotImplementedError


def generate_potential_mapping(assign_dict=None) -> dict:
    """
    Generate a dictionary specifying the pseudopotential mapping for vasp.

    This function only assigns the default potential for every
    atom. The user can specify different mappings if needed.

    Inputs
    ------
    The function itself requires no input when called, but it will
    attempt to read the contents of a 'potential_mapping' file on
    the same folder where the code is being executed (CWD).
    There will be an example on the github repo. Also, an example
    follows:


    ::

        # Header lines for comments that will be ignored.
        # The lines following the headers should not have any comment marks, i.e. '#'.
        Ag = Ag_gw
        Au = Au_gw


    Returns
    -------
    dict:
        Dictionary containing the potential assignation for each atom of
        the periodic table, with the shape:
        `{'H': 'H', 'He': 'He', ...}`.
    """
    # Attempting to read the file containing the user-defined
    # potential mappings. Defaults to none if no potential is given

    # Attempting to open the potential mapping file
    if not assign_dict:
        try:
            atom_array = np.loadtxt(f'{os.getcwd()}/potential_mapping', dtype=str)

            # Checking for 0D arrays which appear when only 1 potential is
            # specified
            atom_list = [str(atom_array)] if atom_array.ndim == 0 else list(atom_array)

            # Creating a dictionary with the custom potentials
            atom_dict = {}
            for udp in atom_list:
                sym, pot = udp.split('=')
                atom_dict[sym.title()] = pot

        except FileNotFoundError:
            atl_cut.custom_print(
                'No potential mapping file found. Using default potentials.', 'warning'
            )
            atom_dict = {}

    # Creating empty dict for the potential mapping
    potential_mapping = {}

    # Mapping every symbol on the periodic table to itself
    # unless it appears on the dictionary
    for symbol in ad.chemical_symbols[1:]:
        user_defined_potential = atom_dict.get(symbol)

        if user_defined_potential:
            potential_mapping[symbol] = user_defined_potential
        else:
            potential_mapping[symbol] = symbol

    return potential_mapping


def get_vdw_params(structure, incar):
    elements = structure.symbol_set
    new_incar = copy.deepcopy(incar)

    c6_ele_list = []
    r0_ele_list = []

    try:
        for element in elements:
            with open(VDW_DATA_PATH / element) as f:
                param_file = f.readlines()
            c6_ele_list.append(float(param_file[-2].strip()))
            r0_ele_list.append(float(param_file[-1].strip()))

        # if len(c6_ele_list) == 1:
        #     new_incar["vdw_c6"] = c6_ele_list[0]
        #     new_incar["vdw_r0"] = r0_ele_list[0]
        # else:
        new_incar['incar']['vdw_c6'] = c6_ele_list
        new_incar['incar']['vdw_r0'] = r0_ele_list

        atl_cut.custom_print(
            f"Gathered vdW info found for element '{element}'.", 'info'
        )

        return new_incar

    except FileNotFoundError:
        atl_cut.custom_print(
            f"No vdW info found for element '{element}', ignoring.", 'warn'
        )
        return incar


if __name__ == '__main__':
    atl_cut.custom_print('This file is not intented to be run as a script.', 'error')
