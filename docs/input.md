## Input specification
The input format of MDB is [TOML](https://toml.io/en/). The syntax from TOML is unchanged. The available parameters are different depending on the selected tool.

Users are advised to use the `mdb_conf_gen` utility to generate a template file which can be customized.

## Database Generation

👷 WIP 🕒

## Active Learning Loop

### Key - Active learning - `[al_learning]`

This key describes the main active learning settings:


- `aiida_profile`: (str) Name of the aiida profile to be used.
- `run_name`: (str) Internal name for the run
- `data_path`: (str) Path to the folder where the initial database is contained. **#check @pol**
- `init_db_path`: (str) Path to the folder where the initial database is contained. **#check @pol**
- `results_dir`: (str) Path for final results. Will be created if not existent.
It will contain a folder named `run_{uuid}`.
- `final_db_name`: (str) Name for the final database. The database will be stored in the extxyz format
- `max_iterations`: (int) Maximum number of AL loop iterations.
- `model_acc_multiplier`: (float)
Multiplier for model accuracy. Loosens model accuracy threshold. Tighter thresholds (lower values) will result in more DFT calculations. Any RMSE E/F values above chem_acc*chem_acc_multiplier will be considered wrong.
- `al_keep_struct_every_n_ps`: (float) Every how many ps of MD simulation keep a structure. 
Influences the total number of energy evaluations and possibly DFT calculations.
- `check_extrapolation`: (bool) Whether to check for extrapolation using the MACE descriptors
- `dft_method`: (str) Selection of energy/force calculator. Options: "vasp", "mace"
- `load_init_models`: (list[int], optional) # Load initial models from several aiida uuids/pk.



### Key - Active learning seed - `[al_seed]`

Parameters to configure the AL seed generation

- `seed_size_frac`: (float) Fraction of the training db set to be used to create the AL seed. This influences the amount of MD calculations and E F evaluations.
- `seed_max_num_structs`: (int) Maximum number of structures in the MD seed.

#### Key - Seed selection settings - `[al_seed.seed_select_settings]`

Parameters to tune the structure selection while creating the AL seeds.

- `seed_select_type`: MD seed selection mode. `random` selects random structures from the seed pool `small_first` selects random structures smaller than small_first_max_size for the first small_first_max_iter iters. random (default) / small_first

- `small_first_max_size`: (int) Maximum size in number of atoms for the structures selected with small_first mode


- `small_first_max_iter`: (int) Apply small_first mode for the first n iterations



### Key - MD Settings - `[md]`

Settings for MD simulations using LAMMPS

#### Key - MD Parameters - `[md.parameters]`


- `temperature_list_K`: (list[float]) List of different temperatures (in K) to be used for the MD simulations. Example: [300, 350, 400]
- `max_temp_multiplier`: (int) Multiplier for the user-specified MD temperature to determine the upper bound of the temperature at the end of the simulation run. Set to 1 to disable the multiplier.
- `num_steps` (int) Total number of MD steps to be run in each MD simulation
- `timestep_duration_ps`: (float) Duration of each timestep (in ps). **check unit, ps or fs**
- `gather_traj_cnt_lattice`: (bool) Consider constant lattice when gathering trajectories
- `use_kokkos`: (bool) Whether to use kokkos to run the LAMMPS MD on gpu

#### Key - MD Filters - `[md.filters]`

Contains settings related to the filtering of structures obtained from MD calculations. Filtering allows the removal of some types of incorrect structures that might pollute the training database


- `check_atoms_no_neighbor`: (bool) Check for structures that have atoms with no neighbors. Specific setting for the neighbor check MD filter.  ⚠ **WARNING: only use when dealing with bulks, surfaces or clusters that have no adsorbed molecules.** ⚠
- `layer_distance.max_layer_distance_ang`: (float) Specific setting for the layer distance MD filter.Maximum accepted distance between layers (in Angstrom).

#### Key - MD Queue - `[md.queue]`
Contains settings related to the MD calculation setup and usage of AiiDA. As of now, the only MD code available is LAMMPS-MACE.
The queue key can take any option from its [matching AiiDA input](https://aiida.readthedocs.io/projects/aiida-core/en/stable/topics/calculations/usage.html#options). Below are listed the bare minimum options for running calculations using an SGE scheduler:

 - `code` (str) AiiDA code name for MD software to be used
 AiiDA scheduler options for MD
- `metadata.options.resources.parallel_env`: (str) name for the parallel environment
- `metadata.options.resources.tot_num_mpiprocs`: (int) Total number of mpi processors
- `metadata.options.queue_name`: (str) Name of the SGE queue
- `metadata.options.max_memory_kb`: (int) Max memory allowed in kB.
- `metadata.options.max_wallclock_seconds`: (int) Maximum requested wall time.
- `metadata.options.withmpi`: (bool) Use MPI.
- `metadata.options.custom_scheduler_commands`: (str) Use this to include extra options, such as GPU allocation on SGE or extra commands. Use triple quoted strings to allow for several options. E.g.: 
```python
'''
#$ -l gpu=1
#$ -l hostname="tekla2189"
'''
```

  
### Key - Committee Evaluation Settings - `[committee_eval]`
Contains settings related to the NNP model evaluation (MACE as of now).

- `committee_num_models`: (int) Total number of MACE models.  Increases resources necessary for training.
- `openmp_threads` (int): Number of OpenMP threads to be used for MACE CPU evaluation.
- `prepend_text`: prepend text for the aiida [PortableCode](https://aiida.readthedocs.io/projects/aiida-core/en/stable/topics/data_types.html#portablecode). Use triple quoted strings to allow for several options.

#### Key - MACE General Settings - `[committee_eval.mace]`

Contains settings for the MACE evaluator.

- `device`: (str) Device to use for the training, either `cpu` or `cuda`
- `default_dtype`: (str) Whether to use single precision or double precision floating point numbers. Either `float32` or  `float64`
- `batch_size`: (int) Size of the training batch
- `compute_stress`: (bool) Whether or not to compute stress (true/false)

#### Key - MACE General Settings - `[committee_eval.metadata.options]

Contains settings related to the scheduler and AiiDA for the MACE Evaluations. This key can take any option from its [matching AiiDA input](https://aiida.readthedocs.io/projects/aiida-core/en/stable/topics/calculations/usage.html#options). Below are listed the bare minimum options for running calculations using an SGE scheduler:

- `parser_name`: (str) "mace-committee-eval-parser"
- `resources.parallel_env`: (str) "c128m1024ib_mpi_32slots"
- `resources.tot_num_mpiprocs`: 32
- `queue_name`: (str) "c128m1024ibgpu4.q"
- `max_wallclock_seconds`: 117280000
- `max_memory_kb`: 102400000
- `withmpi`: false
- `custom_scheduler_commands`: 
- `computer` = "tekla2-new-test"

```
'''#$ -l gpu=1
$ -l hostname="tekla2189"'''
```

### Key - Descriptor Settings - `[descriptors]`

### Key - Descriptor Scheduler Optinons - `[descriptors.metadata]`
AiiDA metadata settings for MACE descriptor calculation

- `computer` = ''
- `account` = ''
- `qos` = ''
- `parser_name` = ''
- `prepend_text` = ''
- `options.max_wallclock_seconds` = ''
- `options.resources.num_cores_per_mpiproc` = ''
- `options.resources.tot_num_mpiprocs` = ''
- `options.resources.num_machines` = ''
- `options.withmpi` = ''


### Key - MACE Scheduler - `[mace_train]`
MACE training code and scheduler settings (aiida)

- `result_force_weight` (float): Weight of the force when considering model performance used in:  weighted_sum = RMSE_E + (force_weight * RMSE_F). The lowest weighted_sum will be considered as the most performant model.
- `code`: (str) AiiDA code name
- `metadata.options.resources.parallel_env`: (str)
- `metadata.options.resources.tot_num_mpiprocs` = 32
- `metadata.options.parser_name` = "mace-training-parser"
- `metadata.options.queue_name` = "c128m1024ibgpu4.q"
- `metadata.options.max_wallclock_seconds` = 117280000
- `metadata.options.max_memory_kb` = 102400000
- `metadata.options.withmpi` = true
- `metadata.options.custom_scheduler_commands`: (str)
```
'''#$ -l gpu=1
$ -l hostname="tekla2188"
'''
```

#### Key - MACE Train Settings - `[mace_train.train_settings]`

MACE Training Settings. Check the [MACE documentation on training](https://mace-docs.readthedocs.io/en/latest/guide/training.html) for more information.

- `name` (str) 
- `energy_key` (str) "energy"
- `valid_fraction` (float) 0.1
- `config_type_weights`: { Default = 1.0 }
- `weight_decay`: 9.34e-07
- `E0s`: "average"
- `num_interactions`: 2
- `model`: "MACE"
- `correlation`: 3
- `hidden_irreps`: "16x0e + 16x1o"
- `lr`: 0.005626773506534471
- `r_max`: 6.0
- `max_ell`: 3
- `max_L`: 2
- `batch_size`: 64
- `max_num_epochs`: 35
- `swa`: true
- `ema`: true
- `ema_decay`: 0.99
- `amsgrad`: true
- `restart_latest`: true
- `device`: (str) Either `cuda`/`cpu`
- `default_dtype`: (str) Either float32/float64
- `wandb`: (bool)


## Key - DFT Settings - `[dft]`

### Key - MACE as DFT calculator - `[dft.mace]`
MACE Settings as DFT calculator. Ignored if dft_method = "vasp" 
Options intended for MACE will be passed as arguments during MACE execution. The scheduler options will be used in the builder.metadata.options from AiiDA.

- `mace_potential_path`: (str) Path to MACE potential file
- `device`: "cuda"           # Options: "cpu", "cuda"
- `default_dtype`: "float32" # Options: "float32", "float64"
- `batch_size`: 64
- `compute_stress`: true     # Options: true, false
- `options.parser_name`: "mace-eval-parser"
- `options.code_string`: "mace_run_eval_gpu@tekla2-new-test"
- `options.resources`: { parallel_env = "c128m1024ib_mpi_32slots", tot_num_mpiprocs = 32 }
- `options.max_wallclock_seconds`: 117280000
- `options.max_memory_kb`: 102400000
- `options.custom_scheduler_commands`: (str)
```
'''#$ -l gpu=1
$ -l hostname="tekla2189"
'''
```
- `options.withmpi`: false


### Key - VASP as DFT calculator - `[dft.vasp]`

Settings for VASP as DFT calculator. Ignored if dft_method = "mace" 

- `potential_family` = "vasp-5.3-PBE"
- `potential_family` = "vasp-5.4-PBE-2023"
- `structure_types` = ['bulk', 'surface', 'cluster']

[dft.vasp.queue]
- `queue.type` = "sge"
- `queue.node_cpus` = 12
- `queue.code_string` = "vasp-std-5.4.4-new@tekla2"
- `queue.options_resources` = { parallel_env = "c12m48ib_mpi", tot_num_mpiprocs = 12 }
- `queue.multiple` = 1
- `queue.custom_scheduler_commands` = '#$ -l hostname="tekla2044"'


### Key - VASP k-spacing - `[dft.vasp.kspacing]`

Description of the phase diagram. **WIP**
- `alpha` = 0.135088484104361
- `m1` = 0.100530964914873
- `beta`-prime = 0.102415920507027
- `m2` = 0.100530964914873
- `gamma` = 0.141371669411541
- `m3` = 0.166504410640259
- `epsilon` = 0.105557513160617
- `eta` = 0.0993371597065093
- `m4` = 0.0948760981384118
- `delta` = 0.0994491889005363

### Key - VASP INCAR - `[dft.vasp.incar]`

General incar settings to be used as a template for all calculations. Different types of calculations, i.e., relaxations, bulks, clusters and surfaces, can have different options. Type-specific options must be specified in the corresponding key for each type (see below) and will overwrite the keys on the general incar in this section.

- `istart`: 0
- `icharg`: 2
- `gga`: "Pe"
- `ispin`: 1
- `encut`: 450    # electronic steps
- `ediff`: 1e-6
- `ismear`: 0
- `sigma`: 0.03
- `algo`: "Fast"
- `lreal`: "Auto"
- `nelm`: 60      # ionic steps
- `ibrion`: -1
- `nsw`: 1
- `ediffg`: -0.03
- `isif`: 2
- `potim`: 0.3
- `lwave`: false
- `lcharg`: false
- `ncore`: 4
- `lelf`: false
- `ivdw`: 11      # van der Waals


#### Key - VASP INCAR for relaxations - `[dft.vasp.relax.incar]`

- `ibrion` = 2
- `nsw` = 350
- `isif` = 3

#### Key - VASP INCAR for surfaces - `[dft.vasp.surface.incar]`
- `ldipol` = true
- `idipol` = 3

#### Key - VASP INCAR for clusters - `[dft.vasp.cluster.incar]`
- `ldipol` = true
- `dipol` = [0.5, 0.5, 0.5]
- `idipol` = 4
