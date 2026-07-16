
```{role} alt
:class: code-alt
```
```{role} codeheader
:class: code-header
```

## Active Learning Loop

Generate a active learning template file using `atl_gen_configuration_file -t active_learning`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### Active Learning - `[active_learning]`

General active learning settings.


- {alt}`aiida_profile`:
  - **Description**: Name of the AiiDA profile to be used.
  - **Type**: `(str)`

- {alt}`enable_ntfysh`:
  - **Description**: Enable notifications using ntfy.sh for the run.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_name`:
  - **Description**: Internal name for the run.
  - **Type**: `(str)`

- {alt}`init_db_path`:
  - **Description**: Path to the folder containing the initial database.
  - **Type**: `(str, PosixPath)`

- {alt}`results_dir`:
  - **Description**: Path for final results. A folder named run_{uuid} will be created inside.
  - **Type**: `(str, PosixPath)`

- {alt}`reset_seed_db`:
  - **Description**: Whether to reset the seed database when resuming a stopped AL loop.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`log_path`:
  - **Description**: Path for the log file. Defaults to results_dir if not specified.
  - **Type**: `(optional, str, PosixPath)`

- {alt}`final_db_name`:
  - **Description**: Name for the final database (extxyz format).
  - **Type**: `(str)`
  - **Default**: `'final_data_test'`.

- {alt}`max_iterations`:
  - **Description**: Maximum number of AL loop iterations.
  - **Type**: `(int)`
  - **Default**: `3`.

- {alt}`load_init_models`:
  - **Description**: Load initial models from a list of AiiDA UUIDs/PKs. The format is `[UUID1/PK1, UUID2/PK2, ...]`.
  - **Type**: `(optional, list[int])`
  - **Default**: `[]`.

- {alt}`load_descriptor_calc`:
  - **Description**: Load a single descriptor calculation from a AiiDA UUID/PK.
  - **Type**: `(optional, str)`
  - **Default**: `''`.

- {alt}`load_md_calcs`:
  - **Description**: Load MD calculations from a list of AiiDA UUIDs/PKs.
  - **Type**: `(optional, list[int])`
  - **Default**: `[]`.

- {alt}`al_mode`:
  - **Description**: Active learning mode.
  - **Type**: `(optional, str)`
  - **Default**: `'data_acquisition'`.
  - Possible values are: `md`, `data_reduction`, `data_acquisition`.

### Active Learning Stopping Conditions - `[stop_conditions]`

Settings for the stopping conditions of the active learning loop.

:::{attention}
This section is optional.
:::


- {alt}`energy_threshold_mev`:
  - **Description**: Energy error threshold in meV/atom for the stopping criterion of the active learning loop. If not specified, a default of 43.0 is used.
  - **Type**: `(float)`
  - **Default**: `43.0`.

- {alt}`forces_threshold_mev_per_ang`:
  - **Description**: Forces error threshold in meV/Å for the stopping criterion of the active learning loop. If not specified, a default of 50.0 is used.
  - **Type**: `(float)`
  - **Default**: `50.0`.

- {alt}`stopping_threshold_type`:
  - **Description**: This option allows to select which error metric to compare with a threshold for the stopping criterion of the active learning loop. If not specified, this is not inclduded in the stopping criterion.
  - **Type**: `(str)`
  - **Default**: `'None'`.
  - Possible values are: `validation`, `testing`.

### Debug / Mock - `[debug]`

Developer debugging options for the active learning loop.

:::{attention}
This section is optional.
:::


#### Mock - `[debug.mock]`

Replace remote calcjobs (training, descriptors, MD, DFT) with instant local surrogates. For debugging loop logic only; results are physically meaningless.


- {alt}`enable`:
  - **Description**: Whether to enable mock mode.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`stages`:
  - **Description**: Stages to mock. If omitted or empty, all stages are mocked. Mocking 'training' implies its model-consuming stages (md, descriptors, test_db) are mocked too, since a placeholder model cannot feed a real remote job.
  - **Type**: `(optional, list[str])`
  - **Default**: `[]`.
  - Possible values are: `md`, `dft`, `training`, `descriptors`, `test_db`.

- {alt}`dft_potential`:
  - **Description**: Classical potential used for mock DFT energies/forces. 'emt' uses ASE EMT (falls back to Lennard-Jones for unsupported elements); 'lj' always uses Lennard-Jones.
  - **Type**: `(optional, str)`
  - **Default**: `'emt'`.
  - Possible values are: `emt`, `lj`.

- {alt}`md_rattle_stdev`:
  - **Description**: Standard deviation (Å) of the random perturbation applied to the seed frame in mock MD.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`md_n_frames`:
  - **Description**: Number of perturbed frames produced per seed structure in mock MD.
  - **Type**: `(optional, int)`
  - **Default**: `1`.

- {alt}`train_rmse_e_start`:
  - **Description**: Mock committee/test-DB energy RMSE (meV/atom) at iteration 0; decays geometrically over iterations.
  - **Type**: `(optional, float)`
  - **Default**: `90.0`.

- {alt}`train_rmse_f_start`:
  - **Description**: Mock committee/test-DB forces RMSE (meV/A) at iteration 0; decays geometrically over iterations.
  - **Type**: `(optional, float)`
  - **Default**: `275.0`.

- {alt}`train_rmse_decay`:
  - **Description**: Per-iteration geometric decay factor for mock RMSE curves (0-1).
  - **Type**: `(optional, float)`
  - **Default**: `0.7`.

### Test Set Settings - `[test_db]`

Settings for the test database used to evaluate model performance during active learning. The test database is defined at the start of the active learning run, and is kept constant throughout the entire process. It can either be generated at random from the initial database, or loaded from a file. The test set is only used for evaluation and is not included in the training data.

The test database evaluation is performed after training the committee of models of each active learning iteration, using the sampler model. The test set and the evaluation results are logged and saved in the base workchain and in each of the active learning loop steps as outputs.

Since the test database can be a subset of the training database, which is removed in place, a backup of the database is created under the same path and name as the original database, using the prefix `_original` and suffix `.bak`.


- {alt}`use_test_db`:
  - **Description**: Whether to use a constant test set during active learning.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`ignore_container`:
  - **Description**: Whether to ignore container settings for MACE training.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`test_db_path`:
  - **Description**: Path to the test set database file.
  - **Type**: `(optional, str, PosixPath)`

- {alt}`test_db_frac`:
  - **Description**: Fraction of the initial database to be used as test set if no test_db_path is provided.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

#### AiiDA Metadata for Test Set Calculations - `[test_db.metadata]`

AiiDA metadata options for the test set.

:::{attention}
This section is optional.
:::


- {alt}`computer`:
  - **Description**: Name of the AiiDA computer where the test set calculations were run.
  - **Type**: `(str)`

- {alt}`prepend_text`:
  - **Description**: Text to prepend to the AiiDA calculation label for the test set calculations.
  - **Type**: `(optional, str)`

##### AiiDA Options for Test Set Calculations - `[test_db.metadata.options]`

AiiDA options dictionary for the test set calculations.

:::{attention}
This section is optional.
:::


- {alt}`queue_name`:
  - **Description**: (SGE) Queue name.
  - **Type**: `(optional, str)`

- {alt}`qos`:
  - **Description**: (SLURM) QoS name.
  - **Type**: `(optional, str)`

- {alt}`account`:
  - **Description**: (SLURM) Account name.
  - **Type**: `(optional, str)`

- {alt}`max_wallclock_seconds`:
  - **Description**: Maximum wallclock time in seconds.
  - **Type**: `(optional, int)`

- {alt}`max_memory_kb`:
  - **Description**: Maximum memory in KB.
  - **Type**: `(optional, int)`

- {alt}`withmpi`:
  - **Description**: Whether to run with MPI.
  - **Type**: `(optional, bool)`

- {alt}`custom_scheduler_commands`:
  - **Description**: Custom scheduler commands.
  - **Type**: `(optional, str)`

###### Resources - `[test_db.metadata.options.resources]`

Resources dictionary.

:::{attention}
This section is optional.
:::


- {alt}`parallel_env`:
  - **Description**: Parallel environment.
  - **Type**: `(optional, str)`

- {alt}`tot_num_mpiprocs`:
  - **Description**: Total number of MPI processes.
  - **Type**: `(optional, int)`

- {alt}`num_cores_per_mpiproc`:
  - **Description**: Number of cores per MPI process.
  - **Type**: `(optional, int)`

- {alt}`num_machines`:
  - **Description**: Number of machines.
  - **Type**: `(optional, int)`

#### Model Settings for Test Set Evaluation - `[test_db.model_settings]`

Settings for the model used to evaluate the test set during active learning.


- {alt}`model_type`:
  - **Description**: Type of model to be used for test set evaluation.
  - **Type**: `(optional, str)`
  - **Default**: `'mace'`.
  - Possible values are: `mace`.

- {alt}`default_dtype`:
  - **Description**: Default data type for the model.
  - **Type**: `(optional, str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`device`:
  - **Description**: Device to be used for the model.
  - **Type**: `(optional, str)`
  - **Default**: `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

### Data Reduction Settings - `[data_reduction]`

Settings for the data reduction AL mode.


- {alt}`large_database_path`:
  - **Description**: Path to the large database file from which to select structures.
  - **Type**: `(str, PosixPath)`

- {alt}`initial_selection_size`:
  - **Description**: Number of structures to select from the large database for initial training.
  - **Type**: `(int)`
  - **Default**: `100`.

- {alt}`initial_selection_method`:
  - **Description**: Selection method for initial structures.
  - **Type**: `(str)`
  - **Default**: `'lowest_energy'`.
  - Possible values are: `random`, `lowest_energy`, `fps`.

- {alt}`structures_per_iteration`:
  - **Description**: Number of structures to select per iteration from the large database.
  - **Type**: `(int)`
  - **Default**: `50`.

- {alt}`iterative_selection_method`:
  - **Description**: Selection method for iterative structures.
  - **Type**: `(str)`
  - **Default**: `'uncertainty'`.
  - Possible values are: `random`, `fps`, `uncertainty`, `lowest_energy`.

### Active Learning Seed Settings - `[al_seed]`

Settings for the AL seed generation for MD.


- {alt}`seed_size_frac`:
  - **Description**: Sets total structures in an MD seed as a fraction of the training db size.
  - **Type**: `(float)`
  - **Default**: `0.1`.

- {alt}`seed_min_num_structs`:
  - **Description**: Minimum number of structures in an MD seed.
  - **Type**: `(optional, int)`
  - **Default**: `25`.

- {alt}`seed_max_num_structs`:
  - **Description**: Maximum number of structures in the MD seed.
  - **Type**: `(int)`
  - **Default**: `500`.

- {alt}`delete_seed_structs`:
  - **Description**: Whether to delete structures from the seed database even if they are in domain.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

#### Seed Selection Settings - `[al_seed.seed_select_settings]`

MD seed selection mode settings.


- {alt}`seed_select_type`:
  - **Description**: MD seed selection mode.
  - **Type**: `(str)`
  - **Default**: `'random'`.
  - Possible values are: `random`, `small_first`.

- {alt}`small_first_max_size`:
  - **Description**: Maximum size in number of atoms for structures selected with small_first mode.
  - **Type**: `(int)`
  - **Default**: `50`.

- {alt}`small_first_max_iter`:
  - **Description**: Apply small_first mode for the first n iterations.
  - **Type**: `(int)`
  - **Default**: `5`.

#### Seed Ranking Settings for Seed Selection - `[al_seed.seed_ranking_settings]`

Settings for the seed ranking methods.

:::{attention}
This section is optional.
:::


- {alt}`seed_ranking_algorithm`:
  - **Description**: Algorithm used for seed selection.
  - **Type**: `(optional, str)`
  - **Default**: `'random'`.
  - Possible values are: `random`, `descriptor_fps`.

##### Farthest Point Sampling (FPS) Configuration - `[al_seed.seed_ranking_settings.descriptor_fps]`

Farthest Point Sampling (FPS) ranking.


- {alt}`descriptor_type`:
  - **Description**: What descriptors to use for the seed selection process.
  - **Type**: `(optional, str)`
  - **Default**: `'soap'`.
  - Possible values are: `soap`, `mace`.

- {alt}`initial_structure`:
  - **Description**: Whether to gather a structure at random or select the one with the lowest energy available.
  - **Type**: `(optional, str)`
  - **Default**: `'random'`.
  - Possible values are: `random`, `lowest_energy`.

###### FPS Seed Selection - Selected Descriptor Settings - `[al_seed.seed_ranking_settings.descriptor_fps.descriptor]`

Entry containing settings that depend on the descriptor type selected.

:::{attention}
This section is optional.
:::


- {alt}`r_cut`:
  - **Description**: Cutoff radius for SOAP descriptor.
  - **Type**: `(optional, float)`
  - **Default**: `6.0`.

- {alt}`n_max`:
  - **Description**: Maximum number of radial basis functions for SOAP.
  - **Type**: `(optional, int)`
  - **Default**: `8`.

- {alt}`l_max`:
  - **Description**: Maximum degree of spherical harmonics for SOAP.
  - **Type**: `(optional, int)`
  - **Default**: `6`.

- {alt}`periodic`:
  - **Description**: Whether to consider the system as periodic.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`average`:
  - **Description**: Averaging mode for SOAP descriptor.
  - **Type**: `(optional, str)`
  - **Default**: `'off'`.
  - Possible values are: `inner`, `outer`, `off`.

- {alt}`model_path`:
  - **Description**: Path to the trained MACE model.
  - **Type**: `(optional, str)`

- {alt}`device`:
  - **Description**: What device to use for MACE.
  - **Type**: `(optional, str)`
  - **Default**: `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

- {alt}`dtype`:
  - **Description**: Floating point number precision for MACE.
  - **Type**: `(optional, str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`enable_cueq`:
  - **Description**: Enable CUEQ for MACE.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

### Interpolation Check Settings - `[interpolation]`

Settings for the interpolation check. The interpolation check determines whether a structure is within the model's knowledge domain by checking the committee models disagreement, considering the model accuracy threshold.


- {alt}`model_acc_multiplier`:
  - **Description**: Multiplier for model accuracy threshold. Higher values mean more DFT calculations.
  - **Type**: `(float)`
  - **Default**: `10.0`.

- {alt}`disagreement_check_type`:
  - **Description**: Approach for energy and force (E&F) committee disagreement check. With `training`, compare E&F with a threshold obtained from the training RMSE values multiplied by a threshold. With `md_threshold`, compare E&F with a threshold obtained from the standard deviaiton of the MD frames.
  - **Type**: `(str)`
  - **Default**: `'training'`.
  - Possible values are: `training`, `md_threshold`.

- {alt}`target_accuracy_e_meV_per_at`:
  - **Description**: Target accuracy for energy per atom in meV/atom. This should roughly correspond to the noise floor of your reference method, e.g., 1-2 meV/atom for DFT. By default, a relatively large value of 43 meV/atom is selected, since it correponds with a typical measure of chemical accuracy. However, this might be too high for some properties, such as long term diffusion or phase transitions.
  - **Type**: `(optional, float)`
  - **Default**: `43.0`.

- {alt}`target_accuracy_f_meV_per_A`:
  - **Description**: Target accuracy for force, in meV/Angstrom. This should roughly correspond to the noise floor of your reference method, e.g., 10-20 meV/atom for DFT. By default, a relatively large value of 50 meV/atom is selected, since it correponds with a typical measure of chemical accuracy.
  - **Type**: `(optional, float)`
  - **Default**: `50.0`.

### Extrapolation Check Settings - `[extrapolation]`

Settings for extrapolation checks.


- {alt}`check_extrapolation_type`:
  - **Description**: Method for extrapolation check. With `min-max` or `basic`, check for extrapolation using the range of the model descriptors. With `alpha-shape` or `advanced`, check for extrapolation using the concave hull of the model descriptors. With `disabled` or `none`, disable the extrapolation check, only leaving committee disagreement for EF for the domain.
  - **Type**: `(optional, str)`
  - **Default**: `'none'`.
  - Possible values are: `disabled`, `none`, `basic`, `min-max`, `alpha-shape`, `advanced`.

#### Concave Hull Extrapolation Check Settings - `[extrapolation.concave_hull]`

Settings for the concave hull / alpha shape extrapolation check.

:::{attention}
This section is optional.
:::


- {alt}`target_alpha_range_min`:
  - **Description**: Minimum alpha value for the concave hull.
  - **Type**: `(optional, float)`
  - **Default**: `1.0`.

- {alt}`target_alpha_range_max`:
  - **Description**: Maximum alpha value for the concave hull.
  - **Type**: `(optional, float)`
  - **Default**: `350.0`.

- {alt}`default_alpha_if_issues`:
  - **Description**: Default alpha value if there are issues with the concave hull generation.
  - **Type**: `(optional, float)`
  - **Default**: `10.0`.

- {alt}`nn_dist_scale_factor`:
  - **Description**: Scaling factor for the alpha candidate calculation, where `alpha_candidate = nn_dist_scale_factor / mean_nn_dist`
  - **Type**: `(optional, float)`
  - **Default**: `1.5`.

- {alt}`frac_points_allowed_out`:
  - **Description**: Maximum fraction of points allowed to be outside the concave hull. If the fraction of points outside the hull exceeds this value, alpha will be decreased iteratively until the condition is met or alpha reaches zero. Value is expressed as a fraction, thus 0.002 means 0.2%.
  - **Type**: `(optional, float)`
  - **Default**: `0.002`.

- {alt}`qt_offset_frac`:
  - **Description**: Offset fraction for the boundary of the root quadtree as a fraction of the data range. This will leave an extra margin around the data to avoid edge effects.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`qt_data_frac_capacity`:
  - **Description**: Fraction of the total number of data points to be used as the capacity of each quadtree node. Any quadtree node that goes above the capacity will be split further.
  - **Type**: `(optional, float)`
  - **Default**: `0.015`.

- {alt}`qt_subdivision_factor`:
  - **Description**: Subdivision factor for searching dense leaves of the quadtree. Higher values lead to more subdivisions and finer search.
  - **Type**: `(optional, int)`
  - **Default**: `4`.

- {alt}`concave_hull_scale_factor`:
  - **Description**: Scaling factor for the concave hull to be used when checking for extrapolation. For example, 0.1 results in a 10% size increase of the hull. A value of 0.0 means no scaling.
  - **Type**: `(optional, float)`
  - **Default**: `0.0`.

- {alt}`boundary_method`:
  - **Description**: Algorithm for boundary determination during the extrapolation check. With `concave_hull`, the boundary is the alpha shape of the latent space (per quadtree cluster). With `morphological_closing`, the boundary is obtained by rasterising the latent space and applying a morphological closing operation.
  - **Type**: `(optional, str)`
  - **Default**: `'concave_hull'`.
  - Possible values are: `concave_hull`, `morphological_closing`.

- {alt}`morph_disk_size`:
  - **Description**: Disk size for the morphological closing structuring element. Larger values bridge wider gaps between points. Only used when `boundary_method` is `morphological_closing`.
  - **Type**: `(optional, int)`
  - **Default**: `10`.

- {alt}`morph_threshold`:
  - **Description**: Grayscale intensity threshold (0-255) for point detection in morphological closing. Only used when `boundary_method` is `morphological_closing`.
  - **Type**: `(optional, int)`
  - **Default**: `250`.

- {alt}`morph_dpi`:
  - **Description**: DPI of the internal rasterisation used by morphological closing. Higher values give finer boundaries. Only used when `boundary_method` is `morphological_closing`.
  - **Type**: `(optional, int)`
  - **Default**: `100`.

### Active Learning Safeguard Settings - `[safeguard]`

Settings for active learning safeguard mechanisms. The safeguard will run long MD simulations on selected structures and perform an uncertainty quantification check in order to determine if the active learning loop is robust enough to stop at the current point.


- {alt}`enable`:
  - **Description**: Whether to enable the safeguard mechanism.
  - **Type**: `(bool)`
  - **Default**: `False`.

- {alt}`target_structure_mode`:
  - **Description**: Type of structures to include in the safeguard. Multiple types can be selected. If `base` is provided, the structures labelled as 'base' in the initial database will be used as target structures, after applying lateral expansion controlled with the base_struct_supercell_size option. If `target` is provided, a selection of targeted structures (via paths or atl_id) must be included through the `struct_target_list` option below.
  - **Type**: `(str)`
  - **Default**: `'base'`.
  - Possible values are: `base`, `target`.

- {alt}`base_struct_supercell_size`:
  - **Description**: 3D vector representing the lateral expansion factors to be used. Only used if `target_structure_mode` is set to `base`.
  - **Type**: `(optional, list[int])`
  - **Default**: `[3, 3, 3]`.

- {alt}`struct_target_list`:
  - **Description**: List of structure IDs or paths to be used as target structures in the safeguard. Only used if `target_structure_mode` is set to `target`.
  - **Type**: `(optional, list[str, int])`

- {alt}`ignore_container`:
  - **Description**: Whether to ignore the container specified in the container settings for the safeguard.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

#### Safeguard MD Settings - `[safeguard.md]`

MD simulation options for safeguard.


##### Safeguard MD Parameters - `[safeguard.md.parameters]`

MD simulation parameters for safeguard. These settings will override the general MD settings for the safeguard simulations. This section supports the same parameters as the `md.parameters` section.


- {alt}`temperature_list_K`:
  - **Description**: List of different temperatures (in K) for MD simulations.
  - **Type**: `(list[float])`
  - **Default**: `[300.0, 500.0, 900.0]`.


- {alt}`sample_frames_during_md`:
  - **Description**: Whether to sample frames during MD simulations for AL.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.


- {alt}`al_keep_struct_every_n_ps`:
  - **Description**: Keep a structure every N picoseconds of MD simulation.
  - **Type**: `(float)`
  - **Default**: `0.5`.


- {alt}`log_save_interval`:
  - **Description**: Log energy and force information every N MD steps.
  - **Type**: `(optional, int)`
  - **Default**: `1`.


- {alt}`num_at_large_struct`:
  - **Description**: Number of structures with a number of atoms larger than `large_struct_size` to consider for MD simulations.
  - **Type**: `(optional, int)`
  - **Default**: `'None'`.


- {alt}`max_temp_multiplier`:
  - **Description**: Multiplier for MD temperature to determine the upper bound of the temperature.
  - **Type**: `(float)`
  - **Default**: `1.3`.


- {alt}`num_steps`:
  - **Description**: Total number of timesteps for each MD simulation.
  - **Type**: `(int)`
  - **Default**: `33334`.


- {alt}`timestep_duration_ps`:
  - **Description**: Duration of each timestep in picoseconds.
  - **Type**: `(float)`
  - **Default**: `0.003`.


- {alt}`md_thermostat`:
  - **Description**: Ensemble used in the MD simulation. The ensemble parameter changes the selection of the ASE integrator and thermostat and barostat. `langevin` and `nvt` ensembles use the ASE Langevin thermostat. `npt` and `npt-mtk` uses ASE's implementation of the Full Martyna-Tobias-Klein (MTK) method [^1], similar to the one used for NPT in LAMMPS. The `npt-melchionna` ensemble uses a combined Nose-Hoover and Parrinello-Rahman  method, as proposed by Melchionna et al. and provided in ASE as the default NPT class. However, this implementation is not recommended for use by ASE due stability issues!  
[^1] MTKNPT: https://ase-lib.org/ase/md.html#full-martyna-tobias-klein-mtk-dynamics
  - **Type**: `(optional, str)`
  - **Default**: `'langevin'`.
  - Possible values are: `langevin`, `nvt`, `npt`, `npt-mtk`, `npt-melchionna`.


- {alt}`langevin_friction_ps-1`:
  - **Description**: Friction coefficient for the Langevin thermostat in picoseconds.
  - **Type**: `(float)`
  - **Default**: `10.0`.


- {alt}`npt_ttime_fs`:
  - **Description**: Time constant for temperature coupling in NPT thermostat in femtoseconds.
  - **Type**: `(float)`
  - **Default**: `100.0`.


- {alt}`npt_ptime_fs`:
  - **Description**: Time constant for pressure coupling in NPT thermostat in femtoseconds.
  - **Type**: `(float)`
  - **Default**: `25.0`.


- {alt}`md_stage_order`:
  - **Description**: List containing the names of the stages, written in the order in which the MD stages will be applied. Each stage must be defined under `md.parameters.stages`.
  - **Type**: `(list[str])`
  - **Default**: `[]`.


- {alt}`gather_traj_cnt_lattice`:
  - **Description**: Consider constant lattice when gathering trajectories.
  - **Type**: `(bool)`
  - **Default**: `True`.


- {alt}`use_kokkos`:
  - **Description**: Whether to use Kokkos to run the MD.
  - **Type**: `(bool)`
  - **Default**: `True`.


- {alt}`device`:
  - **Description**: Device for the MACE model in MD simulations.
  - **Type**: `(str)`
  - **Default**: `'cuda'`.
  - Possible values are: `cpu`, `cuda`.


- {alt}`enable_cueq`:
  - **Description**: Enable CUEQ for MACE.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.


- {alt}`default_dtype`:
  - **Description**: Default data type for the MACE model in MD simulations.
  - **Type**: `(str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.


- {alt}`max_energy_threshold_per_atom`:
  - **Description**: Maximum energy threshold per atom in eV.
  - **Type**: `(optional, float)`
  - **Default**: `1000.0`.


- {alt}`num_cpus_large_struct`:
  - **Description**: Number of CPUs to use for structures larger than `large_struct_size`.
  - **Type**: `(optional, int)`
  - **Default**: `16`.


#### Metadata - `[safeguard.metadata]`

AiiDA metadata and scheduler options for the safeguard.


- {alt}`computer`:
  - **Description**: AiiDA computer name for safeguard calculations.
  - **Type**: `(str)`
  - **Example**: `'my_cluster'`.

- {alt}`prepend_text`:
  - **Description**: Text to prepend to job scripts for AiiDA.
  - **Type**: `(optional, str)`
  - **Example**: `'module load singularity
export PATH=$PATH:.'`.

- {alt}`options`:
  - **Description**: AiiDA scheduler options for safeguard calculations.
  - **Type**: `(dict)`

### Molecular Dynamics Settings - `[md]`

Settings for MD simulations.


- {alt}`ignore_container`:
  - **Description**: Whether to ignore the container specified in the container settings for the MD calculations.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

#### Parameters - `[md.parameters]`

MD simulation parameters.


- {alt}`temperature_list_K`:
  - **Description**: List of different temperatures (in K) for MD simulations.
  - **Type**: `(list[float])`
  - **Default**: `[300.0, 500.0, 900.0]`.

- {alt}`sample_frames_during_md`:
  - **Description**: Whether to sample frames during MD simulations for AL.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`al_keep_struct_every_n_ps`:
  - **Description**: Keep a structure every N picoseconds of MD simulation.
  - **Type**: `(float)`
  - **Default**: `0.5`.

- {alt}`log_save_interval`:
  - **Description**: Log energy and force information every N MD steps.
  - **Type**: `(optional, int)`
  - **Default**: `1`.

- {alt}`num_at_large_struct`:
  - **Description**: Number of structures with a number of atoms larger than `large_struct_size` to consider for MD simulations.
  - **Type**: `(optional, int)`
  - **Default**: `'None'`.

- {alt}`max_temp_multiplier`:
  - **Description**: Multiplier for MD temperature to determine the upper bound of the temperature.
  - **Type**: `(float)`
  - **Default**: `1.3`.

- {alt}`num_steps`:
  - **Description**: Total number of timesteps for each MD simulation.
  - **Type**: `(int)`
  - **Default**: `33334`.

- {alt}`timestep_duration_ps`:
  - **Description**: Duration of each timestep in picoseconds.
  - **Type**: `(float)`
  - **Default**: `0.003`.

- {alt}`md_thermostat`:
  - **Description**: Ensemble used in the MD simulation. The ensemble parameter changes the selection of the ASE integrator and thermostat and barostat. `langevin` and `nvt` ensembles use the ASE Langevin thermostat. `npt` and `npt-mtk` uses ASE's implementation of the Full Martyna-Tobias-Klein (MTK) method [^1], similar to the one used for NPT in LAMMPS. The `npt-melchionna` ensemble uses a combined Nose-Hoover and Parrinello-Rahman  method, as proposed by Melchionna et al. and provided in ASE as the default NPT class. However, this implementation is not recommended for use by ASE due stability issues!  
[^1] MTKNPT: https://ase-lib.org/ase/md.html#full-martyna-tobias-klein-mtk-dynamics
  - **Type**: `(optional, str)`
  - **Default**: `'langevin'`.
  - Possible values are: `langevin`, `nvt`, `npt`, `npt-mtk`, `npt-melchionna`.

- {alt}`langevin_friction_ps-1`:
  - **Description**: Friction coefficient for the Langevin thermostat in picoseconds.
  - **Type**: `(float)`
  - **Default**: `10.0`.

- {alt}`npt_ttime_fs`:
  - **Description**: Time constant for temperature coupling in NPT thermostat in femtoseconds.
  - **Type**: `(float)`
  - **Default**: `100.0`.

- {alt}`npt_ptime_fs`:
  - **Description**: Time constant for pressure coupling in NPT thermostat in femtoseconds.
  - **Type**: `(float)`
  - **Default**: `25.0`.

- {alt}`md_stage_order`:
  - **Description**: List containing the names of the stages, written in the order in which the MD stages will be applied. Each stage must be defined under `md.parameters.stages`.
  - **Type**: `(list[str])`
  - **Default**: `[]`.

- {alt}`gather_traj_cnt_lattice`:
  - **Description**: Consider constant lattice when gathering trajectories.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`use_kokkos`:
  - **Description**: Whether to use Kokkos to run the MD.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`device`:
  - **Description**: Device for the MACE model in MD simulations.
  - **Type**: `(str)`
  - **Default**: `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- {alt}`enable_cueq`:
  - **Description**: Enable CUEQ for MACE.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`default_dtype`:
  - **Description**: Default data type for the MACE model in MD simulations.
  - **Type**: `(str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`max_energy_threshold_per_atom`:
  - **Description**: Maximum energy threshold per atom in eV.
  - **Type**: `(optional, float)`
  - **Default**: `1000.0`.

- {alt}`num_cpus_large_struct`:
  - **Description**: Number of CPUs to use for structures larger than `large_struct_size`.
  - **Type**: `(optional, int)`
  - **Default**: `16`.

##### MD Simulation Stages - `[md.parameters.stages]`

Collection of named keys representing a particular stage of an MD simulation. These stages will be chained together and applied to single MD calculation job following the order defined in `md.parameters.md_stage_order`, resulting in a continuous MD calculation with different settings for every stage. Each stage key will contain contain settings to be applied during each stage, overriding the default MD settings specified under `md.parameters`. Each dictionary must contain a mandatory `use_during_al_steps` or `use_for_structure_types` key, which specifies when given stage must be run. If none of these two keys are not present, the stage will be ignored.


###### MD Simulation Stages - `[md.parameters.stages.XXXXX]`

This key describes settings for dynamic entries. Several entries can be added by using different key names.

The key name (`XXXXX`) is used as the reference name. **Replace XXXXX with a name of your choice.**

Accepted parameters for each entry:


- {alt}`use_during_al_steps`:
  - **Description**: String representing a number of AL steps or interval of AL steps in which the stage will be used. For example, `0` means that the stage will be used during step 0, while `3-6` means that the stage will be used from AL step 3 to AL step 6 (inclusive). The example `'0-1, 3, 4, 5-9, 11'` would execute the current stage for steps 0, 1, 3, 4, 5, 6, 7, 8, 9, and 11.
  - **Type**: `(optional, str)`
  - **Example**: `'0-1, 3, 4, 5-9, 11'`.

- {alt}`use_for_structure_types`:
  - **Description**: List of structure types for which to use the current stage. Structure types must be among those defined in the initial database (e.g., `bulk`, `surface`, `cluster`), if a structure contains multiple types, the stage will be applied if at least one type matches for the initial structure in the MD simulation.
  - **Type**: `(optional, list[str])`
  - Possible values are: `bulk`, `surface`, `cluster`.

- {alt}`temperature_list_K`:
  - **Description**: List of different temperatures (in K) for MD simulations.
  - **Type**: `(list[float])`
  - **Default**: `[300.0, 500.0, 900.0]`.


- {alt}`sample_frames_during_md`:
  - **Description**: Whether to sample frames during MD simulations for AL.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.


- {alt}`al_keep_struct_every_n_ps`:
  - **Description**: Keep a structure every N picoseconds of MD simulation.
  - **Type**: `(float)`
  - **Default**: `0.5`.


- {alt}`log_save_interval`:
  - **Description**: Log energy and force information every N MD steps.
  - **Type**: `(optional, int)`
  - **Default**: `1`.


- {alt}`num_at_large_struct`:
  - **Description**: Number of structures with a number of atoms larger than `large_struct_size` to consider for MD simulations.
  - **Type**: `(optional, int)`
  - **Default**: `'None'`.


- {alt}`max_temp_multiplier`:
  - **Description**: Multiplier for MD temperature to determine the upper bound of the temperature.
  - **Type**: `(float)`
  - **Default**: `1.3`.


- {alt}`num_steps`:
  - **Description**: Total number of timesteps for each MD simulation.
  - **Type**: `(int)`
  - **Default**: `33334`.


- {alt}`timestep_duration_ps`:
  - **Description**: Duration of each timestep in picoseconds.
  - **Type**: `(float)`
  - **Default**: `0.003`.


- {alt}`md_thermostat`:
  - **Description**: Ensemble used in the MD simulation. The ensemble parameter changes the selection of the ASE integrator and thermostat and barostat. `langevin` and `nvt` ensembles use the ASE Langevin thermostat. `npt` and `npt-mtk` uses ASE's implementation of the Full Martyna-Tobias-Klein (MTK) method [^1], similar to the one used for NPT in LAMMPS. The `npt-melchionna` ensemble uses a combined Nose-Hoover and Parrinello-Rahman  method, as proposed by Melchionna et al. and provided in ASE as the default NPT class. However, this implementation is not recommended for use by ASE due stability issues!  
[^1] MTKNPT: https://ase-lib.org/ase/md.html#full-martyna-tobias-klein-mtk-dynamics
  - **Type**: `(optional, str)`
  - **Default**: `'langevin'`.
  - Possible values are: `langevin`, `nvt`, `npt`, `npt-mtk`, `npt-melchionna`.


- {alt}`langevin_friction_ps-1`:
  - **Description**: Friction coefficient for the Langevin thermostat in picoseconds.
  - **Type**: `(float)`
  - **Default**: `10.0`.


- {alt}`npt_ttime_fs`:
  - **Description**: Time constant for temperature coupling in NPT thermostat in femtoseconds.
  - **Type**: `(float)`
  - **Default**: `100.0`.


- {alt}`npt_ptime_fs`:
  - **Description**: Time constant for pressure coupling in NPT thermostat in femtoseconds.
  - **Type**: `(float)`
  - **Default**: `25.0`.


- {alt}`md_stage_order`:
  - **Description**: List containing the names of the stages, written in the order in which the MD stages will be applied. Each stage must be defined under `md.parameters.stages`.
  - **Type**: `(list[str])`
  - **Default**: `[]`.


- {alt}`gather_traj_cnt_lattice`:
  - **Description**: Consider constant lattice when gathering trajectories.
  - **Type**: `(bool)`
  - **Default**: `True`.


- {alt}`use_kokkos`:
  - **Description**: Whether to use Kokkos to run the MD.
  - **Type**: `(bool)`
  - **Default**: `True`.


- {alt}`device`:
  - **Description**: Device for the MACE model in MD simulations.
  - **Type**: `(str)`
  - **Default**: `'cuda'`.
  - Possible values are: `cpu`, `cuda`.


- {alt}`enable_cueq`:
  - **Description**: Enable CUEQ for MACE.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.


- {alt}`default_dtype`:
  - **Description**: Default data type for the MACE model in MD simulations.
  - **Type**: `(str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.


- {alt}`max_energy_threshold_per_atom`:
  - **Description**: Maximum energy threshold per atom in eV.
  - **Type**: `(optional, float)`
  - **Default**: `1000.0`.


- {alt}`num_cpus_large_struct`:
  - **Description**: Number of CPUs to use for structures larger than `large_struct_size`.
  - **Type**: `(optional, int)`
  - **Default**: `16`.


#### MD Trajectory Filters - `[md.filters]`

Settings for MD trajectory filters.


- {alt}`save_filtered_structures`:
  - **Description**: Whether to save filtered structures.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

##### Check_Atoms_No_Neighbor - `[md.filters.check_atoms_no_neighbor]`

Filter for structures with atoms that have no neighbors.


- {alt}`enable`:
  - **Description**: No description available.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`covalent_radius_multiplier`:
  - **Description**: Multiplier for covalent radii to define cutoff for neighbor check.
  - **Type**: `(float)`
  - **Default**: `1.05`.

##### Layer_Distance - `[md.filters.layer_distance]`

Filter for layer distances in surface slabs.


- {alt}`enable`:
  - **Description**: No description available.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`max_layer_distance_ang`:
  - **Description**: Maximum accepted distance between layers in Angstrom.
  - **Type**: `(float)`
  - **Default**: `3.5`.

##### Exploding_Structures - `[md.filters.exploding_structures]`

Filter for exploding structures based on covalent radius limits.


- {alt}`enable`:
  - **Description**: Whether to enable the exploding structures filter.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`cov_rad_multiplier_max`:
  - **Description**: Maximum multiplier for covalent radius threshold.
  - **Type**: `(optional, float)`
  - **Default**: `10.0`.

- {alt}`cov_rad_multiplier_min`:
  - **Description**: Minimum multiplier for covalent radius threshold.
  - **Type**: `(optional, float)`
  - **Default**: `1.5`.

- {alt}`explode_check_interval_perc`:
  - **Description**: Interval percentage (as a fraction) of MD steps to check for exploding structures.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`max_F`:
  - **Description**: Maximum force component allowed in the structure.
  - **Type**: `(optional, float)`
  - **Default**: `25.0`.

- {alt}`max_V`:
  - **Description**: Maximum velocity component allowed in the structure.
  - **Type**: `(optional, float)`
  - **Default**: `2.0`.

#### MD Metadata and Scheduler Options (AiiDA) - `[md.metadata]`

AiiDA metadata and scheduler options for MD simulations.


- {alt}`code`:
  - **Description**: AiiDA code name for MD software.
  - **Type**: `(optional, str)`
  - **Example**: `'mace_lammps@cluster'`.

- {alt}`computer`:
  - **Description**: AiiDA computer name for MD calculations.
  - **Type**: `(str)`
  - **Example**: `'my_cluster'`.

- {alt}`prepend_text`:
  - **Description**: Text to prepend to job scripts for AiiDA.
  - **Type**: `(optional, str)`
  - **Example**: `'module load singularity
export PATH=$PATH:.'`.

- {alt}`options`:
  - **Description**: AiiDA scheduler options for MD calculations.
  - **Type**: `(optional, dict)`

### Code and Containerization Settings - `[code]`

Settings for containerized code execution.


#### Container - `[code.container]`

Container settings for code execution.


- {alt}`use_container`:
  - **Description**: Whether to use a containerized version of the code.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`image_name`:
  - **Description**: Path to the container image on calculation nodes.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/container.sif'`.

- {alt}`engine_command`:
  - **Description**: Command template for the container engine.
  - **Type**: `(optional, str)`
  - **Example**: `'singularity exec --bind .:/atl_data --nv --contain --writable-tmpfs {image_name}'`.

- {alt}`prepend_text`:
  - **Description**: Text to prepend to job scripts for container setup.
  - **Type**: `(optional, str)`
  - **Example**: `'module load singularity
export PATH=$PATH:.'`.

### MLIP Backend Settings - `[mlip]`

Configuration for the MLIP backend used in the active learning workflow. Supports a modular backend system with a plugin interface for adding new models. If this section is absent, the workflow defaults to MACE for all stages.


- {alt}`training_backend`:
  - **Description**: Which MLIP backend to use for model training, committee evaluation, and calculator creation. Available backends are registered in the atlas.active_learning.backends registry.
  - **Type**: `(optional, str)`
  - **Default**: `'mace'`.

- {alt}`md_backend`:
  - **Description**: Which MLIP backend to use for MD simulations. Defaults to the training_backend value if not specified.
  - **Type**: `(optional, str)`
  - **Example**: `'mace'`.

### MLIP Training Settings - `[mace_train]`

Settings for MACE model training. Used when mlip.training_backend is 'mace' (default).

:::{attention}
This section is optional.
:::


- {alt}`result_force_weight`:
  - **Description**: Weight of the force when considering model performance in weighted sum calculation.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`test_fraction`:
  - **Description**: Fraction of the training data to be used for testing.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`code`:
  - **Description**: AiiDA code name for MLIP training. Optional when using PortableCode (omit to use the bundled training script).
  - **Type**: `(optional, str)`
  - **Example**: `'mace_train@cluster'`.

- {alt}`computer`:
  - **Description**: AiiDA computer name for MLIP training. Required when 'code' is not set (PortableCode mode).
  - **Type**: `(optional, str)`
  - **Example**: `'my_cluster'`.

- {alt}`ignore_container`:
  - **Description**: Whether to ignore container settings for MACE training.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`multihead_finetuning`:
  - **Description**: Whether to use multihead finetuning.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`metadata`:
  - **Description**: AiiDA metadata and scheduler options for MACE training.
  - **Type**: `(optional, dict)`

- {alt}`train_settings`:
  - **Description**: MACE training parameters and hyperparameters.
  - **Type**: `(optional, dict)`

### Allegro Training Settings - `[allegro_train]`

Settings for Allegro model training. Used when mlip.training_backend is 'allegro'.

:::{attention}
This section is optional.
:::


- {alt}`result_force_weight`:
  - **Description**: Weight of the force when considering model performance in weighted sum calculation.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`code`:
  - **Description**: AiiDA code name for Allegro training. Optional when using PortableCode.
  - **Type**: `(optional, str)`
  - **Example**: `'nequip-train@cluster'`.

- {alt}`computer`:
  - **Description**: AiiDA computer name for Allegro training. Required when 'code' is not set.
  - **Type**: `(optional, str)`
  - **Example**: `'my_cluster'`.

- {alt}`ignore_container`:
  - **Description**: Whether to ignore container settings for Allegro training.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`metadata`:
  - **Description**: AiiDA metadata and scheduler options for Allegro training.
  - **Type**: `(optional, dict)`

- {alt}`num_workers`:
  - **Description**: Number of DataLoader workers for Allegro training.
  - **Type**: `(optional, int)`
  - **Default**: `4`.

- {alt}`batch_size`:
  - **Description**: Batch size for Allegro training.
  - **Type**: `(optional, int)`
  - **Default**: `1`.

- {alt}`train_settings`:
  - **Description**: Allegro/NequIP training parameters and hyperparameters.
  - **Type**: `(optional, dict)`

### Committee Evaluation Settings - `[committee_eval]`

Settings for committee evaluation using multiple MLIP models.


- {alt}`committee_num_models`:
  - **Description**: Total number of MLIP models in the committee.
  - **Type**: `(optional, int)`
  - **Default**: `4`.

- {alt}`openmp_threads`:
  - **Description**: Number of OpenMP threads for MLIP CPU evaluation.
  - **Type**: `(optional, int)`
  - **Default**: `24`.

- {alt}`ignore_container`:
  - **Description**: Whether to ignore container settings for committee evaluation.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`prepend_text`:
  - **Description**: Text to prepend to job scripts for committee evaluation.
  - **Type**: `(optional, str)`

- {alt}`metadata`:
  - **Description**: AiiDA metadata and scheduler options for committee evaluation.
  - **Type**: `(optional, dict)`

#### Committee Evaluation - MACE Settings - `[committee_eval.mace]`

Settings for MACE evaluator. Used when training_backend is 'mace'.

:::{attention}
This section is optional.
:::


- {alt}`device`:
  - **Description**: Device for MLIP evaluation.
  - **Type**: `(optional, str)`
  - **Default**: `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

- {alt}`default_dtype`:
  - **Description**: Default data type for MLIP evaluation.
  - **Type**: `(optional, str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`batch_size`:
  - **Description**: Batch size for MLIP evaluation.
  - **Type**: `(optional, int)`
  - **Default**: `32`.

- {alt}`compute_stress`:
  - **Description**: Whether to compute stress during evaluation.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

#### Committee Evaluation - Allegro Settings - `[committee_eval.allegro]`

Settings for Allegro evaluator. Used when training_backend is 'allegro'.

:::{attention}
This section is optional.
:::


- {alt}`device`:
  - **Description**: Device for MLIP evaluation.
  - **Type**: `(optional, str)`
  - **Default**: `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

- {alt}`default_dtype`:
  - **Description**: Default data type for MLIP evaluation.
  - **Type**: `(optional, str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`batch_size`:
  - **Description**: Batch size for MLIP evaluation.
  - **Type**: `(optional, int)`
  - **Default**: `32`.

- {alt}`compute_stress`:
  - **Description**: Whether to compute stress during evaluation.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

### Descriptor Computation Settings - `[descriptors]`

Settings for descriptor computation and dimensionality reduction.


- {alt}`descriptor_type`:
  - **Description**: Type of descriptor to compute.
  - **Type**: `(str)`
  - **Default**: `'mace'`.
  - Possible values are: `mace`, `soap`.

- {alt}`dimensionality_reduction_method`:
  - **Description**: Dimensionality reduction method for model descriptors.
  - **Type**: `(optional, str)`
  - **Default**: `'none'`.
  - Possible values are: `autoencoder`, `pca`, `none`.

- {alt}`ignore_container`:
  - **Description**: Whether to ignore container settings for descriptor computation.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`metadata`:
  - **Description**: AiiDA metadata for descriptor computation.
  - **Type**: `(dict)`

- {alt}`dtype`:
  - **Description**: Data type of descriptor to compute.
  - **Type**: `(optional, str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`device`:
  - **Description**: Device for descriptor computation.
  - **Type**: `(optional, str)`
  - **Default**: `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

#### Autoencoder Dimensionality Reduction Settings - `[descriptors.autoencoder]`

Settings for autoencoder-based dimensionality reduction.


##### Train_Settings - `[descriptors.autoencoder.train_settings]`

Training settings for the autoencoder.


- {alt}`device`:
  - **Description**: Device for autoencoder training.
  - **Type**: `(optional, str)`
  - **Default**: `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- {alt}`dtype`:
  - **Description**: Data type for autoencoder training.
  - **Type**: `(optional, str)`
  - **Default**: `'float32'`.
  - Possible values are: `float32`, `float64`.

- {alt}`model_path`:
  - **Description**: Path to save the autoencoder model.
  - **Type**: `(optional, str)`
  - **Default**: `'autoencoder_model.pth'`.

- {alt}`load_model`:
  - **Description**: Whether to load the model from the model path.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`dataset`:
  - **Description**: Path to the training dataset.
  - **Type**: `(optional, str)`
  - **Default**: `'all_descriptors.npz'`.

- {alt}`l1_hidden_dim`:
  - **Description**: Number of units in the first hidden layer.
  - **Type**: `(optional, int)`
  - **Default**: `256`.

- {alt}`l2_hidden_dim`:
  - **Description**: Number of units in the second hidden layer.
  - **Type**: `(optional, int)`
  - **Default**: `32`.

- {alt}`bottleneck_dim`:
  - **Description**: Dimensionality of the bottleneck (latent space).
  - **Type**: `(optional, int)`
  - **Default**: `2`.

- {alt}`bias_flag`:
  - **Description**: Flag to include bias terms in the layers.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`num_epochs`:
  - **Description**: Number of epochs to train the model.
  - **Type**: `(int)`
  - **Default**: `50`.

- {alt}`batch_size`:
  - **Description**: Batch size for training.
  - **Type**: `(optional, int)`
  - **Default**: `2048`.

- {alt}`patience`:
  - **Description**: Early stopping patience (epochs without val loss improvement). Also controls LR reduction schedule.
  - **Type**: `(optional, int)`
  - **Default**: `5`.

- {alt}`lr`:
  - **Description**: Learning rate for the optimizer.
  - **Type**: `(optional, float)`
  - **Default**: `0.001`.

- {alt}`weight_decay`:
  - **Description**: L2 regularization parameter.
  - **Type**: `(optional, float)`
  - **Default**: `'1e-5'`.

- {alt}`loss`:
  - **Description**: Loss function type.
  - **Type**: `(optional, str)`
  - **Default**: `'mse'`.
  - Possible values are: `mse`, `mae`.

- {alt}`train_frac`:
  - **Description**: Fraction of the data to use for training.
  - **Type**: `(optional, float)`
  - **Default**: `0.8`.

- {alt}`valid_frac`:
  - **Description**: Fraction of the data to use for validation.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`test_frac`:
  - **Description**: Fraction of the data to use for testing.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`wandb`:
  - **Description**: Whether to log metrics to wandb.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`wandb_name`:
  - **Description**: Name of the wandb run.
  - **Type**: `(optional, str)`
  - **Default**: `''`.

- {alt}`wandb_project`:
  - **Description**: Name of the wandb project.
  - **Type**: `(optional, str)`
  - **Default**: `''`.

### DFT Calculation Settings - `[dft]`

DFT settings specific to active learning (different from top-level dft section).


- {alt}`ignore_container`:
  - **Description**: Whether to ignore container settings for DFT calculations.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`dft_method`:
  - **Description**: Selection of DFT calculator.
  - **Type**: `(optional, str)`
  - **Default**: `'mace'`.
  - Possible values are: `vasp`, `mace`.

- {alt}`calc_type`:
  - **Description**: Type of calculation.
  - **Type**: `(optional, str)`
  - **Default**: `'single_point'`.
  - Possible values are: `single_point`, `relax`, `static`.

- {alt}`dft_calc_limit`:
  - **Description**: Maximum number of DFT calculations to perform per AL step.
  - **Type**: `(optional, int)`
  - **Default**: `'None'`.

#### MACE DFT Calculator Settings - `[dft.mace]`

MACE settings as DFT calculator.

:::{attention}
This section is optional.
:::


- {alt}`mace_potential_path`:
  - **Description**: Path to MACE potential file.
  - **Type**: `(str)`
  - **Example**: `'model.model'`.

- {alt}`metadata`:
  - **Description**: AiiDA metadata for MACE calculations.
  - **Type**: `(optional, dict)`

- {alt}`options`:
  - **Description**: AiiDA scheduler options for MACE calculations.
  - **Type**: `(optional, dict)`

##### Settings - `[dft.mace.settings]`

Options for MACE that will be passed as arguments during execution.


- {alt}`device`:
  - **Description**: Device for MACE calculations.
  - **Type**: `(optional, str)`
  - **Default**: `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- {alt}`default_dtype`:
  - **Description**: Default data type for MACE calculations.
  - **Type**: `(optional, str)`
  - **Default**: `'float64'`.
  - Possible values are: `float32`, `float64`.

- {alt}`batch_size`:
  - **Description**: Batch size for MACE calculations.
  - **Type**: `(optional, int)`
  - **Default**: `11`.

- {alt}`compute_stress`:
  - **Description**: Whether to compute stress.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

##### Filter - `[dft.mace.filter]`

Settings for filtering structures after DFT calculations, based on energy and force thresholds.

:::{attention}
This section is optional.
:::


- {alt}`filter_dft_calcs`:
  - **Description**: Whether to filter structures based on energy and force thresholds.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`threshold_E_meV`:
  - **Description**: Energy threshold in meV/atom for filtering structures.
  - **Type**: `(optional, float)`
  - **Default**: `1000.0`.

- {alt}`threshold_F_meV`:
  - **Description**: Force threshold in meV/Å for filtering structures.
  - **Type**: `(optional, float)`
  - **Default**: `10000.0`.

#### VASP DFT Calculator Settings - `[dft.vasp]`

VASP settings as DFT calculator.

:::{attention}
This section is optional.
:::


- {alt}`calc_type`:
  - **Description**: Type of calculation.
  - **Type**: `(optional, str)`
  - **Default**: `'static'`.
  - Possible values are: `static`, `relax`.

- {alt}`dft_calc_limit`:
  - **Description**: Maximum number of DFT calculations to perform per AL step.
  - **Type**: `(optional, int)`
  - **Default**: `200`.

- {alt}`potential_family`:
  - **Description**: VASP potential family name.
  - **Type**: `(str)`
  - **Example**: `'vasp-5.4-PBE-2024'`.

- {alt}`structure_types`:
  - **Description**: List of structure types to process.
  - **Type**: `(optional, list[str])`
  - **Default**: `['bulk', 'surface', 'cluster']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- {alt}`kspacing`:
  - **Description**: K-spacing settings for different phases or default value. The k-spacing values will be applied on a per-phase basis, according to the keys added to the dictionary which must match phases in the database. The `ATL_DEFAULT` key can be used to set a default k-spacing value for all phases not explicitly listed.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'ATL_DEFAULT': 0.15, 'alpha': 0.135}
```

- {alt}`incar`:
  - **Description**: INCAR settings for VASP calculations.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'encut': 450, 'gga': 'Pe', 'icharg': 2, 'istart': 0}
```

##### Filter - `[dft.vasp.filter]`

Settings for filtering structures after DFT calculations, based on energy and force thresholds.

:::{attention}
This section is optional.
:::


- {alt}`filter_dft_calcs`:
  - **Description**: Whether to filter structures based on energy and force thresholds.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`threshold_E_meV`:
  - **Description**: Energy threshold in meV/atom for filtering structures.
  - **Type**: `(optional, float)`
  - **Default**: `1000.0`.

- {alt}`threshold_F_meV`:
  - **Description**: Force threshold in meV/Å for filtering structures.
  - **Type**: `(optional, float)`
  - **Default**: `10000.0`.

##### Queue - `[dft.vasp.queue]`

Queue settings for VASP calculations.


- {alt}`queue_type`:
  - **Description**: Scheduler type.
  - **Type**: `(str)`
  - **Default**: `'slurm'`.

- {alt}`computer`:
  - **Description**: AiiDA computer name for VASP calculations.
  - **Type**: `(str)`
  - **Example**: `'aiida-computer-name'`.

- {alt}`code_string`:
  - **Description**: Name of the VASP code as defined in AiiDA.
  - **Type**: `(str)`
  - **Example**: `'vasp@computer'`.

- {alt}`withmpi`:
  - **Description**: Whether to run with MPI.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`qos`:
  - **Description**: Quality of service parameter.
  - **Type**: `(optional, str)`
  - **Example**: `'gp_partition'`.

- {alt}`account`:
  - **Description**: Account to be used for calculations.
  - **Type**: `(optional, str)`
  - **Example**: `'account_name'`.

- {alt}`node_cpus`:
  - **Description**: Number of CPUs per node.
  - **Type**: `(optional, int)`
  - **Default**: `48`.

- {alt}`max_wallclock_seconds`:
  - **Description**: Maximum wallclock time in seconds.
  - **Type**: `(optional, int)`
  - **Default**: `28800`.

- {alt}`options_resources`:
  - **Description**: Resource options for the scheduler.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'num_machines': 1, 'tot_num_mpiprocs': 112}
```

- {alt}`multiple`:
  - **Description**: Multiple factor for resources.
  - **Type**: `(optional, int)`
  - **Default**: `1`.

- {alt}`custom_scheduler_commands`:
  - **Description**: Custom scheduler commands.
  - **Type**: `(optional, str)`

##### Surface - `[dft.vasp.surface]`

Surface-specific INCAR settings.


- {alt}`incar`:
  - **Description**: INCAR settings specific to surface calculations.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'idipol': 3, 'ispin': 2, 'ldipol': True}
```

##### Cluster - `[dft.vasp.cluster]`

Cluster-specific INCAR settings.


- {alt}`incar`:
  - **Description**: INCAR settings specific to cluster calculations.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'dipol': [0.5, 0.5, 0.5], 'idipol': 4, 'ldipol': True}
```

##### AiiDA-VASP Settings - `[dft.vasp.aiida_vasp]`

AiiDA-VASP specific settings.


- {alt}`parser_settings`:
  - **Description**: Contains entries to include in the results gathered using the aiida-vasp parser settings
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{   'add_bands': False,
    'add_born_charges': False,
    'add_charge_density': False,
    'add_dielectrics': False,
    'add_dos': False,
    'add_dynmat': False,
    'add_energies': True,
    'add_forces': False,
    'add_hessian': False,
    'add_kpoints': False,
    'add_misc': True,
    'add_projectors': False,
    'add_stress': False,
    'add_structure': False,
    'add_trajectory': False,
    'add_wavecar': False}
```

- {alt}`critical_notifications`:
  - **Description**: Critical error and warning notifications. These represent VASP errors and warnings to be treated as critical, which will result in an error code being thrown by the aiida calculation job. The example contains all defults.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{   'add_bandocc': True,
    'add_brmix': True,
    'add_cnormn': True,
    'add_denmp': True,
    'add_dentet': True,
    'add_edddav_zhegv': True,
    'add_eddrmm_zhegv': True,
    'add_edwav': True,
    'add_fexcp': True,
    'add_fock_acc': True,
    'add_magmom': True,
    'add_no_potimm': True,
    'add_non_collinear': True,
    'add_not_hermitian': True,
    'add_psmaxn': True,
    'add_pzstein': True,
    'add_real_optlay': True,
    'add_rhosyg': True,
    'add_rspher': True,
    'add_set_indpw_full': True,
    'add_sgrcon': True}
```
