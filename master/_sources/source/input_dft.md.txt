
```{role} alt
:class: code-alt
```
```{role} codeheader
:class: code-header
```

## DFT Calculations

Generate a dft template file using `atl_gen_configuration_file -t dft`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General - `[general]`

General settings for the DFT script.


- {alt}`log_path`:
  - **Description**: Path where the logs will be stored.
  - **Type**: `(str, PosixPath)`
  - **Default**: `'/tmp/'`.

- {alt}`result_file_path`:
  - **Description**: Path for the results file (extxyz format).
  - **Type**: `(str, PosixPath)`
  - **Default**: `'dft_calculation_results'`.

- {alt}`source_db`:
  - **Description**: Path to the source database file (.extxyz or mdb .xz format).
  - **Type**: `(optional, str, PosixPath)`

- {alt}`aiida_group_name`:
  - **Description**: Name of the AiiDA group for the calculations.
  - **Type**: `(str)`
  - **Example**: `'my_dft_run'`.

- {alt}`max_batch`:
  - **Description**: Maximum number of structures to process in one batch.
  - **Type**: `(int)`
  - **Default**: `100`.

- {alt}`queue_check_interval_seconds`:
  - **Description**: Interval in seconds to check the queue for submitting new calculations.
  - **Type**: `(int)`
  - **Default**: `240`.

- {alt}`start_on_struct_idx`:
  - **Description**: Number of structures to skip before starting the calculations.
  - **Type**: `(int)`
  - **Default**: `0`.

- {alt}`dry_run`:
  - **Description**: If True, a dry-run is performed and no calculations are submitted.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`selected_structure_type`:
  - **Description**: Structure types to process. If empty, all types are included.
  - **Type**: `(optional, list[str])`
  - Possible values are: `bulk`, `surface`, `cluster`.

### Calculation - `[calculation]`

DFT calculation settings.


- {alt}`calc_type`:
  - **Description**: Type of calculation.
  - **Type**: `(str)`
  - **Default**: `'static'`.
  - Possible values are: `static`, `relaxation`.

- {alt}`aiida_potential_family`:
  - **Description**: AiiDA potential family name.
  - **Type**: `(str)`
  - **Example**: `'vasp-5.4-PBE-2023'`.

- {alt}`potential_mapping`:
  - **Description**: Mapping of elements to specific potential labels.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'Si': 'Si_GW'}
```

### Kpoints - `[kpoints]`

K-point settings.


- {alt}`kspacing`:
  - **Description**: K-spacing in Å⁻¹ for different phases or a single value for all structures. The k-spacing values will be applied on a per-phase basis, according to the keys added to the dictionary which must match phases in the database. The `ATL_DEFAULT` key can be used to set a default k-spacing value for all phases not explicitly listed.
  - **Type**: `(dict)`
  - **Example**:

```python
{'ATL_DEFAULT': 0.15, 'alpha': 0.125}
```

### Queue - `[queue]`

Queue settings for HPC schedulers (e.g., SLURM).


- {alt}`code_string`:
  - **Description**: Name of the code as defined in AiiDA.
  - **Type**: `(str)`
  - **Example**: `'vasp@my_cluster'`.

- {alt}`account`:
  - **Description**: Account to be used for the calculations.
  - **Type**: `(optional, str)`

- {alt}`qos`:
  - **Description**: Quality of service parameter.
  - **Type**: `(optional, str)`

- {alt}`node_cpus`:
  - **Description**: Number of CPUs per node.
  - **Type**: `(int)`

- {alt}`max_wallclock_seconds`:
  - **Description**: Maximum wallclock time in seconds.
  - **Type**: `(int)`
  - **Default**: `16200`.

- {alt}`max_memory_kb`:
  - **Description**: Maximum memory per node in KB.
  - **Type**: `(optional, int)`

- {alt}`multiple`:
  - **Description**: Whether to use multiple nodes.
  - **Type**: `(bool)`
  - **Default**: `False`.

- {alt}`custom_scheduler_commands`:
  - **Description**: Custom scheduler commands.
  - **Type**: `(optional, str)`

#### Options_Resources - `[queue.options_resources]`

Scheduler resource options.


- {alt}`tot_num_mpiprocs`:
  - **Description**: Total number of MPI processes.
  - **Type**: `(int)`
  - **Default**: `24`.

- {alt}`parallel_env`:
  - **Description**: Parallel environment to be used.
  - **Type**: `(str)`
  - **Default**: `' '`.

### Aiida_Vasp - `[aiida_vasp]`

Settings for the AiiDA-VASP plugin.

:::{attention}
This section is optional.
:::


- {alt}`critical_notifications`:
  - **Description**: Errors and warnings to be treated as critical (general).
  - **Type**: `(optional, dict)`

#### Parser_Settings - `[aiida_vasp.parser_settings]`

Parser settings for aiida-vasp.


- {alt}`add_kpoints`:
  - **Description**: Whether to add k-points information to the parsed results.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

##### Critical_Notifications - `[aiida_vasp.parser_settings.critical_notifications]`

Critical error and warning notifications to be treated as important.


- {alt}`add_edddav_zhegv`:
  - **Description**: Add EDDDAV ZHEGV error notification.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`add_eddrmm_zhegv`:
  - **Description**: Add EDDRMM ZHEGV error notification.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`add_not_hermitian`:
  - **Description**: Add not hermitian error notification.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`add_brmix`:
  - **Description**: Add BRMIX error notification.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`add_bandocc`:
  - **Description**: Add band occupation error notification.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

### Incar - `[incar]`

INCAR settings for different structure types.


- {alt}`bulk`:
  - **Description**: INCAR settings for bulk structures.
  - **Type**: `(optional, dict)`

- {alt}`surface`:
  - **Description**: INCAR settings for surface structures.
  - **Type**: `(optional, dict)`

- {alt}`cluster`:
  - **Description**: INCAR settings for cluster structures.
  - **Type**: `(optional, dict)`
