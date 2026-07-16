
```{role} alt
:class: code-alt
```
```{role} codeheader
:class: code-header
```

## DFT Parameter Benchmark

The DFT Parameter Benchmark tool tests VASP parameters (kspacing, ENCUT, ISPIN, …) per crystallographic phase to find optimal settings that satisfy a given convergence threshold.
For each parameter sweep it:
- Selects one representative bulk structure per phase.
- Submits calculations via AiiDA for each parameter value.
- Analyses convergence and recommends the cheapest settings within the threshold.
- Generates convergence plots and a copy-pasteable TOML snippet.

Generate a dft benchmark template file using `atl_gen_configuration_file -t dft_benchmark`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General - `[general]`

General settings for the benchmark.


- {alt}`log_path`:
  - **Description**: Path where the logs will be stored.
  - **Type**: `(str, PosixPath)`
  - **Default**: `'/tmp/'`.

- {alt}`aiida_group_name`:
  - **Description**: Name of the AiiDA group for benchmark calculations.
  - **Type**: `(str)`
  - **Default**: `'dft_benchmark'`.

- {alt}`max_batch`:
  - **Description**: Maximum number of calculations in the queue at once.
  - **Type**: `(int)`
  - **Default**: `50`.

- {alt}`queue_check_interval_seconds`:
  - **Description**: Interval in seconds to check the queue.
  - **Type**: `(int)`
  - **Default**: `240`.

- {alt}`dry_run`:
  - **Description**: If True, list calculations without submitting.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

### Database - `[database]`

Database settings.


- {alt}`db_path`:
  - **Description**: Path to the initial database (extxyz format).
  - **Type**: `(str, PosixPath)`

### Calculation - `[calculation]`

DFT calculation settings.


- {alt}`calc_type`:
  - **Description**: Type of calculation (benchmarks use static).
  - **Type**: `(str)`
  - **Default**: `'static'`.
  - Possible values are: `static`.

- {alt}`aiida_potential_family`:
  - **Description**: AiiDA potential family name.
  - **Type**: `(str)`
  - **Example**: `'vasp-5.4-PBE-2023'`.

- {alt}`potential_mapping`:
  - **Description**: Mapping of elements to specific potential labels.
  - **Type**: `(optional, dict)`

### Kpoints - `[kpoints]`

Base k-point settings (used as reference for non-kspacing benchmarks).


- {alt}`kspacing`:
  - **Description**: K-spacing in Å⁻¹. Use ATL_DEFAULT for a single value.
  - **Type**: `(dict)`
  - **Example**:

```python
{'ATL_DEFAULT': 0.05}
```

### Queue - `[queue]`

Queue settings for HPC schedulers.


- {alt}`code_string`:
  - **Description**: Name of the code as defined in AiiDA.
  - **Type**: `(str)`

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
  - **Description**: Parallel environment.
  - **Type**: `(str)`
  - **Default**: `' '`.

### Incar - `[incar]`

Base INCAR settings (only bulk needed for benchmarks).


- {alt}`bulk`:
  - **Description**: Base INCAR settings for bulk structures.
  - **Type**: `(optional, dict)`

### Benchmark sweep configuration. Each sub-key defines a parameter to test. - `[benchmark]`

Benchmark sweep configuration. Each sub-key defines a parameter to test.


#### Benchmark sweep configuration. Each sub-key defines a parameter to test. - `[benchmark.XXXXX]`

This key describes settings for dynamic entries. Several entries can be added by using different key names.

The key name (`XXXXX`) is used as the reference name. **Replace XXXXX with a name of your choice.**

Accepted parameters for each entry:


- {alt}`values`:
  - **Description**: List of parameter values to test.
  - **Type**: `(list)`

- {alt}`reference_value`:
  - **Description**: Explicit reference value (overrides auto-detection).
  - **Type**: `(optional, float)`

- {alt}`direction`:
  - **Description**: Which end is tightest: 'min' (smallest) or 'max' (largest).
  - **Type**: `(optional, str)`
  - Possible values are: `min`, `max`.
