# CLI Arguments

This section describes the command line arguments for various scripts provided by the ATLAS library. These arguments allow users to control the behavior and configuration of active learning loops, database generation, and other features. Each argument is explained along with its usage and default value (if applicable).

## Run Active Learning - `atl_active_learning`

Handles ATLAS active learning loop runs. Running the `atl_active_learning` command without any subcommand or the gui subcommand will start a new run.
In order to do this, users must provide a TOML settings file which can be configured as seen in the [inputs section](input_active_learning.md).

Other available subcommands allow to generate reports or resume loops using files generated from previous runs.

**Usage:** `atl_active_learning <SUBCOMMAND> <OPTIONS>`

### General Arguments

- `-c, --config_file` (PATH)
  - **Description**: Path pointing to a TOML settings file. By default, `active_learning_settings.toml` will be searched in the current working directory (CWD).
  - **Type**: `Path`
  - **Default**: `./active_learning_settings.toml`

### Subcommands

Subcommands define the operating mode of the `atl_active_learning` tool.

#### `report` - Generate reports

Generates different types of reports for a ATLAS active learning loop or a ATLAS initial database.

**Usage:** `atl_active_learning report <REPORT_TYPE> [OPTIONS...]`

The `report` command requires specifying a `REPORT_TYPE` as a positional argument, followed by options specific to that report type.

**Available `REPORT_TYPE`s:**

- `init_db`: Generate a report for a ATLAS initial database.
- `al_loop`: Generate a report for an active learning loop by providing an AiiDA PK/UUID or a log file path.
- `al_loop_batch`: Generate a report for a series of active learning loops, as a way of comparing them.
- `al_loop_performance`: Generate a performance report for an active learning loop.

##### Report type: `init_db`

Generates a report for a ATLAS initial database. This can include plots of energy and force distributions, with options to handle outliers and customize data presentation.

**Usage:** `atl_active_learning report init_db [-h] --db_path <PATH> [OPTIONS...]`

- `--db_path <PATH>, -d <PATH>`
  - **Description**: Path to the database file.
  - **Required**: Yes (as per usage string).
- `--threshold_E <FLOAT>`
  - **Description**: Threshold for Energy (E) to consider a structure as an outlier, in eV. If `--per_atom` is enabled, this threshold is in eV/atom.
  - **Type**: `float`
- `--threshold_F <FLOAT>`
  - **Description**: Threshold for Force (F) to consider a structure as an outlier, in eV/Å (or eV/Å per atom if `--per_atom` is enabled).
  - **Type**: `float`
- `--remove_outliers`
  - **Description**: If specified, removes outliers (defined by `--threshold_E` and `--threshold_F`) from the plot.
  - **Default**: Outliers are included.
- `--color_type {phase,struct_type}`
  - **Description**: Specifies the criterion to use for coloring the plot.
  - **Type**: `Choice`
  - **Choices**: `phase`, `struct_type`
- `--per_atom`
  - **Description**: If specified, displays energy and force values normalized per atom.
  - **Default**: Values are not normalized per atom.
- `-h, --help`
  - **Description**: Show this help message and exit.

##### Report type: `al_loop`

Generates a comprehensive report for a single active learning loop, identified by an AiiDA PK/UUID or a ATLAS log file path. This report can include plots on database sizes (training and seed), NNP performance metrics, the number of structures added or removed, error analysis plots comparing DFT and NN predictions, and visualization of database evolution in latent space.

**Usage:** `atl_active_learning report al_loop [-h] (--loop_id <ID> | --log_path <PATH>) [OPTIONS...]`

- `--loop_id <ID>, -i <ID>`
  - **Description**: AiiDA PK/UUID of the active learning loop. (This or `--log_path` is required).
- `--log_path <PATH>, -log <PATH>`
  - **Description**: Path to the ATLAS log of the active learning loop. (This or `--loop_id` is required).
- `--device <DEVICE>, -d <DEVICE>`
  - **Description**: String representing a device to run inference on (e.g., 'cpu', 'cuda:0').
  - **Type**: `str`
- `--get_error_plot`
  - **Description**: If specified, generates an error plot comparing DFT and NN predictions.
  - **Default**: Not set.
- `--remove_outliers`
  - **Description**: If specified (and relevant, e.g., for an error plot), removes outliers from the plot. Thresholds for outlier removal might be specified by other options (e.g., `--threshold_E_meV`, `--threshold_F_meV`, `--threshold_meV`).
  - **Default**: Outliers are included.
- `--model <PATH>`
  - **Description**: Path to the model file (e.g., for generating error plots).
  - **Type**: `Path`
- `--database <PATH>`
  - **Description**: Path to the database file (e.g., for error plots or latent space analysis).
  - **Type**: `Path`
- `--threshold_E_meV <FLOAT>`
  - **Description**: Threshold for the difference between $E_{DFT}$ and $E_{NN}$ (in meV) to mark a structure as an outlier in error plots.
  - **Type**: `float`
  - **Default**: `150` (meV)
- `--threshold_F_meV <FLOAT>`
  - **Description**: Threshold for the difference between $F_{DFT}$ and $F_{NN}$ (in meV/Å, assuming typical units for forces converted to meV scale for comparison) to mark a structure as an outlier in error plots.
  - **Type**: `float`
  - **Default**: `25000` (meV)
- `--title <TITLE>`
  - **Description**: Custom run name to be used as the title for plots. If not provided, a name will be gathered from the calculation files.
  - **Type**: `str`
- `--db_latent_space_evolution`
  - **Description**: If specified, plots the database evolution over time in latent space. Requires `--autoencoder_folder`.
  - **Default**: Not set.
- `--autoencoder_folder <PATH>`
  - **Description**: Path to the autoencoder folder. This folder should contain an autoencoder model and the database files for every step, required for `--db_latent_space_evolution`.
  - **Type**: `Path`
- `--db_path <PATH>, --d <PATH>`
  - **Description**: Path to the database file (alternative or additional way to specify a database, context might depend on other flags).
  - **Type**: `Path`
- `-h, --help`
  - **Description**: Show this help message and exit.

##### Report type: `al_loop_batch`

Generates a comparative report for a series of active learning loops. This allows for assessing and contrasting the outcomes or behaviors of multiple loops, typically identified by their AiiDA PKs/UUIDs or log file paths.

**Usage:** `atl_active_learning report al_loop_batch [OPTIONS...]`
*(Note: The specific command-line options for `al_loop_batch` were not detailed in the provided help snippets. You may need to run `atl_active_learning report al_loop_batch --help` directly or consult the tool's source for a complete list of options. It is expected to take identifiers for multiple loops, possibly similar to `--loop_ids` in `al_loop_performance`.)*

- **Description (from general help)**: "Generate a report for a series of active learning loops, as a way of comparing them by providing an AiiDA PK/UUID or a log file path."
- **Description (from existing markdown)**: "Generate a report for a series of active learning loops, as a way of comparing them by providing an AiiDA PK."
- `-h, --help`
  - **Description**: Show this help message and exit.

##### Report type: `al_loop_performance`

Generates a performance report for an active learning loop, focusing on computational aspects. The plots typically display step durations and computational resource usage, which can be useful for identifying bottlenecks or understanding the efficiency of the loop. Multiple AiiDA PKs/UUIDs can be provided for multi-stage loops (e.g., when a loop was resumed).

**Usage:** `atl_active_learning report al_loop_performance [-h] (--loop_id <ID> | --log_path <PATH>) [OPTIONS...]`

- `--loop_ids <ID> [<ID> ...], -i <ID> [<ID> ...]`
  - **Description**: AiiDA PK/UUIDs of the active learning loop(s). Several IDs can be provided for multi-stage loops (e.g., when using `resume`) or to analyze multiple distinct loops in a performance context. Get IDs from your log or terminal output.
  - **Type**: `List[str|int]`
- `--output_filename <PATH>, -o <PATH>`
  - **Description**: Filename for the report plot to be generated.
  - **Type**: `Path`
- `-h, --help`
  - **Description**: Show this help message and exit.

#### `resume` - Resume Stopped AL Loops

Resumes an active learning loop using a results folder. A resumed run will use the settings from the `.toml` file contained in the results folder.

**Usage**: `atl_active_learning resume [-h] --dir_resume <PATH> [--config_file <PATH>]`

- `--dir_resume, -d` (`<PATH>`)
  - **Description**: Path to the results directory of a previous active learning loop run.
  - **Required**: Yes
- `--config_file, -c` (`<PATH>`)
  - **Description**: Path pointing to a TOML settings file. Optional, as all calculation folders should contain a TOML file.
  - **Default**: `None`

#### `gui` - Launch dashboard

Launches a dashboard to track the active learning loop.

**Usage:** `run_active_learning gui [-h] [--update_interval n_sec] [--port port] [--debug] [--online]`

- `--update_interval` (`n_sec`)
  - **Description**: Refresh time interval in seconds.
  - **Type**: `int`
  - **Default**: `60`
- `--port` (`port`)
  - **Description**: Port to use for the webapp.
  - **Type**: `int`
  - **Default**: `8000`
- `--debug`
  - **Description**: Enable Flask debug mode.
  - **Default**: `False`
- `--online`
  - **Description**: Enable online mode.
  - **Default**: `False`

## Active Learning Dashboard - `monitor_al_loop`

Monitor a ATLAS active learning loop using a dashboard.

**Usage:** `monitor_al_loop [-h] [--process_id UUID/PK] [--update_interval n_sec] [--port port] [--debug] [--online]`

- `--process_id` (`UUID/PK`)
  - **Description**: Process id (pk/uuid) of the WorkChain to monitor.
  - **Type**: `str`
- `--update_interval` (`n_sec`)
  - **Description**: Refresh time interval in seconds.
  - **Type**: `int`
  - **Default**: `60`
- `--port` (`port`)
  - **Description**: Port to use for the webapp.
  - **Type**: `int`
  - **Default**: `8000`
- `--debug`
  - **Description**: Enable Flask debug mode.
  - **Action**: `store_const`
  - **Default**: `False`
- `--online`
  - **Description**: Enable online mode.
  - **Default**: `False`

## Generate Configuration File - `atl_gen_configuration_file`

Generates default configuration files for ATLAS in the TOML format. The generated `.toml` files can be used as a template for active learning runs and initial database generation.

**Usage:** `atl_gen_configuration_file [-h] -t TYPE [-p PATH] [-o]`

- `-t, --config_type` (`TYPE`)
  - **Description**: Type of the configuration file to be generated. Available types:
    - `active_learning`: Configuration file for an active learning loop.
    - `initial_db`: Configuration file for initial database generation.
  - **Type**: `str`
  - **Choices**: `["active_learning", "initial_db"]`
  - **Required**: Yes
- `-p, --path` (`PATH`)
  - **Description**: Path to store the configuration file. Defaults to the CWD.
  - **Type**: `Path`
  - **Default**: `.`
- `-o, --overwrite`
  - **Description**: Whether to overwrite the destination file if it already exists.
  - **Default**: `False`

## Generate Initial Database - `atl_gen_init_db`

Generates an initial database for atlas. A `.toml` configuration is required, which is described in the [corresponding inputs section](input_database_generation.md)

**Usage:** `atl_gen_init_db [-h] [-c PATH]`

- `-c, --config_file` (`PATH`)
  - **Description**: Path to a TOML settings file. By default, `database_generation_settings.toml` will be searched in the CWD.
  - **Type**: `Path`
  - **Default**: `./database_generation_settings.toml`

## Run DFT Database - `atl_run_dft_database`

Runs VASP DFT calculations for a database of structures using AiiDA-VASP. A .toml configuration file is required to specify the input settings for HPC and VASP. Refer to the [corresponding inputs section](input_dft.md) for more details about the available configuration options.

**Usage**: `atl_run_dft_database [-h] --db_file FILE --config FILE`

- `--db_file FILE, -i FILE`
  - **Description**: Path to the extxyz file containing the database of structures.
  - **Type**: `Path`
  - **Required**: Yes

- `--config FILE, -c FILE`
  - `Description`: Path to the TOML file with HPC and VASP input configuration.
  - `Type`: `Path`
  - `Required`: Yes

## Autoencoder for Dimensionality Reduction - `atl_train_autoencoder`

Train an autoencoder model for dimensionality reduction using the generated descriptors.
The model is trained on descriptors (either `SOAP` or `MACE`) which are provided through a numpy array in the `.npy` format that holds vstacked arrays of atomic descriptors.

**Usage:** `atl_train_autoencoder [-h] [--device DEVICE] [--dtype DTYPE] [--model_path MODEL_PATH] [--load_model LOAD_MODEL] [--rng_seed RNG_SEED] [--dataset DATASET] [--l1_hidden_dim L1_HIDDEN_DIM] [--l2_hidden_dim L2_HIDDEN_DIM] [--bottleneck_dim BOTTLENECK_DIM] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--patience PATIENCE] [--lr LR] [--weight_decay WEIGHT_DECAY] [--bias_flag] [--loss LOSS] [--train_frac TRAIN_FRAC] [--valid_frac VALID_FRAC] [--test_frac TEST_FRAC] [--wandb] [--wandb_name WANDB_NAME] [--wandb_project WANDB_PROJECT]`

### Arguments

- `--device`
  - **Description**: Device to run the model on. Options: `cpu`, `cuda`, or `auto`.
  - **Type**: `str`
  - **Default**: `cpu`
- `--dtype`
  - **Description**: Data type for the model.
  - **Type**: `str`
  - **Default**: `float32`
- `--model_path`
  - **Description**: Path to save the model.
  - **Type**: `str`
  - **Default**: `autoencoder_model.pth`
- `--load_model`
  - **Description**: Load the model from the model path.
  - **Type**: `bool`
  - **Default**: `False`
- `--rng_seed`
  - **Description**: Seed for the random number generator.
  - **Type**: `int`
- `--dataset`
  - **Description**: Path to the training dataset.
  - **Type**: `str`
  - **Default**: `all_descriptors.npz`
- `--l1_hidden_dim`
  - **Description**: Number of units in the first hidden layer.
  - **Type**: `int`
  - **Default**: `256`
- `--l2_hidden_dim`
  - **Description**: Number of units in the second hidden layer.
  - **Type**: `int`
  - **Default**: `32`
- `--bottleneck_dim`
  - **Description**: Dimensionality of the bottleneck (latent space).
  - **Type**: `int`
  - **Default**: `2`
- `--num_epochs`
  - **Description**: Number of epochs to train the model.
  - **Type**: `int`
  - **Default**: `50`
- `--batch_size`
  - **Description**: Batch size for training.
  - **Type**: `int`
  - **Default**: `4096`
- `--patience`
  - **Description**: Patience for early stopping.
  - **Type**: `int`
  - **Default**: `5`
- `--lr`
  - **Description**: Learning rate for the optimizer.
  - **Type**: `float`
  - **Default**: `0.001`
- `--weight_decay`
  - **Description**: L2 regularization.
  - **Type**: `float`
  - **Default**: `1e-5`
- `--bias_flag`
  - **Description**: Add this argument to unclude bias terms in the layers.
  - **Action**: `store_true`
- `--loss`
  - **Description**: Loss function type. Options: `mse`, `weighted_mse`.
  - **Type**: `str`
  - **Default**: `mse`
- `--train_frac`
  - **Description**: Fraction of data used for training.
  - **Type**: `float`
  - **Default**: `0.8`
- `--valid_frac`
  - **Description**: Fraction of data used for validation.
  - **Type**: `float`
  - **Default**: `0.1`
- `--test_frac`
  - **Description**: Fraction of data used for testing.
  - **Type**: `float`
  - **Default**: `0.1`
- `--wandb`
  - **Description**: Add this argument to log metrics to Weights & Biases.
  - **Action**: `store_true`
- `--wandb_name`
  - **Description**: Name of the Weights & Biases run.
  - **Type**: `str`
- `--wandb_project`
  - **Description**: Name of the Weights & Biases project.
  - **Type**: `str`

## MLIP Benchmark - `atl_benchmark_mlip`

Evaluates and compares the performance of Machine Learning Interatomic Potentials (MLIPs) using a suite of benchmarks. This tool can load models from `.model` files or from ATLAS workchains using the AiiDA pk/uuid and run various evaluations, such as molecular dynamics simulations, defect calculations, and surface energy analysis, returning figures and output with results.

**Usage:** `atl_benchmark_mlip [OPTIONS]`

### Main Arguments

- `--model_files <PATH>...`
  - **Description**: Paths to one or more `.model` files to be evaluated.
  - **Type**: `List[Path]`
- `--aiida_pks <PK>...`
  - **Description**: AiiDA workchain PKs/UUIDs to load models from.
  - **Type**: `List[int]`
- `--output_dir <PATH>`
  - **Description**: Directory to save all benchmark results and plots.
  - **Default**: `./mlip_evaluation`

### Slab Generation Arguments

These arguments configure the atomic structure (slab) used in many benchmarks.

- `--metal <SYMBOL>`
  - **Description**: Metal symbol for the benchmark systems (e.g., 'Cu', 'Al').
  - **Default**: `Cu`
- `--surface_indices <INT INT INT>`
  - **Description**: Miller indices for the surface of the slab.
  - **Default**: `1 1 1`
- `--supercell_size <INT INT INT>`
  - **Description**: Size of the supercell for the slab.
  - **Default**: `3 3 4`
- `--vacuum <FLOAT>`
  - **Description**: Vacuum layer thickness in Angstrom.
  - **Default**: `10.0`

### MD Parameters

These arguments control the molecular dynamics simulations.

- `--temp <FLOAT>`
  - **Description**: MD temperature in Kelvin.
  - **Default**: `300.0`
- `--n_steps <INT>`
  - **Description**: Number of MD steps to run.
  - **Default**: `10000`
- `--timestep <FLOAT>`
  - **Description**: MD timestep in femtoseconds.
  - **Default**: `2.0`
- `--friction <FLOAT>`
  - **Description**: Friction coefficient for the Langevin thermostat.
  - **Default**: `5e-3`
- `--device {cuda,cpu}`
  - **Description**: Device to run the calculations on.
  - **Default**: `cuda`
- `--dtype {float32,float64}`
  - **Description**: Data type for the calculations.
  - **Default**: `float64`

### Benchmark Selection

Use these flags to select which benchmarks to run. Multiple benchmarks can be run in a single command.

- `--run_energy_md`
  - **Description**: Run the energy MD benchmark.
- `--run_accuracy_test_set`
  - **Description**: Run energy and force error benchmark on a test set. Requires `--test_set_path`.
- `--test_set_path <PATH>`
  - **Description**: Path to the held-out test set for accuracy benchmarks.
- `--run_elastic_properties`
  - **Description**: Run elastic properties benchmark.
- `--run_defect_formation_energy`
  - **Description**: Run defect formation energy benchmark.
- `--run_surface_energies`
  - **Description**: Run surface energies benchmark.
- `--run_phonon_dispersion`
  - **Description**: Run phonon dispersion benchmark.
- `--run_high_temp_md`
  - **Description**: Run high-temperature MD benchmark.
- `--run_melting_point`
  - **Description**: Run melting point calculation benchmark.
- `--run_gsfe`
  - **Description**: Run Generalized Stacking Fault Energy (GSFE) benchmark.
- `--run_learning_curves`
  - **Description**: Plot learning curves from AL runs (requires `--aiida_pks`).
- `--run_final_db_size`
  - **Description**: Compare final database sizes from AL runs (requires `--aiida_pks`).
- `--run_evaluate_database`
  - **Description**: Evaluate models against a user-provided structure database. Requires `--database_path`.
- `--database_path <PATH>`
  - **Description**: Path to the structure database file for evaluation.

### UI Options

- `--no_rich_ui`
  - **Description**: Disable the Rich UI and use plain text output.

## DFT Benchmark - `atl_dft_benchmark`

Runs DFT calculations to benchmark and select optimal VASP settings for a given material system.

**Usage:** `atl_dft_benchmark [OPTIONS]`

- `--config <PATH>`
  - **Description**: Path to the TOML configuration file with benchmark settings.
- `--db_file <PATH>`
  - **Description**: Path to the structure database file.

## Database Analysis - `atl_database_analysis`

Analyzes database composition, diversity, and completeness metrics.

**Usage:** `atl_database_analysis [OPTIONS]`

- `--db_path <PATH>`
  - **Description**: Path to the database file (xz-compressed InitialDatabase or extxyz).
- `--metric <METRIC>`
  - **Description**: Diversity metric to compute. Options: `vendi_score`, `circles_metric`.
- `--descriptor_type <TYPE>`
  - **Description**: Descriptor type for diversity calculations. Options: `soap`, `mace`.
- `--output_dir <PATH>`
  - **Description**: Directory to save analysis results and plots.

## GUI - `atl_gui`

Launches the ATLAS Hub, a PySide6 desktop application for configuring and running ATLAS workflows.

**Usage:** `atl_gui [OPTIONS]`

- `--help`
  - **Description**: Show help message and exit.

The ATLAS Hub provides a graphical interface for:

- Project management with a dedicated workspace for each material system
- Database generation configuration and execution
- DFT calculation submission and monitoring
- Active learning loop setup, launch, and live tracking
- Report generation and visualization
- Settings management (AiiDA profiles, computers, codes, Materials Project API key)

To use the GUI, install the optional `gui` dependency:

```bash
pip install -e '.[gui]'
```
