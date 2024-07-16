# MatDBForge

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="media/logo_light.png">
    <img alt="MatDBForge: A workflow for materials NNP generation" src="media/logo_light.png">
  </picture>

MatDBForge is a Python library that enables the generation of a chemical compounds database for training NNPs (Neural Network Potential). It provides functionalities to create and manage a database of materials structures for training machine learning models, and allows to interact with workflow tools in order to automate the structure labelling.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Structure and Examples](#structure-and-examples)
- [License](#license)

## Installation

To install MatDBForge, you can use pip in a python virtual environment or conda environment.

First, create a virtual environment and activate it:

👷 WIP 🕒

```
python3.10 -m venv mdb
source mdb/bin/activate
```

Next, clone the repository:

```
git clone git@github.com:pol-sb/MatDBForge.git
```

Finally, enter the folder containing the repository and install the library using pip in the desired python environment:

```shell
cd MatDBForge
python3 -m pip install MatDBForge **replace by . ?**
```

The active learning (AL) loop uses the [aiida](https://github.com/aiidateam/aiida-core) library for managing the workflow. In order to run the

## Usage

The goal of this library is to provide workflows, functions and utilities for streamlining the training of neural networks potentials (NNPs). 

During the library installation, several entry points will be added so that the user can easily run the different utilities:
- `mdb_conf_gen`: Generate a `.toml` template configuration file to be used in any of the different operation modes of the code.
- `mdb_gen_init_db`: 👷 WIP 🕒
- `mdb_active_learning`: Launch an AL loop using a configuration file and a labelled initial database.
- `mdb_monitor_al_loop:` Launch a flask dashboard locally to monitor a running active learning loop. Open http://127.0.0.1:8000 (or port specified in the launch arguments) in a browser to visualize the dashboard.

All of the entry points provide usage documentation when launched with the `-h`/`--help` argument, e.g.:


```bash
$> mdb_conf_gen --help

>>> usage: gen_default_config [-h] -t TYPE [-p PATH] [-o]
>>> 
>>> Generate MDB default configuration files in the TOML format.
>>> 
>>> options:
>>>   -h, --help
                     show this help message and exit
>>>   -t TYPE, --config_type TYPE
>>>                  Type of the configuration file to be generated. Available types are:
>>>                         - active_learning: Configuration file for active learning loop.
>>>                         - initial_db: Configuration file for initial database generation.
>>>   -p PATH, --path PATH
                        Path in which to store the file.
                        Will use the CWD by default. Folders will be created if necessary.
>>>  -o, --overwrite
                        Whether to overwrite the destination file, if existent.
```

The utilities for generation and running the AL loop use inputs in the TOML format. Users are advised to use `mdb_conf_gen` to generate a template file which can be customized. 

A description of all the possible parameters is available in the documentation for the input files: [Input](./docs/input.md). 


### Usage: Training a MACE NNP from scratch

👷 WIP 🕒


## Structure and Examples

The main functionalities are organized into the following modules:

- `workflows`: Contains functions and methods that allow to connect the database with workflow tools, mainly with the goal of performing DFT calculations.
- `core`: Includes core functionalities and utilities used by the library, such as the generation and management of the database.
- `active_learning`: Contains classes and functions leveraged during the active learning loops.
- `examples`: Provides example scripts that demonstrate the usage of the library.


The following examples demonstrate the usage of MatDBForge:

- [launch_sp_calcs_db_aiida.py](src/MatDBForge/examples/aunch_sp_calcs_db_aiida.py): This example script demonstrates how to launch single-point calculations for a given set of structures and store the results in the database.
- [create_init_db_new.py](src/MatDBForge/examples/create_init_db_new.py): This example script showcases how to create and initialize a new database with initial data.

Please refer to the examples in the examples directory for more details on how to utilize the library for your specific needs.

## License
👷 WIP 🕒
