# Installation

## Installation procedure

To install ATLAS, you can use pip in a python virtual environment or conda environment. Development has been made with `python3.11` in mind, which can be installed through the OS's package manager or conda.

### 1. Creating a pyton environment

First, **create a virtual environment** and activate it. This can be done in several ways, but we provide some examples using `conda`, python `venv` or `uv`.

#### Option A - `conda`

```bash
# Create a conda environment named atlas which uses python 3.11
conda create -n atlas python=3.11

# Activate the environment
conda activate atlas
```

#### Option B - `venv`

An example for an Ubuntu 22.04 system, using python3.11 and venv:

```bash
# Install python3.11 and venv
sudo apt install python3.11 python3.11-venv

# Using python venv - create and activate the environment
python3 -m venv atlas
source atlas/bin/activate
```

#### Option C - `uv`

First, install the `uv` tool. Either as shown below using the standalone installer, or please refer to the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)) for more options.

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Once `uv` is isntalled, create an environment named atlas specifically with Python 3.11:

```bash
# Create the virtual environment
uv venv atlas --python 3.11
```

Make sure to navigate to a folder where you would like your python environment to be located, or specify the desired path.
You can activate the newly created environment as follows:

```bash
source atlas/bin/activate
```

With the environment now activated, the library can be installed.

### 2. Getting the ATLAS code

```bash
# Clone the reposittory
git clone https://github.com/pol-sb/atlas.git
```

### 3. Installing the library in the activated python environment

There are several installation mechanisms, and several optional dependencies depending on what packages you want to use. Check the list and details of optional dependencies in the [pyproject.toml](../../pyproject.toml). Currently, the following are available:

- `mace`
- `dev`
- `gui`

Optional dependencies are installed using the following syntax:

```bash
python3 -m pip install ./ATLAS['OPTIONAL_DEPENDENCY_NAME']
```

Some installation examples follow:

#### Using `pip`

```bash
# Install the library and the MACE dependencies in the venv using pip
python3 -m pip install ./ATLAS['mace']
```

#### Using `uv`

```bash
# Install the library and the MACE and dev dependencies using uv
uv pip install ./ATLAS['mace', 'dev']

#### Installing the GUI

The ATLAS Hub desktop GUI requires the optional `gui` dependency:

```bash
pip install -e '.[gui]'
```
```

### 4. Initialize configuration files

Finally, initialize configuration files by running the **initial configuration command** (`atl_init_setup`). Then, enter your [Materials Project API key](https://next-gen.materialsproject.org/api) in the path displayed in the output to finish the setup process:

```shell
# Run the last setup step - configuration initialization
atl_init_setup
```

> [!NOTE]
> If the user is only interested in database generation, the setup can be completed only up until this point, skipping the following AiiDA setup.

### 5. Setup steps specific to active learning

- The **active learning (AL)** loop uses the [AiiDA](https://github.com/aiidateam/aiida-core) library for managing the workflow. In order to run the AL loop in compute clusters, codes and computers must be conifigured in AiiDA. See the [AiiDA installation guide](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_quick.html) for installation instructions.
- DFT calculations with VASP use the [aiida-vasp](https://aiida-vasp.readthedocs.io/en/latest/) plugin, which needs additional configuration. Please, follow the [instructions on their website](https://aiida-vasp.readthedocs.io/en/latest/getting_started/general.html).

- The steps required to set up the active learning loop with the simplest AiiDA configuration are the following:
  1. Set up an aiida profile and database with `verdi presto`.
  2. Create the AiiDA computer and code entries for ATLAS and aiida-vasp.
  3. Add the potential datasets for aiida-vasp ([information here](https://aiida-vasp.readthedocs.io/en/latest/getting_started/potentials.html)).
