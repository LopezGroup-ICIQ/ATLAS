
```{role} alt
:class: code-alt
```
```{role} codeheader
:class: code-header
```

## Database Generation

Generate a database generation template file using `atl_gen_configuration_file -t database_generation`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### Database - `[database]`

General settings and file paths for the database.


- {alt}`database_name`:
  - **Description**: Name of the database to be used for internal reference and as the filename.
  - **Type**: `(str)`
  - **Example**: `'my_material_db'`.

- {alt}`min_num_atoms`:
  - **Description**: Minimum number of atoms allowed in the generated structures.
  - **Type**: `(int)`
  - **Default**: `64`.

- {alt}`max_num_atoms`:
  - **Description**: Maximum number of atoms allowed in the generated structures.
  - **Type**: `(int)`
  - **Default**: `128`.

- {alt}`min_cell_size`:
  - **Description**: Minimum cell size in Angstrom.
  - **Type**: `(float)`
  - **Default**: `5.0`.

- {alt}`relax_struct_path`:
  - **Description**: Path to a folder containing DFT optimized structures.
  - **Type**: `(optional, str)`
  - **Default**: `''`.

- {alt}`database_path`:
  - **Description**: Path where the final database will be saved.
  - **Type**: `(str, PosixPath)`
  - **Default**: `''`.

- {alt}`rng_seed`:
  - **Description**: Numerical value used to fix the RNG seed. If not specified, it will be chosen randomly each run.
  - **Type**: `(optional, int)`
  - **Example**: `42`.

- {alt}`overwrite_db`:
  - **Description**: Allow database overwrite. If false, and the database exists, the new database name will include a timestamp.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

#### Composition - `[database.composition]`

Settings for the composition of the database.


- {alt}`size`:
  - **Description**: Maximum number of structures to generate for the database.
  - **Type**: `(int)`
  - **Default**: `7500`.

##### Ratios - `[database.composition.ratios]`

Fraction of different structure types. The sum of the fractions must be equal to 1.0.


- {alt}`bulk`:
  - **Description**: Fraction of structures that will be bulk.
  - **Type**: `(float)`
  - **Default**: `0.4`.

- {alt}`surface`:
  - **Description**: Fraction of structures that will be surfaces.
  - **Type**: `(float)`
  - **Default**: `0.6`.

- {alt}`cluster`:
  - **Description**: Fraction of structures that will be clusters.
  - **Type**: `(optional, float)`
  - **Default**: `0.0`.

#### Plot_Db - `[database.plot_db]`

Display and Export Options for the phase diagram plot.


- {alt}`show`:
  - **Description**: Whether to display the database with a phase diagram after creation.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`format`:
  - **Description**: Format for the figure.
  - **Type**: `(optional, str)`
  - **Default**: `'png'`.
  - Possible values are: `png`, `svg`.

##### Rc_Params - `[database.plot_db.rc_params]`

Matplotlib rcParams for the plot.


- {alt}`font.family`:
  - **Description**: Font family for the phase diagram plot.
  - **Type**: `(str)`
  - **Default**: `'monospace'`.

- {alt}`font.size`:
  - **Description**: Font size for the phase diagram plot.
  - **Type**: `(int)`
  - **Default**: `14`.

#### Show_Db_Ase - `[database.show_db_ase]`

ASE GUI display options.

:::{attention}
This section is optional.
:::


- {alt}`show`:
  - **Description**: Whether to display the database using ASE GUI after creation.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

#### Export - `[database.export]`

Export options for the database.


- {alt}`export`:
  - **Description**: Whether to export the database.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`format`:
  - **Description**: Export format supported by ASE (e.g., 'extxyz').
  - **Type**: `(str)`
  - **Default**: `'extxyz'`.

- {alt}`file_path`:
  - **Description**: Path where the exported file will be saved.
  - **Type**: `(str, PosixPath)`
  - **Default**: `''`.

- {alt}`file_name`:
  - **Description**: Name of the exported file.
  - **Type**: `(str)`
  - **Default**: `'export_db_filename'`.

### Phase_Diagram - `[phase_diagram]`

Description of the phase diagram.


- {alt}`material_name`:
  - **Description**: Internal name for the material in the phase diagram.
  - **Type**: `(str)`
  - **Default**: `'default_material_name'`.

- {alt}`element_list`:
  - **Description**: List of elements to include in the phase diagram.
  - **Type**: `(list[str])`
  - **Example**:

```python
['Cu', 'O']
```

- {alt}`base_element`:
  - **Description**: Symbol of the most abundant element in the phase.
  - **Type**: `(str)`
  - **Example**: `'Cu'`.

#### Phase Definition - `[phase_diagram.phase]`

Defines a specific phase within the phase diagram. Multiple phases can be added.


##### Phase Definition - `[phase_diagram.phase.XXXXX]`

This key describes settings for dynamic entries. Several entries can be added by using different key names.

The key name (`XXXXX`) is used as the reference name. **Replace XXXXX with a name of your choice.**

Accepted parameters for each entry:


- {alt}`name`:
  - **Description**: Name to be used as reference for the phase.
  - **Type**: `(str)`
  - **Example**: `'alpha'`.

- {alt}`cluster_element`:
  - **Description**: Symbol of the element defining the cluster.
  - **Type**: `(optional, str)`

- {alt}`prototype`:
  - **Description**: Materials Project ID of the prototypical structure.
  - **Type**: `(str)`
  - **Example**: `'mp-30'`.

- {alt}`offset`:
  - **Description**: Fraction of composition allowed over and under the phase limits.
  - **Type**: `(float)`
  - **Default**: `0.1`.

- {alt}`limit_max_num_structures`:
  - **Description**: Maximum number of structures to generate for this phase.
  - **Type**: `(optional, int)`
  - **Default**: `100`.

- {alt}`allow_modifications`:
  - **Description**: Allow modifications (supercells, replacements, etc.) to the base structure.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`use_cache`:
  - **Description**: Store structures in cache to speed up generation. Can consume a lot of disk space.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

Parameters using `composition.` prefix:


Parameters using `replacements.` prefix:

- {alt}`replacements.replace`:
  - **Description**: Whether to replace specific elements. Elements in element_list will be considered for replacement and replaced by a single element species.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.
- {alt}`replacements.element_list`:
  - **Description**: List of elements to be replaced.
  - **Type**: `(optional, list[str])`
  - **Example**:

```python
['Ti']
```
- {alt}`replacements.replace_with`:
  - **Description**: Element to replace with.
  - **Type**: `(optional, str)`
  - **Example**: `'Ir'`.

### Generation - `[generation]`

Structure generation settings.


- {alt}`generate_type`:
  - **Description**: Types of structures to generate.
  - **Type**: `(list[str])`
  - **Default**: `['bulk', 'surface', 'cluster']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

#### Bulk - `[generation.bulk]`

Bulk structure generation settings.


- {alt}`num_struct`:
  - **Description**: Number of structures to generate.
  - **Type**: `(int)`
  - **Default**: `25`.

- {alt}`num_repeat`:
  - **Description**: Number of repeats for each structure.
  - **Type**: `(int)`
  - **Default**: `5`.

- {alt}`supercell_max_idx`:
  - **Description**: Maximum Miller index for the bulk supercells.
  - **Type**: `(int)`
  - **Default**: `2`.

#### Surface - `[generation.surface]`

Surface structure generation settings.


- {alt}`min_miller_index`:
  - **Description**: Minimum Miller index used to generate surface structures.
  - **Type**: `(int)`
  - **Default**: `1`.

- {alt}`max_miller_index`:
  - **Description**: Maximum Miller index used to generate surface structures.
  - **Type**: `(int)`
  - **Default**: `3`.

- {alt}`min_slab_size_ang`:
  - **Description**: Minimum slab thickness in Angstrom.
  - **Type**: `(optional, float)`
  - **Default**: `7.0`.

- {alt}`min_vacuum_size_ang`:
  - **Description**: Minimum size of the vacuum layer in Angstroms.
  - **Type**: `(float)`
  - **Default**: `12.0`.

- {alt}`get_supercells`:
  - **Description**: Whether to generate supercells for surface structures.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`fixed_layers`:
  - **Description**: Number of fixed layers in the surface slab.
  - **Type**: `(int)`
  - **Default**: `3`.

- {alt}`max_number_supercells`:
  - **Description**: Maximum number of surface supercells to generate.
  - **Type**: `(int)`
  - **Default**: `200`.

- {alt}`save_in_db`:
  - **Description**: Whether to save generated surfaces in the database.
  - **Type**: `(bool)`
  - **Default**: `True`.

- {alt}`num_replacements`:
  - **Description**: Number of replacement percentages to generate for each structure.
  - **Type**: `(int)`
  - **Default**: `20`.

- {alt}`num_repeat_replace`:
  - **Description**: Number of repeats for each replacement.
  - **Type**: `(int)`
  - **Default**: `2`.

- {alt}`frac_slabs_save`:
  - **Description**: Fraction of slabs to save after generation.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`frac_supercells_save`:
  - **Description**: Fraction of unreplaced supercells to save after generation.
  - **Type**: `(optional, float)`
  - **Default**: `0.1`.

- {alt}`max_slab_num`:
  - **Description**: Maximum number of slabs to gather from the slab generation.
  - **Type**: `(int)`
  - **Default**: `15`.

- {alt}`n_workers`:
  - **Description**: Maximum number of workers for parallel processing.
  - **Type**: `(optional, int)`

#### Cluster - `[generation.cluster]`

Cluster structure generation settings.

:::{attention}
This section is optional.
:::


- {alt}`size_range`:
  - **Description**: List of cluster sizes (number of atoms) to generate.
  - **Type**: `(list[int])`
  - **Default**: `[10, 20, 30, 40, 50, 60]`.

- {alt}`add_dimer`:
  - **Description**: Whether to add a 2-atom dimer structure.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`save_in_db`:
  - **Description**: Whether to save generated clusters in the database.
  - **Type**: `(optional, bool)`
  - **Default**: `True`.

- {alt}`get_replacements`:
  - **Description**: Whether to generate replacement clusters (substitute atoms).
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`get_perturbed`:
  - **Description**: Whether to generate perturbed clusters (Gaussian displacement).
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`num_struct`:
  - **Description**: Number of replacement/perturbation structures to generate per base cluster.
  - **Type**: `(optional, int)`
  - **Default**: `2`.

- {alt}`num_repeat`:
  - **Description**: Number of repeats for each replacement/perturbation.
  - **Type**: `(optional, int)`
  - **Default**: `2`.

- {alt}`limit_max_num_structures`:
  - **Description**: Maximum number of cluster structures to generate in total.
  - **Type**: `(optional, int)`
  - **Default**: `100`.

- {alt}`perturbation_ang`:
  - **Description**: Gaussian perturbation magnitude in Angstroms.
  - **Type**: `(optional, float)`
  - **Default**: `0.04`.

- {alt}`cluster_method`:
  - **Description**: Method for initial cluster geometry. 'wulff' uses Wulff construction (FCC/BCC, falls back to spherical for HCP). 'spherical' uses random spherical packing.
  - **Type**: `(optional, str)`
  - **Default**: `'wulff'`.

- {alt}`basin_hopping`:
  - **Description**: Whether to relax clusters via basin hopping with a Lennard-Jones potential.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`bh_totalsteps`:
  - **Description**: Number of basin hopping steps for cluster relaxation.
  - **Type**: `(optional, int)`
  - **Default**: `10`.

- {alt}`bh_fmax`:
  - **Description**: Force convergence criterion (eV/Angstrom) for basin hopping.
  - **Type**: `(optional, float)`
  - **Default**: `0.05`.

- {alt}`lj_sigma`:
  - **Description**: Lennard-Jones sigma parameter (Angstrom) for basin hopping. When omitted, sigma is auto-derived from the element's bulk nearest-neighbor distance.
  - **Type**: `(optional, float)`
  - **Default**: `'None'`.

- {alt}`lj_epsilon`:
  - **Description**: Lennard-Jones epsilon parameter (eV) for basin hopping.
  - **Type**: `(optional, float)`
  - **Default**: `1.0`.

### Deformation - `[deformation]`

Lattice deformation settings.

:::{attention}
This section is optional.
:::


- {alt}`lattice_frac_deform_max`:
  - **Description**: Maximum deformation value as a percentage of the lattice side length.
  - **Type**: `(float)`
  - **Default**: `0.05`.

- {alt}`lattice_frac_deform_min`:
  - **Description**: Minimum deformation value as a percentage of the lattice side length.
  - **Type**: `(float)`
  - **Default**: `0.01`.

- {alt}`num_repeats`:
  - **Description**: Number of repeats for each structure with random deformations.
  - **Type**: `(int)`
  - **Default**: `5`.

- {alt}`limit_max_num_deformations`:
  - **Description**: Maximum number of lattice deformations to generate.
  - **Type**: `(int)`
  - **Default**: `100`.

### Perturbation - `[perturbation]`

Perturbation settings.

:::{attention}
This section is optional.
:::


- {alt}`filter_struct_types`:
  - **Description**: Types of structures to which the perturbation will be applied.
  - **Type**: `(list[str])`
  - **Default**: `['bulk', 'surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- {alt}`limit_max_num_perturbs`:
  - **Description**: Maximum number of perturbations to generate.
  - **Type**: `(int)`
  - **Default**: `100`.

- {alt}`num_repeats`:
  - **Description**: Number of repeats for each structure with random perturbations.
  - **Type**: `(int)`
  - **Default**: `1`.

- {alt}`perturbation_ang`:
  - **Description**: Perturbation magnitude in Angstrom.
  - **Type**: `(optional, float)`
  - **Default**: `0.04`.

### Adsorbates - `[adsorbates]`

Adsorbate placement settings.

:::{attention}
This section is optional.
:::


- {alt}`filter_struct_types`:
  - **Description**: Types of structures to which adsorbates will be added.
  - **Type**: `(optional, list[str])`
  - **Default**: `['surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- {alt}`limit_max_num_perturbs`:
  - **Description**: Maximum number of structures with adsorbates to generate.
  - **Type**: `(optional, int)`
  - **Default**: `100`.

- {alt}`num_repeats`:
  - **Description**: Number of repeats for each structure.
  - **Type**: `(int)`
  - **Default**: `1`.

- {alt}`adsorbate_species`:
  - **Description**: List of adsorbate species to consider.
  - **Type**: `(list[str])`
  - **Example**:

```python
['H', 'H2O']
```

### Struct_Filters - `[struct_filters]`

Settings for filtering out incorrect structures.

:::{attention}
This section is optional.
:::


#### No_Neighbors - `[struct_filters.no_neighbors]`

Filter for structures with atoms that have no neighbors.


- {alt}`cov_rad_multiplier`:
  - **Description**: Multiplier applied to the covalent radii to be used as cutoff radius for the neighbor check.
  - **Type**: `(optional, float)`
  - **Default**: `1.2`.

#### Layer_Distance - `[struct_filters.layer_distance]`

Filter for layer distances in surface slabs.

:::{attention}
This section is optional.
:::


- {alt}`max_layer_distance_ang`:
  - **Description**: Maximum accepted distance between layers in Angstrom.
  - **Type**: `(optional, float)`
  - **Default**: `4.0`.

#### Duplicate_Slabs - `[struct_filters.duplicate_slabs]`

Filter for duplicate slabs.

:::{attention}
This section is optional.
:::


- {alt}`tolerance`:
  - **Description**: Tolerance for the duplicate slabs filter.
  - **Type**: `(optional, float)`
  - **Default**: `0.2`.

### Vacancies - `[vacancies]`

Settings for vacancy generation.


- {alt}`filter_struct_types`:
  - **Description**: Types of structures to which vacancies will be applied.
  - **Type**: `(list[str])`
  - **Default**: `['bulk', 'surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- {alt}`limit_max_num_vacancies`:
  - **Description**: Maximum number of structures with vacancies to generate.
  - **Type**: `(optional, int)`
  - **Default**: `400`.

- {alt}`num_repeats`:
  - **Description**: Number of repeats for each structure with different random vacancies.
  - **Type**: `(int)`
  - **Default**: `3`.

- {alt}`max_vacancy_percentage`:
  - **Description**: Maximum vacancies to generate as a percentage of the total number of atoms.
  - **Type**: `(float)`
  - **Default**: `0.75`.

- {alt}`min_vacancy_percentage`:
  - **Description**: Minimum vacancies to generate as a percentage of the total number of atoms.
  - **Type**: `(float)`
  - **Default**: `0.025`.

- {alt}`element_list`:
  - **Description**: List of elements to consider for the vacancies.
  - **Type**: `(list[str])`
  - **Example**:

```python
['O']
```

### Targeted_Modification - `[targeted_modification]`

Settings for targeted structural modifications.

:::{attention}
This section is optional.
:::


#### Central_Atom_Octahedral - `[targeted_modification.central_atom_octahedral]`

Apply perturbations to the central atom in octahedral sites.

:::{attention}
This section is optional.
:::


- {alt}`filter_phases`:
  - **Description**: Only apply the modification to the following phases.
  - **Type**: `(optional, list[str])`
  - **Example**:

```python
['rutile', 'original_IrO2']
```

- {alt}`filter_struct_types`:
  - **Description**: Types of structures to which the modification will be applied.
  - **Type**: `(optional, list[str])`
  - **Default**: `['bulk', 'surface']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- {alt}`central_element`:
  - **Description**: Symbol of the central element of the octahedral site.
  - **Type**: `(optional, str)`
  - **Example**: `'Ir'`.

- {alt}`num_repeats`:
  - **Description**: Number of repeats for each structure with different perturbations.
  - **Type**: `(optional, int)`
  - **Default**: `3`.

- {alt}`limit_max_num_modifications`:
  - **Description**: Maximum number of modified structures to generate.
  - **Type**: `(optional, int)`
  - **Default**: `200`.

- {alt}`max_perturbation_ang`:
  - **Description**: Maximum perturbation movement of the central atom in Angstrom.
  - **Type**: `(optional, float)`
  - **Default**: `0.2`.

### Concave_Hull - `[concave_hull]`

Settings for descriptors and boundary determination.

:::{attention}
This section is optional.
:::


- {alt}`gen_concave_hull`:
  - **Description**: Whether to generate the boundary of the descriptors for all structures in the database.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`boundary_method`:
  - **Description**: Algorithm for boundary determination.
  - **Type**: `(optional, str)`
  - **Default**: `'concave_hull'`.
  - Possible values are: `concave_hull`, `morphological_closing`.

- {alt}`descriptor`:
  - **Description**: Descriptor to use for the boundary generation.
  - **Type**: `(optional, str)`
  - **Default**: `'SOAP'`.
  - Possible values are: `SOAP`, `MACE`.

- {alt}`dim_reduction`:
  - **Description**: Dimensionality reduction method for the boundary generation.
  - **Type**: `(optional, str)`
  - **Default**: `'autoencoder'`.
  - Possible values are: `PCA`, `autoencoder`.

- {alt}`plot_filename`:
  - **Description**: Filename for the figure displaying the boundary.
  - **Type**: `(optional, str)`
  - **Default**: `'descriptors_concave_hull.png'`.

- {alt}`morph_disk_size`:
  - **Description**: Disk size for morphological closing structuring element. Larger values bridge wider gaps between points.
  - **Type**: `(optional, int)`
  - **Default**: `10`.

- {alt}`morph_threshold`:
  - **Description**: Grayscale intensity threshold (0-255) for point detection in morphological closing.
  - **Type**: `(optional, int)`
  - **Default**: `250`.

- {alt}`morph_dpi`:
  - **Description**: DPI of the internal rasterisation used by morphological closing. Higher values give finer boundaries.
  - **Type**: `(optional, int)`
  - **Default**: `100`.
