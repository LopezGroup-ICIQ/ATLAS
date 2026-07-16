
```{role} alt
:class: code-alt
```
```{role} codeheader
:class: code-header
```

## Latent Space Analysis

Generate a latent space analysis template file using `atl_gen_configuration_file -t latent_space_analysis`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### Interpolation Check Settings - `[interpolation]`

Settings for the interpolation check. The interpolation check determines whether a structure is within the model's knowledge domain by checking the committee models disagreement, considering the model accuracy threshold.


- {alt}`disagreement_check_type`:
  - **Description**: Approach for energy and force (E&F) committee disagreement check. With `training`, compare E&F with a threshold obtained from the training RMSE values multiplied by a threshold. With `md_threshold`, compare E&F with a threshold obtained from the standard deviaiton of the MD frames.
  - **Type**: `(str)`
  - **Default**: `'training'`.
  - Possible values are: `training`, `md_threshold`.

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
  - **Default**: `50.0`.

- {alt}`default_alpha_if_issues`:
  - **Description**: Default alpha value if there are issues with the concave hull generation.
  - **Type**: `(optional, float)`
  - **Default**: `5.0`.

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
  - **Description**: AiiDA metadata and scheduler options for descriptor computation.
  - **Type**: `(optional, dict)`

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
