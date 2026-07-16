"""Autoencoder model for dimensionality reduction and data reconstruction."""

import logging
import pathlib as pl
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from atlas.core.code_utils import custom_print

# Silencing specific warnings and log messages
warnings.filterwarnings('ignore', category=UserWarning, message='.*weights_only.*')

# Force third party loggers to only show errors and critical messages
logging.getLogger('mace').setLevel(logging.ERROR)
logging.getLogger('e3nn').setLevel(logging.ERROR)


def locate_standarization_files(autoencoder_path=pl.Path('.')):
    """Check for the existence of standardization files.

    These files are stored as numpy arrays containing values necessary
    to apply same standarization scale.

    Parameters
    ----------
    autoencoder_path: pl.Path
        Path for the autoencoder model

    Returns
    -------
    mean_vals: np.ndarray or False
        Array of mean values for each feature, or False if not found.
    """
    if isinstance(autoencoder_path, str):
        autoencoder_path = pl.Path(autoencoder_path)

    mean_vals, std_vals = None, None

    for file in autoencoder_path.parent.glob('*vals.npy'):
        if 'mean' in file.name:
            mean_vals = np.load(file)
        if 'std' in file.name:
            std_vals = np.load(file)

    return mean_vals, std_vals


class Autoencoder(nn.Module):
    """
    Autoencoder model for dimensionality reduction and data reconstruction.

    This class implements a simple feedforward autoencoder model using PyTorch's
    `nn.Module`. It consists of two main components:

    - Encoder: Compresses the input data to a lower-dimensional
               bottleneck (latent space).
    - Decoder: Reconstructs the input data from the bottleneck representation.

    Attributes
    ----------
    encoder : nn.Sequential
        A neural network stack that reduces the input dimensionality down to the
        specified bottleneck dimension, capturing important features in a
        compressed form.

    decoder : nn.Sequential
        A neural network stack that reconstructs the input from the bottleneck
        representation, attempting to match the original input as closely as possible.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    l1_dim : int
        Dimension of the first hidden layer in the encoder (last hidden layer
        in the decoder).
    l2_dim : int
        Dimension of the second hidden layer in the encoder (second-to-last
        hidden layer in the decoder).
    bottleneck_dim : int, optional, default=2
        Dimension of the bottleneck layer (latent space) where input is compressed.
    bias_flag : bool, optional, default=True
        Whether to include a bias term in each linear layer. Setting to False makes the
        model less dependent on training data and more generalizable to unseen data.



    Methods
    -------
    forward(x):
        Performs a forward pass through the encoder and decoder, compressing
        the input to the bottleneck and reconstructing it to its original dimension.

    Example Usage:
    --------------
    >>> model = Autoencoder(
    ...     input_dim=100, l1_dim=64, l2_dim=32, bottleneck_dim=10
    ... )
    >>> input_data = torch.randn(1, 100)
    >>> reconstructed_data = model(input_data)
    """

    def __init__(
        self,
        input_dim: int,
        l1_dim: int,
        l2_dim: int,
        bottleneck_dim: int = 2,
        bias_flag=True,
    ):
        super().__init__()

        # Encoder: Input -> Bottleneck (latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, l1_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l1_dim, l2_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l2_dim, bottleneck_dim, bias=bias_flag),
        )

        # Decoder: Bottleneck -> Input (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, l2_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l2_dim, l1_dim, bias=bias_flag),
            nn.ReLU(),
            nn.Linear(l1_dim, input_dim, bias=bias_flag),
        )

    def forward(self, x):
        # Pass input through the encoder to obtain the latent space (bottleneck)
        # and reconstruct the input from the latent space
        return self.decoder(self.encoder(x))


def load_autoencoder_model(model_path: str, data_arr: np.ndarray = None):
    from torch.serialization import safe_globals

    # Only my classes and pytorch classes are allowed.
    # Only bugs in my classes should be dangerous.
    trusted_classes = [
        Autoencoder,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.activation.ReLU,
    ]

    # To allow loading of custom model classes without changing
    # to weights_only = False
    # Context manager is used to avoid affecting global state
    # and preventing other security risks.
    with safe_globals(trusted_classes):
        try:
            model = torch.load(model_path)
        except Exception as e:
            custom_print(
                f'Error loading model from {model_path}. '
                f'Please check the file path and format. Error: {e}',
                'error',
            )
            raise

    state_dict = model if isinstance(model, dict) else False

    if state_dict:
        input_dim = state_dict['encoder.0.weight'].shape[1]
        l1_dim = state_dict['encoder.0.weight'].shape[0]
        l2_dim = state_dict['encoder.2.weight'].shape[0]

        # Check if the model already exists
        if pl.Path(model_path).exists() and isinstance(model, dict):
            # Extract bottleneck_dim from the final encoder layer (layer 4)
            bottleneck_dim = state_dict['encoder.4.weight'].shape[0]

            # Check if bias keys exist in the state dict
            has_bias = 'encoder.0.bias' in state_dict

            model = Autoencoder(
                input_dim=input_dim,
                l1_dim=l1_dim,
                l2_dim=l2_dim,
                bottleneck_dim=bottleneck_dim,
                bias_flag=has_bias,
            )
            model.load_state_dict(state_dict)

    custom_print('Model loaded successfully!', 'done')

    return model


def get_latent_space_autoencoder(
    model,
    descriptor_dict: dict,
    device: str = None,
    dtype=torch.float32,
    quiet: bool = False,
    standardize_data: bool = False,
    autoencoder_path: str | pl.Path = None,
):
    if quiet:
        init_print_type = 'debug'
        final_print_type = 'debug'
    else:
        init_print_type = 'info'
        final_print_type = 'done'

    # Changing device to available GPU if available, else CPU.
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # Remember that I must call model.eval() to set dropout and batch
    # normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Standardize the data if required
    should_standardize: bool = standardize_data and (autoencoder_path is not None)

    # Loading standardization parameters from arrays in path
    if should_standardize:
        autoencoder_path = pl.Path(autoencoder_path)
        mean_vals, std_vals = locate_standarization_files(autoencoder_path)
        custom_print('Loaded standardization parameters.', init_print_type)

    # Reduce the dimensionality of the input points to 2D
    custom_print('Computing latent space for all structures...', init_print_type)

    # No need to compute gradients for inference
    with torch.no_grad():
        uuids = list(descriptor_dict.keys())

        # Gather all descriptors, handle both per-atom (2D) and averaged (1D)
        all_descrs = []
        all_lengths = []
        for u in uuids:
            d = descriptor_dict[u]['descriptors']
            # Descriptor generators store descriptors as a list of arrays;
            # unwrap to a single array for the autoencoder.
            if isinstance(d, list):
                d = d[0] if len(d) == 1 else np.vstack(d)
            if d.ndim == 1:
                d = d.reshape(1, -1)
            all_descrs.append(d)
            all_lengths.append(d.shape[0])
        descr_tensor = torch.tensor(np.vstack(all_descrs), device=device, dtype=dtype)

        # Standardize batched data
        if should_standardize and mean_vals is not None and std_vals is not None:
            mean_tensor = torch.tensor(mean_vals, device=device, dtype=dtype)
            std_tensor = torch.tensor(std_vals, device=device, dtype=dtype)
            descr_tensor = (descr_tensor - mean_tensor) / std_tensor

        # Run the ENTIRE batch through the model at once
        latent_batch = model.encoder(descr_tensor).cpu().numpy()

        # Map the results back to the dictionary
        ini_posc_desc_list = 0
        fin_posc_desc_list = 0
        for i, uuid in enumerate(uuids):
            curr_struct_length = all_lengths[i]
            fin_posc_desc_list += curr_struct_length

            descriptor_dict[uuid]['latent_space'] = latent_batch[
                ini_posc_desc_list:fin_posc_desc_list
            ]
            ini_posc_desc_list = fin_posc_desc_list

    custom_print('Computed latent space!', final_print_type)
    return descriptor_dict


def evaluate_reconstruction(
    autoencoder_model,
    descriptor_dict: dict,
    device: str | None = None,
    dtype=torch.float32,
    standardize_data: bool = False,
    autoencoder_path: str | pl.Path | None = None,
):
    """Evaluates the autoencoder by computing reconstruction MAE and RMSE."""
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    autoencoder_model.to(device)
    autoencoder_model.eval()

    should_standardize = standardize_data and (autoencoder_path is not None)

    with torch.no_grad():
        uuids = list(descriptor_dict.keys())
        all_descrs = [descriptor_dict[u]['descriptors'][0] for u in uuids]

        # Original ground truth matrix
        original_np = np.vstack(all_descrs)
        descr_tensor = torch.tensor(original_np, device=device, dtype=dtype)

        # Apply standardization to input if required
        if should_standardize:
            mean_vals, std_vals = locate_standarization_files(pl.Path(autoencoder_path))
            mean_tensor = torch.tensor(mean_vals, device=device, dtype=dtype)
            std_tensor = torch.tensor(std_vals, device=device, dtype=dtype)
            descr_tensor = (descr_tensor - mean_tensor) / std_tensor

        # Pass through the full model (encoder + decoder)
        reconstructed_tensor = autoencoder_model(descr_tensor)

        # Invert standardization on the output to calculate errors in the original scale
        if should_standardize and mean_vals is not None and std_vals is not None:
            reconstructed_tensor = (reconstructed_tensor * std_tensor) + mean_tensor

        reconstructed_np = reconstructed_tensor.cpu().numpy()

    # Calculate global metrics
    mae = np.mean(np.abs(original_np - reconstructed_np))
    mse = np.mean((original_np - reconstructed_np) ** 2)
    rmse = np.sqrt(mse)

    # Calculate relative reconstruction error (scaled by total descriptor variance)
    data_variance = np.var(original_np)
    relative_error = mse / data_variance if data_variance > 0 else 0.0

    custom_print(
        f'Reconstruction Metrics -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | '
        f'Rel_Err: {relative_error:.4f}',
        'done',
    )

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'relative_error': relative_error,
        'predictions': reconstructed_np,
        'targets': original_np,
    }


@dataclass
class AutoencoderSettings:
    """Configuration settings for Autoencoder training."""

    # Environment & Hardware
    device: str = 'cpu'
    dtype: str = 'float32'
    rng_seed: int | None = None

    # Data & Paths
    dataset: str = 'all_descriptors.npz'
    model_path: str = 'autoencoder_model.pth'
    load_model: bool = False
    train_frac: float = 0.8
    valid_frac: float = 0.1
    test_frac: float = 0.1
    standardize_data: bool = True

    # Model Architecture
    l1_hidden_dim: int = 256
    l2_hidden_dim: int = 32
    bottleneck_dim: int = 2
    bias_flag: bool = True

    # Hyperparameters & Training
    num_epochs: int = 50
    batch_size: int = 4096
    patience: int = 5
    lr: float = 0.001
    weight_decay: float = 1e-5
    loss: str = 'mse'

    # Weights & Biases Logging
    wandb: bool = False
    wandb_name: str | None = None
    wandb_project: str | None = None
