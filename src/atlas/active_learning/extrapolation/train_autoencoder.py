"""Train an autoencoder for dimensionality reduction."""

import copy
import pathlib as pl
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset, random_split

import atlas.active_learning.extrapolation.autoencoder as ae
import atlas.core.code_utils as atl_cut
from atlas.core.code_utils import custom_print


def _autocast_ctx(device, amp_dtype):
    """Autocast context for mixed-precision training, or a no-op if disabled."""
    if amp_dtype is None:
        return nullcontext()
    dev_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    return torch.autocast(device_type=dev_type, dtype=amp_dtype)


def train_loop(
    data_loader, model, loss_fn, optimizer, device, amp_dtype=None, scaler=None
):
    # Set the model to training mode, important for batch normalization
    # and dropout layers
    # Unnecessary in this situation (since no nn.Dropout, ... etc) in the
    # Autoencoder, but added for best practices.
    #
    # ``amp_dtype`` (torch.bfloat16 / torch.float16 / None) enables autocast
    # mixed-precision training; ``scaler`` is a torch.amp.GradScaler required for
    # float16 (bfloat16 does not need one). Both default off → fp32 behaviour.
    model.train()
    total_loss = 0

    for batch in data_loader:
        # Unpack the batch and move it to the GPU
        # TensorDataset returns a tuple, so we grab the first element
        inputs = batch[0].to(device)

        # Forward pass (Autoencoder targets its own input)
        optimizer.zero_grad()
        with _autocast_ctx(device, amp_dtype):
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

        # Backward pass (gradient scaling for float16)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    # Compute average loss
    avg_loss = total_loss / len(data_loader)

    return total_loss, avg_loss


def val_loop(data_loader, model, loss_fn, device, amp_dtype=None):
    # Set the model to evaluation mode - important for batch
    # normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(data_loader)
    total_loss = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients
    # are computed during validation reducing computations and memory usage
    # for tensors with requires_grad=True
    with torch.no_grad():
        for batch in data_loader:
            # Unpack the tuple and move data to the correct device
            inputs = batch[0].to(device)

            # Forward pass: compute the reconstruction (autocast if enabled)
            with _autocast_ctx(device, amp_dtype):
                reconstruction = model(inputs)

                # Compute loss: reconstruction error
                loss = loss_fn(reconstruction, inputs)

            # Track loss
            total_loss += loss.item()

    # Compute average loss
    avg_loss = total_loss / num_batches

    return avg_loss


def _load_dataset_as_point_array(data: np.ndarray | str | pl.Path) -> np.ndarray:
    # Load numpy array
    descr_data = np.load(data)

    # If npz file, extract the first array
    # This is a workaround for the npz file format
    # that stores the data in a dictionary-like structure
    # with the key 'arr_0' or 'descriptor'
    if isinstance(descr_data, np.lib.npyio.NpzFile):
        # New name
        point_arr = descr_data.get('descriptor')

        # Old name fallback
        if point_arr is None:
            point_arr = descr_data.get('arr_0')

        if point_arr is None:
            raise ValueError('No array found in the npz file.')
    elif isinstance(descr_data, np.ndarray):
        point_arr = descr_data
    else:
        raise ValueError('Invalid data type. Expected np.ndarray or npz file.')
    return point_arr


def safe_to_gpu(arrays, target_dtype, device):
    """
    Checks if there is enough VRAM to move an array to the GPU.
    Returns True if safe, False if it will likely OOM.
    """
    if torch.device(device).type != 'cuda':
        return True  # CPU RAM limits apply instead

    # Calculate required memory
    bytes_per_element = torch.tensor([], dtype=target_dtype).element_size()
    required_bytes = 0
    for arr in arrays:
        required_bytes += arr.size * bytes_per_element
    required_gb = required_bytes / (1024**3)

    # Check available VRAM
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024**3)

    atl_cut.custom_print(f'Target Tensor Size: {required_gb:.3f} GB', 'debug')
    atl_cut.custom_print(f'Available VRAM:     {free_gb:.3f} GB', 'debug')

    # Leave a safety margin (e.g., 500 MB) for PyTorch context and operations
    safety_margin = 500 * (1024**2)

    if (required_bytes + safety_margin) < free_bytes:
        atl_cut.custom_print('It is safe to store tensors on GPU.', 'debug')
        return True
    else:
        print('Tensors will likely cause OOM on GPU.', 'warning')
        return False


def split_dataset(
    data: str | pl.Path | np.ndarray,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    rng_seed: int = None,
    device: str = None,
    dtype=torch.float32,
):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the RNG seed for reproducibility
    if not rng_seed:
        rng_seed = np.random.randint(1, int(1e15))

    # Load the dataset
    if isinstance(data, (str, pl.Path)):
        point_arr = _load_dataset_as_point_array(data)
    elif isinstance(data, np.ndarray):
        point_arr = data

    # Select a fraction of the data for training, validation, and testing
    total_size = point_arr.shape[0]
    valid_data_size = int(total_size * valid_frac)
    test_data_size = int(total_size * test_frac)

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=rng_seed)
    torch.manual_seed(rng_seed)

    # Shuffle in-place
    # This modifies point_arr directly in memory without creating a copy.
    rng.shuffle(point_arr, axis=0)

    # Calculate split indices
    valid_end = valid_data_size
    test_end = valid_end + test_data_size

    # Slicing the array
    # This creates memory-efficient views, which should avoid
    # damn OOM errors, I hope.
    valid_arr = point_arr[:valid_end]
    test_arr = point_arr[valid_end:test_end]
    train_arr = point_arr[test_end:]

    safe_to_gpu(
        arrays=(valid_arr, test_arr, train_arr), target_dtype=dtype, device=device
    )

    train_data = torch.Tensor(train_arr).to(device=device, dtype=dtype)
    valid_data = torch.Tensor(valid_arr).to(device=device, dtype=dtype)
    test_data = torch.Tensor(test_arr).to(device=device, dtype=dtype)

    atl_cut.custom_print('Data loaded:', 'info')
    atl_cut.custom_print(f'  Training data array shape: {train_data.shape}', 'clean')
    atl_cut.custom_print(f'  Validation data array shape: {valid_data.shape}', 'clean')
    atl_cut.custom_print(f'  Test data array shape: {test_data.shape}', 'clean')
    print()

    return train_data, valid_data, test_data


def return_dataset_loader(
    dataset: str | pl.Path | np.ndarray,
    train_frac: float,
    valid_frac: float,
    dtype: str,
    batch_size: int,
    rng_seed: int,
):
    if isinstance(dataset, (str, pl.Path)):
        dataset = _load_dataset_as_point_array(dataset)

    # Wrap the full numpy array in a Dataset (Keep it on the CPU for now!)
    full_dataset = TensorDataset(torch.as_tensor(dataset, dtype=dtype))

    # Calculate sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_frac)
    valid_size = int(total_size * valid_frac)

    # catches any rounding remainder
    test_size = total_size - train_size - valid_size

    # Automatically split the dataset (this handles the random seed safely)
    if not rng_seed:
        rng_seed = np.random.randint(1, int(1e15))
        custom_print(
            f'No RNG seed provided. Generated random seed: {rng_seed}.', 'warning'
        )

    split_generator = torch.Generator().manual_seed(rng_seed)
    train_ds, valid_ds, test_ds = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=split_generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_training(args):
    """
    Train an autoencoder model for dimensionality reduction.

    Parameters
    ----------
    args : Namespace
        Namespace object containing the following attributes:
        - dataset : str
            Path to the dataset file.
        - device : str
            Device to run the model on.
            If not given, the device is set to cuda if available, else cpu.
        - dtype : str
            Data type to use for the model. One of float32 or float64.
            Default is float32.
        - lr : float
            Learning rate for the optimizer.
        - num_epochs : int
            Number of epochs to train the model.
        - batch_size : int
            Batch size for training.
        - patience : int
            Early stopping patience: number of epochs without validation loss
            improvement before stopping training. Also controls LR reduction.
        - train_frac : float
            Fraction of the dataset to use for training.
        - valid_frac : float
            Fraction of the dataset to use for validation.
        - test_frac : float
            Fraction of the dataset to use for testing.
        - l1_hidden_dim : int
            Number of hidden units in the first layer of the autoencoder.
        - l2_hidden_dim : int
            Number of hidden units in the second layer of the autoencoder.
        - weight_decay : float
            Weight decay for the optimizer.
        - bias_flag : bool
            Flag to include bias in the linear layers.
        - verbose : bool
            Flag to print verbose output.
        - rng_seed : int
            Random seed for reproducibility.
        - model_path : str
            Path to save the trained model.
        - wandb : bool
            Flag to enable logging to wandb.
        - wandb_project : str
            Name of the wandb project.
        - wandb_name : str
            Name of the wandb run.
        - standardize_data : bool
            Whether to normalize the data before training the autoencoder


    Returns
    -------
    Autoencoder
        Autoencoder model trained for dimensionality reduction.
    """
    # Set device if no device is given
    if not hasattr(args, 'device') or args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    atl_cut.custom_print(f"Running on device: '{device}'", 'info')

    # Setting verbosity
    if 'verbose' in vars(args):
        match args.verbose:
            case True:
                args.verbose = True
            case False:
                args.verbose = False
    else:
        args.verbose = True

    # Setting dtype
    if not hasattr(args, 'dtype'):
        args.dtype = torch.float32
    else:
        match args.dtype:
            case 'float32':
                args.dtype = torch.float32
            case 'float64':
                args.dtype = torch.float64
            case _:
                args.dtype = torch.float32
    atl_cut.custom_print(f"Using dtype: '{args.dtype}'", 'info')

    # Optional mixed-precision compute for training (storage stays args.dtype).
    # amp_dtype: None/'float32' (default) · 'tf32' · 'bfloat16' · 'float16'.
    amp_dtype = None
    grad_scaler = None

    # Saved TF32 backend flags, set just before the training loop and restored
    # in a finally so this global state never leaks out of run_training.
    _tf32_saved = None
    _amp = getattr(args, 'amp_dtype', None)
    match _amp:
        case 'tf32':
            pass  # flags set (and restored) around the training loop below
        case 'bfloat16':
            amp_dtype = torch.bfloat16
        case 'float16':
            amp_dtype = torch.float16
            grad_scaler = torch.amp.GradScaler(
                'cuda' if 'cuda' in str(device) else 'cpu'
            )
        case _:
            pass
    if _amp:
        atl_cut.custom_print(f"Training compute precision: '{_amp}'", 'info')

    # If no seed is given, generate a random seed
    if not hasattr(args, 'rng_seed'):
        args.rng_seed = np.random.randint(1, int(1e15))
    atl_cut.custom_print(f"Using RNG seed: '{args.rng_seed}'.", 'info')

    # Optionally normalizing data
    # Z-score standardization
    if hasattr(args, 'standardize_data') and args.standardize_data is True:
        atl_cut.custom_print('Standardization enabled. Normalizing data...', 'info')
        if isinstance(args.model_path, str):
            model_path = pl.Path(args.model_path).resolve()
        else:
            model_path = args.model_path.resolve()

        if isinstance(args.dataset, (str, pl.Path)):
            dataset = _load_dataset_as_point_array(args.dataset)
        else:
            dataset = args.dataset

        # Check if standardization files already exist (mean and std values)
        mean_vals, std_vals = ae.locate_standarization_files(model_path)

        if mean_vals is not None and std_vals is not None:
            atl_cut.custom_print('Loaded standardized values.')
        else:
            atl_cut.custom_print('Carrying out data standarization...')
            mean_vals = dataset.mean(axis=0)
            std_vals = dataset.std(axis=0)

            # To avoid division by zero, we can set any zero std to 1
            # (which means no scaling for that feature)
            std_vals[std_vals == 0] = 1.0

        # Save them alongside model
        np.save(model_path.parent / 'ae_mean_vals.npy', mean_vals)
        np.save(model_path.parent / 'ae_std_vals.npy', std_vals)

        # Apply standardization to the dataset
        dataset = (dataset - mean_vals) / std_vals

    else:
        dataset = args.dataset

    # dataset = args.dataset

    # Load data
    # Return as DataLoader for batch processing
    train_data, valid_data, test_data = return_dataset_loader(
        dataset,
        args.train_frac,
        args.valid_frac,
        args.dtype,
        args.batch_size,
        args.rng_seed,
    )

    # train_data, valid_data, test_data = split_dataset(
    #     data=dataset,
    #     device=device,
    #     train_frac=args.train_frac,
    #     valid_frac=args.valid_frac,
    #     test_frac=args.test_frac,
    #     rng_seed=args.rng_seed,
    #     dtype=args.dtype,
    # )

    # Number of input dimensions
    if isinstance(train_data, DataLoader):
        input_dim = train_data.dataset.dataset.tensors[0].shape[1]
    elif isinstance(dataset, np.ndarray):
        input_dim = dataset.shape[1]
    else:
        raise ValueError(f'Unexpected dataset type: {type(dataset)}')

    # Start a new wandb run to track this script
    if hasattr(args, 'wandb') and args.wandb is True:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            # set the name of the run
            name=args.wandb_name,
            # track hyperparameters and run metadata
            config=vars(args),
        )

        # Override for sweeps.
        # Sweep data is found on wandb.config, otherwise it is empty.
        # If wandb.config is created, then try to get the   key actually exists.
        # If it doesn't exist (no sweep), it falls back to the original args values.
        for key, value in wandb.config.items():
            if hasattr(args, key):
                setattr(args, key, value)

        if isinstance(args.dtype, str):
            match args.dtype:
                case 'float64':
                    args.dtype = torch.float64
                case _:
                    args.dtype = torch.float32

        # We want to ensure that the l2_hidden_dim is always smaller than the
        # l1_hidden_dim for a valid autoencoder architecture.
        if args.l2_hidden_dim >= args.l1_hidden_dim:
            args.l2_hidden_dim = args.l1_hidden_dim // 4
            # Log the updated dimension back to wandb so charts are accurate
            wandb.config.update(
                {'l2_hidden_dim': args.l2_hidden_dim}, allow_val_change=True
            )

    # Model initialization
    model = ae.Autoencoder(
        input_dim=input_dim,
        l1_dim=args.l1_hidden_dim,
        l2_dim=args.l2_hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        bias_flag=args.bias_flag,
    )

    model.to(device=device, dtype=args.dtype)

    match args.loss:
        # Define loss function as MSE (Mean Squared Error)
        case 'mse':
            criterion = nn.MSELoss()

        # Alternative loss function that penalizes small errors more heavily
        case 'weighted_mse':

            def weighted_mse_loss(output, target, weight=10):
                diff = (output - target) ** 2
                return torch.mean(weight * diff)

            criterion = weighted_mse_loss

        # Default is MSE
        case _:
            criterion = nn.MSELoss()

    # Define optimizer (Adam optimizer)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # The LR schedule must react strictly before early stopping, or it never
    # fires: ReduceLROnPlateau reduces once bad epochs *exceed* its patience
    # (i.e. at patience + 1), while the early-stopping test below triggers at
    # exactly `args.patience`. Sharing a single value therefore stopped training
    # one epoch before the first LR reduction could ever occur, so every run
    # effectively trained at a constant learning rate. `lr_patience` defaults to
    # a third of the early-stopping patience; callers passing only `patience`
    # keep working unchanged.
    lr_patience = getattr(args, 'lr_patience', None)
    if lr_patience is None:
        lr_patience = max(1, args.patience // 3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-6,
    )

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    atl_cut.custom_print(
        f'Starting autoencoder training for {args.num_epochs} epochs '
        f'(early stopping patience: {args.patience})...',
        'info',
    )

    if _amp == 'tf32':
        # fp32 tensors, tensor-core TF32 matmul (no autocast needed). Saved so
        # the global flags are restored in the finally below.
        _tf32_saved = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        for epoch in range(args.num_epochs):
            _, train_avg_loss = train_loop(
                data_loader=train_data,
                model=model,
                loss_fn=criterion,
                optimizer=optimizer,
                device=device,
                amp_dtype=amp_dtype,
                scaler=grad_scaler,
            )

            val_loss = val_loop(
                data_loader=valid_data,
                model=model,
                loss_fn=criterion,
                device=device,
                amp_dtype=amp_dtype,
            )

            # Perform test evaluation every 5 epochs
            if epoch % 5 == 0:
                test_loss = val_loop(
                    data_loader=test_data,
                    model=model,
                    loss_fn=criterion,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                if args.verbose:
                    atl_cut.custom_print(
                        str(
                            {
                                'Epoch': epoch,
                                'Train Avg. MSE': train_avg_loss,
                                'Validation Avg. MSE': val_loss,
                                'Test Avg. MSE': test_loss,
                                'lr': scheduler.get_last_lr()[0],
                            }
                        ),
                        'none',
                    )
                # log metrics to wandb
                if hasattr(args, 'wandb') and args.wandb:
                    wandb.log(
                        {
                            'epoch': epoch,
                            'train_loss': train_avg_loss,
                            'val_loss': val_loss,
                            'test_loss': test_loss,
                            'lr': scheduler.get_last_lr()[0],
                        }
                    )
            # Log metrics for normal epoch
            else:
                if args.verbose:
                    atl_cut.custom_print(
                        {
                            'Epoch': epoch,
                            'Train Avg. MSE': train_avg_loss,
                            'Validation Avg. MSE': val_loss,
                            'lr': scheduler.get_last_lr()[0],
                        },
                        'none',
                    )
                # log metrics to wandb
                if hasattr(args, 'wandb') and args.wandb:
                    wandb.log(
                        {
                            'epoch': epoch,
                            'train_loss': train_avg_loss,
                            'val_loss': val_loss,
                            'lr': scheduler.get_last_lr()[0],
                        }
                    )

            scheduler.step(metrics=val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                atl_cut.custom_print(
                    f'Early stopping at epoch {epoch} '
                    f'(no improvement for {args.patience} epochs). '
                    f'Best val loss: {best_val_loss:.6f}',
                    'info',
                )
                break
    finally:
        if _tf32_saved is not None:
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            ) = _tf32_saved

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    atl_cut.custom_print('Training complete!', 'done')

    if hasattr(args, 'model_path'):
        # Save the model
        # torch.save(model.state_dict(), args.model_path)
        save_path = pl.Path(args.model_path).absolute()
        torch.save(model, save_path)

        atl_cut.custom_print(f"Autoencoder model saved to '{save_path}'.", 'info')
    else:
        atl_cut.custom_print('Autoencoder model not saved.', 'warning')

    return model
