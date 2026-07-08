"""Allegro training utilities.

Targets the **new** nequip (>= 0.7) / nequip-allegro API, which uses a
Hydra + PyTorch-Lightning config schema and the ``nequip-train`` CLI.
"""

from __future__ import annotations

from pathlib import Path

# Directory (relative to the training working directory) where nequip's
# Hydra run writes checkpoints, logs and metrics. Kept in sync with the
# ``hydra.run.dir`` entry in the generated config so the parser and the
# calcjob retrieve list know where to look.
RESULTS_DIR = 'results'


def build_allegro_train_config(
    settings_dict: dict,
    train_data_path: str,
    model_name: str,
    iteration: int,
    db_size: int,
) -> dict:
    """Build a nequip/Allegro Hydra training config from ATLAS settings.

    Translates the ATLAS TOML-based training settings into the config
    schema expected by the new ``nequip-train`` (Hydra + Lightning).

    Parameters
    ----------
    settings_dict : dict
        Training settings from the ``[allegro_train.train_settings]``
        TOML section. Accepts either the inner settings mapping or a
        dict that still wraps them under a ``train_settings`` key.
    train_data_path : str
        Path to the training data file (extxyz).
    model_name : str
        Unique name for this training run.
    iteration : int
        Current AL loop iteration number.
    db_size : int
        Number of structures in the training database.

    Returns
    -------
    dict
        A dictionary representing the ``nequip-train`` YAML config.
    """
    train_settings = dict(
        settings_dict.get('train_settings', None) or settings_dict
    )

    train_file_name = Path(train_data_path).name

    # -- pull ATLAS hyperparameters (with sensible defaults) ---------------
    seed = train_settings.pop('seed', 0)
    r_max = train_settings.pop('r_max', 5.0)
    num_layers = train_settings.pop('num_layers', 2)
    l_max = train_settings.pop('l_max', 1)
    parity = train_settings.pop('parity', True)
    num_tensor_features = train_settings.pop('num_features', 32)
    num_scalar_features = train_settings.pop('num_scalar_features', 64)
    lr = train_settings.pop('lr', 0.001)
    batch_size = train_settings.pop('batch_size', 5)
    max_epochs = train_settings.pop('max_num_epochs', 100)
    model_dtype = train_settings.pop('default_dtype', 'float32')
    compile_mode = train_settings.pop('compile_mode', 'eager')

    chemical_symbols = train_settings.pop('chemical_symbols', None)
    if chemical_symbols is None:
        chemical_symbols = _detect_chemical_symbols(train_data_path)

    val_fraction = train_settings.pop('validation_fraction', 0.1)
    train_fraction = round(1.0 - val_fraction, 6)
    val_fraction = round(val_fraction, 6)

    monitored_metric = 'val0_epoch/weighted_sum'

    config = {
        'run': ['train'],
        'cutoff_radius': r_max,
        'model_type_names': list(chemical_symbols),
        'chemical_species': '${model_type_names}',
        'monitored_metric': monitored_metric,
        # -- data ---------------------------------------------------------
        'data': {
            '_target_': 'nequip.data.datamodule.ASEDataModule',
            'seed': seed,
            'split_dataset': {
                'file_path': train_file_name,
                'train': train_fraction,
                'val': val_fraction,
            },
            'transforms': [
                {
                    '_target_': (
                        'nequip.data.transforms.'
                        'ChemicalSpeciesToAtomTypeMapper'
                    ),
                    'model_type_names': '${model_type_names}',
                },
                {
                    '_target_': (
                        'nequip.data.transforms.NeighborListTransform'
                    ),
                    'r_max': '${cutoff_radius}',
                },
            ],
            'ase_args': {'format': 'extxyz'},
            'key_mapping': {
                'REF_energy': 'total_energy',
                'REF_forces': 'forces',
            },
            'train_dataloader': {
                '_target_': 'torch.utils.data.DataLoader',
                'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0,
            },
            'val_dataloader': {
                '_target_': 'torch.utils.data.DataLoader',
                'batch_size': batch_size,
                'num_workers': 0,
            },
            'stats_manager': {
                '_target_': 'nequip.data.CommonDataStatisticsManager',
                'type_names': '${model_type_names}',
            },
        },
        # -- trainer ------------------------------------------------------
        'trainer': {
            '_target_': 'lightning.Trainer',
            'accelerator': 'auto',
            'max_epochs': max_epochs,
            'enable_checkpointing': True,
            'logger': {
                '_target_': 'lightning.pytorch.loggers.CSVLogger',
                'save_dir': '${hydra:runtime.output_dir}',
                'name': 'metrics',
            },
            'callbacks': [
                {
                    '_target_': (
                        'lightning.pytorch.callbacks.ModelCheckpoint'
                    ),
                    'dirpath': '${hydra:runtime.output_dir}',
                    'monitor': '${monitored_metric}',
                    'filename': 'best',
                    'save_last': True,
                },
            ],
        },
        # -- training module (loss / metrics / optimizer / model) ---------
        'training_module': {
            '_target_': 'nequip.train.EMALightningModule',
            'loss': {
                '_target_': 'nequip.train.EnergyForceLoss',
                'per_atom_energy': True,
                'coeffs': {'total_energy': 1.0, 'forces': 1.0},
            },
            'val_metrics': {
                '_target_': 'nequip.train.EnergyForceMetrics',
                'coeffs': {'total_energy_rmse': 1.0, 'forces_rmse': 1.0},
            },
            'train_metrics': '${training_module.val_metrics}',
            'optimizer': {
                '_target_': 'torch.optim.Adam',
                'lr': lr,
            },
            'model': {
                '_target_': 'allegro.model.AllegroModel',
                'compile_mode': compile_mode,
                'seed': seed,
                'model_dtype': model_dtype,
                'type_names': '${model_type_names}',
                'r_max': '${cutoff_radius}',
                'radial_chemical_embed': {
                    '_target_': 'allegro.nn.TwoBodyBesselScalarEmbed',
                    'num_bessels': 8,
                    'bessel_trainable': False,
                    'polynomial_cutoff_p': 6,
                },
                'radial_chemical_embed_dim': num_scalar_features,
                'scalar_embed_mlp_hidden_layers_depth': 1,
                'scalar_embed_mlp_hidden_layers_width': num_scalar_features,
                'scalar_embed_mlp_nonlinearity': 'silu',
                'l_max': l_max,
                'num_layers': num_layers,
                'num_scalar_features': num_scalar_features,
                'num_tensor_features': num_tensor_features,
                'allegro_mlp_hidden_layers_depth': 1,
                'allegro_mlp_hidden_layers_width': num_scalar_features,
                'allegro_mlp_nonlinearity': 'silu',
                'parity': parity,
                'tp_path_channel_coupling': True,
                'readout_mlp_hidden_layers_depth': 1,
                'readout_mlp_hidden_layers_width': num_scalar_features,
                'readout_mlp_nonlinearity': 'silu',
                'avg_num_neighbors': (
                    '${training_data_stats:per_type_num_neighbors_mean}'
                ),
                'per_type_energy_shifts': (
                    '${training_data_stats:per_atom_energy_mean}'
                ),
                'per_type_energy_scales': (
                    '${training_data_stats:forces_rms}'
                ),
                'per_type_energy_scales_trainable': False,
                'per_type_energy_shifts_trainable': False,
            },
        },
        # -- keep nequip's Hydra outputs in a known directory -------------
        'hydra': {
            'run': {'dir': RESULTS_DIR},
            'output_subdir': None,
        },
    }

    # Pass through any remaining/advanced settings verbatim so power users
    # can override arbitrary nequip config keys from the TOML.
    for key, value in train_settings.items():
        config.setdefault(key, value)

    return config


def _detect_chemical_symbols(train_data_path: str) -> list[str]:
    """Extract the sorted set of chemical symbols from the training file."""
    train_path = Path(train_data_path)
    if not train_path.exists():
        return []

    from ase.io import read

    images = read(str(train_path), index=':')
    return sorted({s for atoms in images for s in atoms.get_chemical_symbols()})


def deploy_allegro_model(
    checkpoint_path: str | Path,
    device: str = 'cpu',
) -> Path | None:
    """Compile a trained Allegro checkpoint for inference.

    Uses ``nequip-compile`` to turn a Lightning checkpoint into a
    compiled model file usable by :class:`nequip.ase.NequIPCalculator`
    (via ``from_compiled_model``) and by ``pair_allegro`` in LAMMPS.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained ``.ckpt`` checkpoint file.
    device : str
        Target device for the compiled model (``'cpu'`` or ``'cuda'``).

    Returns
    -------
    Path | None
        Path to the compiled ``.nequip.pt2`` file, or None on failure.
    """
    import subprocess

    checkpoint_path = Path(checkpoint_path)
    compiled_path = checkpoint_path.with_suffix('.nequip.pt2')

    result = subprocess.run(
        [
            'nequip-compile',
            '--input-path',
            str(checkpoint_path),
            '--output-path',
            str(compiled_path),
            '--device',
            device,
            '--mode',
            'aotinductor',
            '--target',
            'ase',
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not compiled_path.exists():
        return None

    return compiled_path
