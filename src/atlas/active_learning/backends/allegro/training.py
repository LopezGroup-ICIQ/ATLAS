"""Allegro training utilities.

Targets the new nequip (>= 0.7) / nequip-allegro API, which uses a
Hydra + PyTorch-Lightning config schema and the `nequip-train` CLI.
"""

from __future__ import annotations

from pathlib import Path

# Directory (relative to the training working directory) where nequip's
# Hydra run writes checkpoints, logs and metrics. Kept in sync with the
# `hydra.run.dir` entry in the generated config so the parser and the
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
    schema expected by the new `nequip-train` (Hydra + Lightning).

    Parameters
    ----------
    settings_dict : dict
        Training settings from the `[allegro_train.train_settings]`
        TOML section. Accepts either the inner settings mapping or a
        dict that still wraps them under a `train_settings` key.
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
        A dictionary representing the `nequip-train` YAML config.
    """
    train_settings = dict(settings_dict.get('train_settings') or settings_dict)

    train_file_name = Path(train_data_path).name

    # -- pull ATLAS hyperparameters (with sensible defaults) ---------------
    # When the user does not pin a seed, draw a random one. This function is
    # called once per committee member, so each gets a distinct seed and the
    # committee members are actually different models (required for meaningful
    # uncertainty estimation). The chosen seed is stored in the config dict,
    # so provenance still records the exact value used.
    seed = train_settings.pop('seed', None)
    if seed is None:
        import secrets

        seed = secrets.randbelow(2**31)
    r_max = train_settings.pop('r_max', 5.0)
    num_layers = train_settings.pop('num_layers', 2)
    l_max = train_settings.pop('l_max', 1)
    parity = train_settings.pop('parity', True)
    num_tensor_features = train_settings.pop('num_features', 32)
    num_scalar_features = train_settings.pop('num_scalar_features', 64)
    lr = train_settings.pop('lr', 0.001)
    batch_size = train_settings.pop('batch_size', 5)
    # DataLoader worker processes. 0 (the previous hardcoded value) loads data
    # on the main process and stalls the GPU between steps; a few workers build
    # neighbour lists in parallel. Tunable from the TOML; default to a modest 4.
    num_workers = int(train_settings.pop('num_workers', 4))
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
                        'nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper'
                    ),
                    'model_type_names': '${model_type_names}',
                },
                {
                    '_target_': ('nequip.data.transforms.NeighborListTransform'),
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
                'num_workers': num_workers,
                'persistent_workers': num_workers > 0,
            },
            'val_dataloader': {
                '_target_': 'torch.utils.data.DataLoader',
                'batch_size': batch_size,
                'num_workers': num_workers,
                'persistent_workers': num_workers > 0,
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
                    '_target_': ('lightning.pytorch.callbacks.ModelCheckpoint'),
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
                'per_type_energy_scales': ('${training_data_stats:forces_rms}'),
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


def _resolve_nequip_cmd(tool: str, python_executable: str) -> list[str]:
    """Return a command prefix that runs a nequip CLI tool.

    `tool` is a nequip console script such as `nequip-compile` or
    `nequip-package`. These stubs are fragile inside containers (broken
    shebang / missing exec bit / PATH resolving to a non-executable), which
    raises `PermissionError` when exec'd directly. We instead run them through
    `python_executable`. `python <script>` ignores both the exec bit and
    the shebang, and the script stub imports the right (version-specific)
    module itself.

    Resolution order:
    1. The registered console-script entry point, called as
       `python -c 'from <module> import <attr>; <attr>()'` (version-agnostic;
       only works when the interpreter can enumerate the package metadata).
    2. The tool file found via `shutil.which`, next to the interpreter, or
       by scanning `PATH` directly, run through the interpreter regardless
       of its executable bit. This is the branch that survives the container's
       non-executable stub.
    3. The bare command as a last resort.
    """
    import os
    import shutil
    from pathlib import Path as _Path

    # 1. Resolve the real entry-point target for this nequip version.
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group='console_scripts'):
            if ep.name == tool:
                module, _, attr = ep.value.partition(':')
                if module and attr:
                    return [
                        python_executable,
                        '-c',
                        f'import sys; from {module} import {attr}; sys.exit({attr}())',
                    ]
    except Exception:  # noqa: BLE001 - fall through to path-based lookup
        pass

    # 2. Locate the tool *file* and run it via the interpreter. We do NOT
    #    require it to be executable (`shutil.which` only returns files with
    #    the exec bit, so we also scan PATH and the interpreter's bin dir for
    #    the plain file). Running `python <file>` bypasses the missing exec
    #    bit / broken shebang that raises PermissionError in the container.
    candidate_paths: list[str] = []
    on_path = shutil.which(tool)
    if on_path is not None:
        candidate_paths.append(on_path)
    candidate_paths.append(str(_Path(python_executable).with_name(tool)))
    candidate_paths.extend(
        str(_Path(directory) / tool)
        for directory in os.environ.get('PATH', '').split(os.pathsep)
        if directory
    )
    for candidate in candidate_paths:
        cand_path = _Path(candidate)
        if candidate and cand_path.is_file():
            # Run the script with the python from its *own* bin dir (the venv it
            # was installed into, which has the matching nequip + torch), not
            # necessarily `python_executable`. This reproduces what the
            # shebang intends without depending on the exec bit or on the shell
            # picking the right interpreter.
            interpreter = python_executable
            for py_name in ('python', 'python3'):
                sibling_py = cand_path.parent / py_name
                if sibling_py.is_file():
                    interpreter = str(sibling_py)
                    break
            return [interpreter, str(cand_path)]

    # 3. Last resort.
    return [tool]


def deploy_allegro_model(
    checkpoint_path: str | Path,
    device: str = 'cpu',
    mode: str = 'aotinductor',
    target: str = 'ase',
) -> Path | None:
    """Compile a trained Allegro checkpoint for inference.

    Uses `nequip-compile` to turn a Lightning checkpoint into a
    compiled model file usable by
    :class:`nequip.integrations.ase.NequIPCalculator` (via
    `from_compiled_model`) and by `pair_allegro` in LAMMPS.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained `.ckpt` checkpoint file.
    device : str
        Target device for the compiled model (`'cpu'` or `'cuda'`).
    mode : str
        Compilation backend: `'torchscript'` (in-process JIT, no external
        C++ compiler needed) or `'aotinductor'` (faster, but requires a
        working C++ compiler on the node).
    target : str
        nequip-compile inference target (`'ase'` for the ASE calculator,
        `'pair_allegro'`/`'pair_nequip'` for LAMMPS).

    Returns
    -------
    Path | None
        Path to the compiled model (`*.nequip.pth` for torchscript,
        `*.nequip.pt2` for aotinductor), or None on failure.
    """
    import os
    import shutil
    import subprocess
    import sys

    # `nequip-compile` loads the checkpoint with torch.load; make the
    # subprocess inherit TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD when e3nn/torch
    # need it (see AllegroBackend._patch_torch_load_if_needed).
    from atlas.active_learning.backends.allegro import (
        _patch_torch_load_if_needed,
    )

    _patch_torch_load_if_needed()

    # Resolve to absolute paths: the compile runs with `cwd` set to the
    # checkpoint's directory (see below), so relative paths would otherwise
    # break.
    from atlas.active_learning.backends import model_file_stem

    checkpoint_path = Path(checkpoint_path).resolve()
    # nequip picks the loader from the extension, so it must match the mode.
    # Strip the input's (possibly compound, e.g. '.nequip.zip') extension before
    # appending the compiled suffix, so we get 'foo.nequip.pt2' not
    # 'foo.nequip.nequip.pt2'.
    compiled_suffix = '.nequip.pt2' if mode == 'aotinductor' else '.nequip.pth'
    compiled_path = checkpoint_path.parent / (
        model_file_stem(checkpoint_path.name) + compiled_suffix
    )

    # Reuse an existing compiled artifact instead of recompiling. Compilation is
    # device-specific and slow (aotinductor shells out to a C++ compiler), and a
    # single MD CalcJob may create the calculator repeatedly (e.g. once per MD
    # stage) for the same checkpoint in the same working dir.
    if compiled_path.exists():
        return compiled_path

    # Invoke nequip-compile through the *current* Python interpreter rather
    # than relying on the console-script entry point. Inside containers the
    # `nequip-compile` stub can fail to exec (broken shebang / missing exec
    # bit / PATH resolving to a non-executable), raising PermissionError.
    # Running it via `sys.executable` sidesteps all of that and guarantees
    # the same env that imported nequip.
    base_cmd = _resolve_nequip_cmd('nequip-compile', sys.executable)

    # aotinductor shells out to a C++ compiler. torch's inductor picks the
    # first it finds, which on some images is a broken/non-executable `icpc`.
    # If the user hasn't pinned CXX, point it at an *executable* g++/clang++ so
    # torch uses a working compiler instead of icpc.
    compile_env = os.environ.copy()
    if mode == 'aotinductor' and not compile_env.get('CXX'):
        for _cxx in ('g++', 'clang++', 'c++'):
            _found = shutil.which(_cxx)
            if _found is not None:
                compile_env['CXX'] = _found
                compile_env.setdefault('CC', shutil.which('gcc') or _found)
                break

    # `nequip-compile` takes the checkpoint and output paths as
    # positional arguments (not --input-path/--output-path flags).
    #
    # Run with `cwd` set to the checkpoint's directory: a nequip checkpoint
    # stores the training dataset by a *relative* filename and resolves it
    # against the process CWD when it rebuilds the datamodule to trace the
    # model. Under `singularity --contain` the CWD is otherwise $HOME or /,
    # not the bind-mounted working dir, so that data file (which sits next to
    # the checkpoint) would not be found.
    result = subprocess.run(
        [
            *base_cmd,
            str(checkpoint_path),
            str(compiled_path),
            '--device',
            device,
            '--mode',
            mode,
            '--target',
            target,
        ],
        capture_output=True,
        text=True,
        cwd=str(checkpoint_path.parent),
        env=compile_env,
    )

    if result.returncode != 0 or not compiled_path.exists():
        # Surface the failure so it appears in the calculation's
        # scheduler output rather than failing silently downstream.
        sys.stderr.write(
            f'nequip-compile failed (returncode={result.returncode}) for '
            f'checkpoint {checkpoint_path}:\n{result.stdout}\n{result.stderr}\n'
        )
        return None

    return compiled_path


def package_allegro_model(checkpoint_path: str | Path) -> Path | None:
    """Package a trained Allegro checkpoint into a self-contained archive.

    `nequip-package build` bundles the model code, weights and the metadata
    needed to trace/compile it into a portable `*.nequip.zip`. Unlike a raw
    `.ckpt`, the package can be compiled/loaded downstream without the
    original training dataset, which is why we run it here, at training time,
    where that dataset is still available.

    Must run in the training working directory: a checkpoint references its
    training dataset by a *relative* filename, which nequip-package resolves
    against the CWD when it loads the checkpoint. `parse_training_results`
    (this function's only caller) runs there, so the inherited CWD is correct.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained `.ckpt` checkpoint file.

    Returns
    -------
    Path | None
        Path to the `*.nequip.zip` package, or None on failure.
    """
    import subprocess
    import sys

    from atlas.active_learning.backends.allegro import (
        _patch_torch_load_if_needed,
    )

    _patch_torch_load_if_needed()

    checkpoint_path = Path(checkpoint_path)
    package_path = checkpoint_path.with_name(checkpoint_path.stem + '.nequip.zip')

    base_cmd = _resolve_nequip_cmd('nequip-package', sys.executable)

    # `nequip-package build <checkpoint> <output>`.
    result = subprocess.run(
        [*base_cmd, 'build', str(checkpoint_path), str(package_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not package_path.exists():
        sys.stderr.write(
            f'nequip-package failed (returncode={result.returncode}) for '
            f'checkpoint {checkpoint_path}:\n{result.stdout}\n{result.stderr}\n'
        )
        return None

    return package_path
