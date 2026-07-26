"""General utilities for the MLIP benchmark suite in ATLForge."""

import argparse
import atexit
import math
import pathlib as pl
import re
import shutil
import time
import tomllib
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from atlas.core.code_utils import (
    custom_print as _original_custom_print,
)

# Global UI manager instance
_ui_manager = None
_log_file = None
_plot_data = {}  # Store data for final multi-panel plot
_model_display_names = {}  # Maps model path to display name
_model_data = {}  # Global model data dictionary with path, name, and color

# Ignore all warnings
warnings.filterwarnings('ignore')


COLORS = [
    '#fe8019',
    '#038999',
    '#689d6a',
    '#b16286',
    '#458588',
    '#d79921',
    '#cc241d',
    '#98971a',
]

# Extend COLORS with tab20 colors to support more models
COLORS.extend(plt.cm.tab20.colors)


def adjust_color_brightness(hex_color, brightness_factor):
    """
    Adjust the brightness of a hex color.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., '#fe8019')
    brightness_factor : float
        Factor to adjust brightness
        > 1.0 = brighter
        < 1.0 = darker
        1.0 = no change

    Returns
    -------
    str
        Adjusted hex color string
    """
    if isinstance(hex_color, tuple) and len(hex_color) == 3:
        # If color is already an RGB tuple, convert to hex first
        hex_color = mcolors.to_hex(hex_color)

    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Adjust brightness
    adjusted_rgb = []
    for component in rgb:
        # Scale the component and clamp to 0-255
        adjusted = int(component * brightness_factor)
        adjusted_rgb.append(min(255, max(0, adjusted)))

    # Convert back to hex
    return f'#{adjusted_rgb[0]:02x}{adjusted_rgb[1]:02x}{adjusted_rgb[2]:02x}'


def create_args_from_toml(toml_dict: dict) -> argparse.Namespace:
    """
    Create an argparse.Namespace object from TOML configuration.

    Parameters
    ----------
    toml_dict : dict
        Dictionary loaded from TOML file

    Returns
    -------
    Namespace
        Arguments namespace populated with values from TOML
    """
    args = argparse.Namespace()

    # General settings
    general = toml_dict.get('general', {})
    args.output_dir = pl.Path(general.get('output_dir', './mlip_evaluation'))
    args.metal = general.get('metal', 'Cu')
    args.device = general.get('device', 'cuda')
    args.dtype = general.get('dtype', 'float64')
    args.no_rich_ui = general.get('no_rich_ui', False)

    # Models
    models = toml_dict.get('models', {})
    args.model_files = models.get('model_files', [])
    args.aiida_pks = models.get('aiida_pks', [])
    args.foundation_models = models.get('foundation_models', [])

    # Slab generation
    slab = toml_dict.get('slab_generation', {})
    args.surface_indices = slab.get('surface_indices', [1, 1, 1])
    args.supercell_size = slab.get('supercell_size', [3, 3, 4])
    args.vacuum = slab.get('vacuum', 10.0)

    # MD parameters
    md = toml_dict.get('md_parameters', {})
    args.temp = md.get('temp', 300.0)
    args.n_steps = md.get('n_steps', 10000)
    args.timestep = md.get('timestep', 2.0)
    args.friction = md.get('friction', 0.005)

    # Benchmarks selection
    benchmarks = toml_dict.get('benchmarks', {})
    args.run_energy_md = benchmarks.get('run_energy_md', False)
    args.run_accuracy_test_set = benchmarks.get('run_accuracy_test_set', False)
    args.run_elastic_properties = benchmarks.get('run_elastic_properties', False)
    args.run_defect_formation_energy = benchmarks.get(
        'run_defect_formation_energy', False
    )
    args.run_surface_energies = benchmarks.get('run_surface_energies', False)
    args.run_phonon_dispersion = benchmarks.get('run_phonon_dispersion', False)
    args.run_high_temp_md = benchmarks.get('run_high_temp_md', False)
    args.run_melting_point = benchmarks.get('run_melting_point', False)
    args.run_gsfe = benchmarks.get('run_gsfe', False)
    args.run_learning_curves = benchmarks.get('run_learning_curves', False)
    args.run_final_db_size = benchmarks.get('run_final_db_size', False)
    args.run_md_count = benchmarks.get('run_md_count', False)
    args.run_evaluate_database = benchmarks.get('run_evaluate_database', False)
    args.run_magic_cluster = benchmarks.get('run_magic_cluster', False)

    # Test set
    test_set = toml_dict.get('test_set', {})
    if 'test_set_path' in test_set:
        args.test_set_path = pl.Path(test_set['test_set_path'])
    else:
        args.test_set_path = None

    # Database evaluation
    db_eval = toml_dict.get('database_evaluation', {})
    if 'database_path' in db_eval:
        args.database_path = pl.Path(db_eval['database_path'])
    else:
        args.database_path = None

    # Magic cluster
    mc = toml_dict.get('magic_cluster', {})
    if 'magic_cluster_dft_refs' in mc:
        args.magic_cluster_dft_refs = pl.Path(mc['magic_cluster_dft_refs'])
    else:
        args.magic_cluster_dft_refs = None
    args.magic_cluster_sizes = mc.get(
        'magic_cluster_sizes', [13, 19, 55, 147, 309, 561]
    )

    # Defect formation
    defect_conf = toml_dict.get('defect_formation_energy')
    if defect_conf:
        args.defect_formation_energy_config = defect_conf

    # Surface energy benchmark
    surf = toml_dict.get('surface_energy_benchmark', {})
    args.surface_energy_benchmark_config = surf
    if 'dft_refs' in surf:
        args.surf_ene_benchmark_dft_refs = pl.Path(surf['dft_refs'])
    else:
        args.surf_ene_benchmark_dft_refs = None

    if 'bulk_structure' in surf:
        args.surf_ene_benchmark_bulk_structure = pl.Path(surf['bulk_structure'])
    else:
        args.surf_ene_benchmark_bulk_structure = None

    if 'slab_structures' in surf:
        # Convert dict to list format
        slab_list = [f'{idx}:{path}' for idx, path in surf['slab_structures'].items()]
        args.surf_ene_benchmark_slab_structures = slab_list
    else:
        args.surf_ene_benchmark_slab_structures = None

    # Melting point benchmark
    mp = toml_dict.get('melting_point_benchmark', {})
    args.melting_point_supercell_size = mp.get('supercell_size', [6, 6, 6])
    args.melting_point_solid_temp_K = mp.get('solid_temp_K', 1100.0)
    args.melting_point_liquid_temp_K = mp.get('liquid_temp_K', 1600.0)
    args.melting_point_nve_initial_T_test_K = mp.get('nve_initial_T_test_K', 800.0)
    args.melting_point_supercell_path = mp.get('melting_point_supercell_path', None)

    return args


def load_toml_config(toml_path: pl.Path) -> dict:
    """
    Load TOML configuration file.

    Parameters
    ----------
    toml_path : Path
        Path to the TOML configuration file

    Returns
    -------
    dict
        Parsed TOML configuration
    """
    if not toml_path.exists():
        raise FileNotFoundError(f'TOML config file not found: {toml_path}')

    with open(toml_path, 'rb') as f:
        return tomllib.load(f)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate and compare MLIPs performance using TOML configuration. '
        'All settings must be specified in a TOML configuration file.'
    )
    parser.add_argument(
        'config_path',
        type=pl.Path,
        nargs='?',
        default=None,
        help='Path to TOML configuration file. If not provided, will look for '
        '"atl_benchmark_settings.toml" in the current directory.',
    )

    return parser.parse_args()


def load_model_from_aiida(identifier: int, output_dir: pl.Path):
    """Load a model from an ATL AL Workchain."""
    from aiida import load_profile
    from aiida.orm import load_node

    load_profile()

    try:
        # Loading model from last workchain step
        base_workchain = load_node(identifier)
        last_workchain = base_workchain.called[-1]

        # Extract run name from the base workchain inputs
        run_name = None
        try:
            run_name = base_workchain.inputs.active_learning.run_name.value
            custom_print(
                f"Extracted run name: '{run_name}' from workchain {identifier}", 'debug'
            )
        except (AttributeError, KeyError) as e:
            custom_print(
                f'Could not extract run name from workchain {identifier}: {e}', 'warn'
            )
            # Fallback to using the identifier as the name
            run_name = f'workchain_{identifier}'

        if hasattr(base_workchain.outputs, 'final_model_file'):
            final_model_singlefile_node = base_workchain.outputs.final_model_file
        else:
            final_model_singlefile_node = last_workchain.outputs.m0_model_file

        # Saving model file to output directory
        with final_model_singlefile_node.as_path() as model_file:
            model_file_path = pl.Path(model_file)

            # Copying model to cwd
            model_file_path = shutil.copy(
                model_file_path, output_dir / model_file_path.name
            )

        # Store the mapping of model path to display name
        global _model_display_names
        _model_display_names[str(model_file_path)] = run_name

        custom_print(
            f"Loaded model '{run_name}' (file: {model_file_path.name}) "
            f'using ATL AL workchain PK/UUID: {identifier}',
            'info',
        )
        return model_file_path
    except Exception as e:
        custom_print(
            f'Failed to load model from Workchain pk {identifier}: {e}', 'error'
        )
        return None


def get_model_display_name(model_path: pl.Path) -> str:
    """
    Get the display name for a model.

    Returns run name if available, otherwise filename stem.
    """
    global _model_display_names
    path_str = str(model_path)
    if path_str in _model_display_names:
        return _model_display_names[path_str]
    else:
        # For models not loaded from AiiDA, use filename stem
        return model_path.stem


def create_foundation_model_calculator(
    foundation_model_spec, device='cuda', dtype='float64', **kwargs
):
    """
    Create a calculator for a foundation model.

    Parameters
    ----------
    foundation_model_spec : str
        Foundation model specification (e.g., "mace:small")
    device : str, optional
        Device to run calculations on (default: 'cuda')
    dtype : str, optional
        Data type for calculations (default: 'float64')

    Returns
    -------
    calculator
        Calculator instance for the foundation model

    Raises
    ------
    ValueError
        If foundation model specification is invalid
    NotImplementedError
        If the specified MLIP library is not supported
    """
    from atlas.active_learning.backends import (
        MLIPPretrainedModel,
        get_backend,
        list_backends,
        resolve_model_spec,
    )

    if ':' not in foundation_model_spec:
        raise ValueError(
            f"Invalid foundation model specification: '{foundation_model_spec}'. "
            "Expected format: 'library:model_name'"
        )

    library, model_name = foundation_model_spec.split(':', 1)

    if library.lower() not in list_backends():
        raise NotImplementedError(
            f"Foundation models for '{library}' are not supported. "
            f'Registered backends: {", ".join(list_backends())}.'
        )

    # `resolve_model_spec` normalises the variant through the backend, so the
    # historical 'mace:small' spelling still resolves to 'mace:mp-small'.
    backend_name, pretrained_id = resolve_model_spec(foundation_model_spec)
    backend = get_backend(backend_name)

    if not isinstance(backend, MLIPPretrainedModel):
        raise NotImplementedError(
            f"Backend '{backend_name}' does not provide pretrained foundation "
            'models.'
        )

    try:
        custom_print(
            f'Loading {backend_name} foundation model: {model_name}', 'info'
        )
        calculator = backend.create_calculator(
            model_path=pretrained_id,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        custom_print(
            f'Successfully loaded {backend_name} model: {model_name}', 'done'
        )
        return calculator
    except ImportError as e:
        raise ImportError(
            f"Failed to import the '{backend_name}' framework: {e}. Make sure "
            'it is installed with foundation model support.'
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to create a {backend_name} calculator with model "
            f"'{model_name}': {e}. This could be due to network issues or an "
            'invalid model name.'
        ) from e


class FoundationModelPath:
    """
    A class to represent foundation models in a way compatible with file-based models.

    This allows foundation models to be used alongside file-based models
    in the benchmark system.
    """

    def __init__(self, foundation_model_spec):
        self.foundation_model_spec = foundation_model_spec
        library, model_name = foundation_model_spec.split(':', 1)
        self.stem = f'{library}_{model_name}'
        self.name = f'{library}:{model_name}'

    def __str__(self):
        return self.foundation_model_spec

    def __repr__(self):
        return f"FoundationModelPath('{self.foundation_model_spec}')"


def create_foundation_model_paths(foundation_model_specs):
    """
    Create FoundationModelPath objects from foundation model specifications.

    Parameters
    ----------
    foundation_model_specs : list of str
        List of foundation model specifications

    Returns
    -------
    list of FoundationModelPath
        Foundation model path objects

    Raises
    ------
    ValueError
        If any foundation model specification is invalid
    """
    from atlas.active_learning.backends import list_backends, resolve_model_spec

    paths = []
    for spec in foundation_model_specs:
        # Validate the specification format
        if ':' not in spec:
            raise ValueError(
                f"Invalid foundation model specification: '{spec}'. "
                "Expected format: 'library:model_name'"
            )

        library, _model_name = spec.split(':', 1)

        # Validate supported libraries
        if library.lower() not in list_backends():
            raise ValueError(
                f"Unsupported foundation model library: '{library}'. "
                f'Registered backends: {", ".join(list_backends())}.'
            )

        # Let the backend validate its own variant: it warns on an unrecognised
        # one and raises when the backend serves no pretrained models at all.
        resolve_model_spec(spec)

        paths.append(FoundationModelPath(spec))

    return paths


def create_calculator_for_model(model_path, device='cuda', dtype='float64', **kwargs):
    """
    Create a calculator for either a file-based model or foundation model.

    Parameters
    ----------
    model_path : Path or FoundationModelPath
        Path to model file or foundation model specification
    device : str, optional
        Device to run calculations on (default: 'cuda')
    dtype : str, optional
        Data type for calculations (default: 'float64')
    **kwargs
        Additional arguments to pass to the calculator

    Returns
    -------
    calculator
        Calculator instance for the model
    """
    if hasattr(model_path, 'foundation_model_spec'):
        # Foundation model - ignore kwargs that don't apply to foundation models
        foundation_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ['enable_cueq']  # Foundation models don't support enable_cueq
        }
        return create_foundation_model_calculator(
            model_path.foundation_model_spec,
            device=device,
            dtype=dtype,
            **foundation_kwargs,
        )
    else:
        from atlas.active_learning.backends import get_backend

        backend = get_backend('mace')
        return backend.create_calculator(
            model_path=str(model_path), device=device, dtype=dtype, **kwargs
        )


def get_model_display_names():
    """Get the global model display names dictionary."""
    global _model_display_names
    return _model_display_names


def set_model_display_name(path, name):
    """Set a model display name."""
    global _model_display_names
    _model_display_names[path] = name


def initialize_model_data(model_paths):
    """
    Initialize the global model data dictionary with consistent color assignments.

    Parameters
    ----------
    model_paths : list
        List of model paths (can include FoundationModelPath objects)
    """
    global _model_data
    _model_data = {}

    for i, model_path in enumerate(model_paths):
        path_str = str(model_path)
        model_name = get_model_display_name(model_path)
        model_color = COLORS[i % len(COLORS)]

        _model_data[path_str] = {
            'model_path': model_path,
            'model_name': model_name,
            'model_color': model_color,
            'index': i,
        }


def get_model_data():
    """Get the global model data dictionary."""
    global _model_data
    return _model_data


def get_model_color(model_path):
    """
    Get the assigned color for a model.

    Parameters
    ----------
    model_path : Path or FoundationModelPath
        Model path or model object

    Returns
    -------
    str
        Hex color string assigned to this model
    """
    global _model_data
    path_str = str(model_path)
    if path_str in _model_data:
        return _model_data[path_str]['model_color']
    else:
        # Fallback to first color if model not found
        return COLORS[0]


def get_model_colors_by_names(model_names):
    """
    Get colors for a list of model names in the order they appear.

    Parameters
    ----------
    model_names : list
        List of model display names

    Returns
    -------
    list
        List of hex color strings corresponding to the model names
    """
    global _model_data
    colors = []

    # Create a mapping from names to data
    name_to_data = {data['model_name']: data for data in _model_data.values()}

    for name in model_names:
        if name in name_to_data:
            colors.append(name_to_data[name]['model_color'])
        else:
            # Fallback color if name not found
            colors.append(COLORS[len(colors) % len(COLORS)])

    return colors


def clear_model_data():
    """Clear the global model data dictionary."""
    global _model_data
    _model_data = {}


def save_plot_dual_format(base_path, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib plot in both PNG and SVG formats.

    Parameters
    ----------
    base_path : str or Path
        Path without extension (e.g., 'plot')
    dpi : int, optional
        DPI for PNG output (default: 300)
    bbox_inches : str, optional
        Bounding box setting for tight layout (default: 'tight')

    Returns
    -------
    tuple
        (png_path, svg_path) - Paths to the saved files
    """
    base_path = pl.Path(base_path)
    png_path = base_path.with_suffix('.png')
    svg_path = base_path.with_suffix('.svg')

    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(svg_path, format='svg', bbox_inches=bbox_inches)

    return png_path, svg_path


def get_plot_data():
    """Get the global plot data dictionary."""
    global _plot_data
    return _plot_data


def set_plot_data(key, value):
    """Set a value in the global plot data dictionary."""
    global _plot_data
    _plot_data[key] = value


def clear_plot_data():
    """Clear the global plot data dictionary."""
    global _plot_data
    _plot_data.clear()


def get_ui_manager():
    """Get the global UI manager instance."""
    global _ui_manager
    return _ui_manager


def set_ui_manager(ui_manager):
    """Set the global UI manager instance."""
    global _ui_manager
    _ui_manager = ui_manager


def create_final_multi_panel_plot(args):
    """Creates a single multi-panel figure with all benchmark plots."""
    if not _plot_data:
        custom_print('No plot data available for multi-panel figure.', 'warn')
        return

    # Set SVG font type to 'none' for editable text in Inkscape
    plt.rcParams['svg.fonttype'] = 'none'

    # Count available plot types
    plot_types = list(_plot_data.keys())
    n_plots = len(plot_types)

    if n_plots == 0:
        custom_print('No plots to display.', 'warn')
        return

    # Calculate grid layout (max 4 columns, filling row by row)
    cols = min(4, n_plots)
    rows = math.ceil(n_plots / cols)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_plots > 1 else axes

    # Plot each benchmark
    plot_idx = 0
    for plot_type, plot_info in _plot_data.items():
        ax = axes_flat[plot_idx]

        if plot_type in ['energy_md', 'energy_md_high_energy']:
            # Energy/Force time series plot
            energy_data = plot_info['energy_data']
            timestep = plot_info['timestep']
            model_names = list(energy_data.keys())

            # Get consistent colors for models
            colors = get_model_colors_by_names(model_names)

            for i, model_name in enumerate(model_names):
                energies = energy_data[model_name]
                # Generate time data from timestep and number of points
                time_data = np.arange(len(energies)) * timestep
                color = colors[i]
                ax.plot(time_data, energies, label=model_name, color=color, linewidth=2)

            ax.set_xlabel('Time (fs)')
            ax.set_ylabel('Energy (eV)')
            ax.set_title(plot_info['title'])
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        elif plot_type == 'defect_formation':
            # Bar chart for defect formation energies (supports grouped by surface)
            model_names = plot_info['model_names']
            groups = plot_info.get('groups', ['default'])
            results = plot_info.get('results')
            plot_relative_dft = plot_info.get('plot_relative_dft', False)

            # Legacy/Simple mode: 'values' provided directly (single group)
            if 'values' in plot_info:
                formation_energies = plot_info['values']
                colors = get_model_colors_by_names(model_names)
                bars = ax.bar(
                    model_names,
                    formation_energies,
                    color=colors,
                    edgecolor='#282828',
                    linewidth=1,
                )

                # Add value labels
                for bar, energy in zip(bars, formation_energies, strict=True):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(formation_energies) * 0.01,
                        plot_info.get('value_format', '{:.3f}').format(energy),
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )
            else:
                # Grouped mode
                dft_results = results.get('dft', {})

                # Determine brightness factors for groups
                n_groups = len(groups)
                if n_groups == 1:
                    brightness_factors = [1.0]
                elif n_groups == 2:
                    brightness_factors = [0.8, 1.2]
                elif n_groups == 3:
                    brightness_factors = [0.7, 1.0, 1.3]
                else:
                    brightness_factors = [
                        0.5 + (1.0 * i / (n_groups - 1)) for i in range(n_groups)
                    ]

                mlip_colors = get_model_colors_by_names(model_names)

                # Determine max energy for scaling labels (global or relative)
                max_energy = 0.0
                for model_res in results.values():
                    for group_res in model_res.values():
                        if (
                            isinstance(group_res, dict)
                            and 'formation_energy_eV' in group_res
                        ):
                            max_energy = max(
                                max_energy, abs(group_res['formation_energy_eV'])
                            )

                if plot_relative_dft and dft_results:
                    # SPLIT PLOT MODE: Relative energies (MLIP) top, Absolute DFT bottom

                    # Setup subplot grid
                    from matplotlib.gridspec import GridSpecFromSubplotSpec

                    ax.clear()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    gs = GridSpecFromSubplotSpec(
                        1, 2, ax.get_subplotspec(), width_ratios=[3, 1], wspace=0.3
                    )
                    ax_rel = ax.figure.add_subplot(gs[0])
                    ax_dft = ax.figure.add_subplot(gs[1])

                    # Plot 1: Relative Energies (MLIP Only)
                    bar_width = 0.8 / n_groups if n_groups > 1 else 0.8
                    x_pos = np.arange(len(model_names))

                    for i, group_name in enumerate(groups):
                        group_energies = []
                        group_colors = []
                        group_x_pos = []

                        dft_energy = dft_results.get(group_name, {}).get(
                            'formation_energy_eV', 0.0
                        )

                        for j, model_name in enumerate(model_names):
                            energy = 0.0
                            if group_name in results.get(model_name, {}):
                                energy = results[model_name][group_name][
                                    'formation_energy_eV'
                                ]

                            # Calculate relative energy
                            rel_energy = energy - dft_energy

                            # Color logic
                            base_color = mlip_colors[j]

                            adjusted_color = adjust_color_brightness(
                                base_color, brightness_factors[i]
                            )

                            group_energies.append(rel_energy)
                            group_colors.append(adjusted_color)

                            shift = i * bar_width if n_groups > 1 else 0
                            group_x_pos.append(x_pos[j] + shift)

                        # Plot bars
                        bars = ax_rel.bar(
                            group_x_pos,
                            group_energies,
                            bar_width,
                            color=group_colors,
                            edgecolor='#282828',
                            linewidth=1,
                        )

                        # Labels
                        for bar, val in zip(bars, group_energies, strict=True):
                            height = bar.get_height()
                            ax_rel.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height
                                + (
                                    max_energy * 0.02
                                    if val >= 0
                                    else -max_energy * 0.05
                                ),
                                f'{val:.3f}',
                                ha='center',
                                va='bottom' if val >= 0 else 'top',
                                fontsize=8,
                                rotation=90 if n_groups > 1 else 0,
                            )

                    if n_groups > 1:
                        tick_positions = x_pos + bar_width * (n_groups - 1) / 2
                        ax_rel.set_xticks(tick_positions)
                    else:
                        ax_rel.set_xticks(x_pos)
                    ax_rel.set_xticklabels(model_names)

                    ax_rel.set_ylabel('Energy Diff vs DFT (eV)', fontsize=9)
                    ax_rel.set_title(plot_info['title'], fontsize=10)
                    ax_rel.grid(True, linestyle='--', alpha=0.6)

                    # Plot 2: Absolute DFT Energies
                    dft_x_pos = np.arange(1)  # Only one "model" (DFT)

                    for i, group_name in enumerate(groups):
                        dft_val = dft_results.get(group_name, {}).get(
                            'formation_energy_eV', 0.0
                        )
                        color = adjust_color_brightness(
                            '#282828', brightness_factors[i]
                        )

                        shift = i * bar_width if n_groups > 1 else 0
                        pos = (
                            dft_x_pos[0] + shift - (bar_width * (n_groups - 1) / 2)
                        )  # Center around 0

                        bar = ax_dft.bar(
                            pos,
                            dft_val,
                            bar_width,
                            color=color,
                            edgecolor='black',
                            linewidth=1,
                            label=group_name,
                        )

                        # Label
                        height = bar[0].get_height()
                        ax_dft.text(
                            pos,
                            height + max_energy * 0.02,
                            f'{dft_val:.3f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                        )

                        # Group label inside bar
                        if n_groups > 1:
                            ax_dft.text(
                                pos,
                                height * 0.5,
                                group_name,
                                ha='center',
                                va='center',
                                rotation=90,
                                color='white',
                                fontweight='bold',
                                fontsize=8,
                            )

                    ax_dft.set_xticks([])
                    ax_dft.set_xlabel('DFT References', fontsize=9)
                    ax_dft.set_ylabel('Abs Energy (eV)', fontsize=9)
                    ax_dft.grid(True, linestyle='--', alpha=0.6)

                else:
                    # STANDARD PLOT MODE (Absolute energies, DFT included in main bars)
                    all_model_names = model_names.copy()
                    if dft_results:
                        all_model_names.append('DFT')

                    bar_width = 0.8 / n_groups if n_groups > 1 else 0.8
                    x_pos = np.arange(len(all_model_names))

                    for i, group_name in enumerate(groups):
                        group_energies = []
                        group_colors = []
                        group_x_pos = []

                        for j, model_name in enumerate(all_model_names):
                            energy = 0.0

                            if model_name == 'DFT':
                                if group_name in dft_results:
                                    energy = dft_results[group_name][
                                        'formation_energy_eV'
                                    ]
                                adjusted_color = adjust_color_brightness(
                                    '#282828', brightness_factors[i]
                                )
                            else:
                                if group_name in results.get(model_name, {}):
                                    energy = results[model_name][group_name][
                                        'formation_energy_eV'
                                    ]

                                # Get base color index
                                if model_name in model_names:
                                    base_color = mlip_colors[
                                        model_names.index(model_name)
                                    ]
                                else:
                                    base_color = '#888888'

                                adjusted_color = adjust_color_brightness(
                                    base_color, brightness_factors[i]
                                )

                            group_energies.append(energy)
                            group_colors.append(adjusted_color)

                            shift = i * bar_width if n_groups > 1 else 0
                            group_x_pos.append(x_pos[j] + shift)

                        # Plot bars for this group
                        bars = ax.bar(
                            group_x_pos,
                            group_energies,
                            bar_width,
                            color=group_colors,
                            edgecolor='#282828',
                            linewidth=1,
                        )

                        # Add labels
                        for bar, energy in zip(bars, group_energies, strict=True):
                            height = bar.get_height()
                            # Value label
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + max_energy * 0.02,
                                f'{energy:.3f}',
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                rotation=90 if n_groups > 1 else 0,
                            )

                            # Group label inside bar if multiple groups and
                            # bar is tall enough
                            if n_groups > 1 and abs(height) > (max_energy * 0.15):
                                text_color = (
                                    'white' if brightness_factors[i] < 0.9 else 'black'
                                )
                                # Position near bottom
                                label_y = bar.get_y() + (height * 0.5)

                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    label_y,
                                    f'{group_name}',
                                    ha='center',
                                    va='center',
                                    rotation=90,
                                    color=text_color,
                                    fontsize=8,
                                    fontweight='bold',
                                )

                    if n_groups > 1:
                        tick_positions = x_pos + bar_width * (n_groups - 1) / 2
                        ax.set_xticks(tick_positions)
                    else:
                        ax.set_xticks(x_pos)
                    ax.set_xticklabels(all_model_names)

                    ax.set_xlabel('Model')
                    ax.set_ylabel(plot_info['ylabel'])
                    ax.set_title(plot_info['title'])
                    ax.grid(True, linestyle='--', alpha=0.6)

        elif plot_type == 'surface_energies':
            # Multiple surface energies subplot
            model_names = plot_info['model_names']
            surface_names = plot_info['surface_names']
            results = plot_info['results']
            dft_surface_energies = plot_info.get('dft_surface_energies', {})
            plot_relative_dft = plot_info.get('plot_relative_dft', False)

            # Add DFT as an additional "model" if DFT data is available
            all_model_names = model_names.copy()
            if dft_surface_energies:
                all_model_names.append('DFT')

            if (zoom_info := plot_info.get('zoom_info')) is not None:
                global_zoom = zoom_info.get('global_zoom', {})
                should_zoom_globally = global_zoom.get('enabled', False)

            if should_zoom_globally:
                # Apply global zoom settings
                y_min = global_zoom.get('y_min', None)
                y_max = global_zoom.get('y_max', None)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)

            # If multiple surfaces, create mini-subplots within this panel
            n_surfaces = len(surface_names)
            if n_surfaces == 1:
                # Single surface - bar chart with model-based colors
                surface_name = ''.join(map(str, surface_names[0]))

                if plot_relative_dft and dft_surface_energies:
                    # SPLIT PLOT MODE (Single Surface)
                    from matplotlib.gridspec import GridSpecFromSubplotSpec

                    ax.clear()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    gs = GridSpecFromSubplotSpec(
                        1, 2, ax.get_subplotspec(), width_ratios=[3, 1], wspace=0.3
                    )
                    ax_rel = ax.figure.add_subplot(gs[0])
                    ax_dft = ax.figure.add_subplot(gs[1])

                    # Get DFT energy
                    dft_energy = dft_surface_energies.get(surface_name, 0.0)

                    # 1. Relative Energies (MLIP Only)
                    mlip_energies = []
                    mlip_colors = get_model_colors_by_names(model_names)

                    for _, model_name in enumerate(model_names):
                        energy = 0.0
                        if surface_name in results[model_name]['surfaces']:
                            energy = results[model_name]['surfaces'][surface_name][
                                'surface_energy_J_per_m2'
                            ]
                        mlip_energies.append(energy - dft_energy)

                    bars_rel = ax_rel.bar(
                        model_names,
                        mlip_energies,
                        color=mlip_colors,
                        edgecolor='#282828',
                        linewidth=1,
                    )

                    # Add labels for relative plot
                    max_abs_rel = (
                        max([abs(e) for e in mlip_energies]) if mlip_energies else 1.0
                    )
                    for bar, val in zip(bars_rel, mlip_energies, strict=True):
                        height = bar.get_height()
                        ax_rel.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height
                            + (max_abs_rel * 0.02 if val >= 0 else -max_abs_rel * 0.05),
                            f'{val:.3f}',
                            ha='center',
                            va='bottom' if val >= 0 else 'top',
                            fontsize=9,
                        )

                    ax_rel.set_ylabel('Surf. Energy Diff vs DFT (J/m²)', fontsize=9)
                    ax_rel.set_title(
                        f'{plot_info["metal"]}({surface_name})', fontsize=10
                    )
                    ax_rel.grid(True, linestyle='--', alpha=0.6)
                    # if len(model_names) > 4:
                    #     ax_rel.tick_params(axis='x', labelrotation=45)
                    #     plt.setp(ax_rel.get_xticklabels(), ha='right')

                    # 2. Absolute DFT Energy
                    bar_dft = ax_dft.bar(
                        ['DFT'],
                        [dft_energy],
                        color='#282828',
                        edgecolor='black',
                        linewidth=1,
                    )

                    height = bar_dft[0].get_height()
                    ax_dft.text(
                        bar_dft[0].get_x() + bar_dft[0].get_width() / 2.0,
                        height + dft_energy * 0.01,
                        f'{dft_energy:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )

                    ax_dft.set_ylabel('Abs Energy (J/m²)', fontsize=9)
                    ax_dft.set_title('DFT Reference', fontsize=10)
                    ax_dft.grid(True, linestyle='--', alpha=0.6)

                else:
                    # STANDARD PLOT MODE (Single Surface)
                    energies = []
                    colors = []

                    # Get consistent colors for MLIP models
                    mlip_colors = get_model_colors_by_names(model_names)

                    for model_name in all_model_names:
                        if model_name == 'DFT':
                            # Get DFT energy for this surface
                            if surface_name in dft_surface_energies:
                                energies.append(dft_surface_energies[surface_name])
                            else:
                                energies.append(0)
                            # Use distinct color for DFT (black)
                            colors.append('#282828')
                        else:
                            # Regular MLIP model
                            mlip_idx = model_names.index(model_name)
                            if surface_name in results[model_name]['surfaces']:
                                energies.append(
                                    results[model_name]['surfaces'][surface_name][
                                        'surface_energy_J_per_m2'
                                    ]
                                )
                            else:
                                energies.append(0)
                            # Use model-specific consistent color
                            colors.append(mlip_colors[mlip_idx])

                    bars = ax.bar(
                        all_model_names,
                        energies,
                        color=colors,
                        edgecolor='#282828',
                        linewidth=1,
                    )
                    ax.set_title(f'{plot_info["metal"]}({surface_name}) Surface Energy')
                    ax.set_ylabel('Surface Energy (J/m²)')

                    # Rotate x-axis labels if more than 4 bars for better visibility
                    # if len(all_model_names) > 4:
                    #     ax.tick_params(axis='x', labelrotation=45)
                    #     plt.setp(ax.get_xticklabels(), ha='right')

                    # Add value labels
                    for bar, energy in zip(bars, energies, strict=True):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + max(energies) * 0.01,
                            f'{energy:.3f}',
                            ha='center',
                            va='bottom',
                            fontsize=9,
                        )
            else:
                # Multiple surfaces - grouped bar chart
                if plot_relative_dft and dft_surface_energies:
                    # SPLIT PLOT MODE (Multiple Surfaces)
                    from matplotlib.gridspec import GridSpecFromSubplotSpec

                    ax.clear()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    gs = GridSpecFromSubplotSpec(
                        1, 2, ax.get_subplotspec(), width_ratios=[3, 1], wspace=0.3
                    )
                    ax_rel = ax.figure.add_subplot(gs[0])
                    ax_dft = ax.figure.add_subplot(gs[1])

                    # 1. Relative Energies (MLIP Only)
                    bar_width = 0.8 / n_surfaces if n_surfaces > 1 else 0.8
                    x_pos = np.arange(len(model_names))

                    # Determine brightness factors
                    if n_surfaces == 1:
                        brightness_factors = [1.0]
                    elif n_surfaces == 2:
                        brightness_factors = [0.8, 1.2]
                    elif n_surfaces == 3:
                        brightness_factors = [0.7, 1.0, 1.3]
                    else:
                        brightness_factors = [
                            0.5 + (1.0 * i / (n_surfaces - 1))
                            for i in range(n_surfaces)
                        ]

                    mlip_colors = get_model_colors_by_names(model_names)
                    max_abs_rel = 0.0

                    # First pass to find max value for scaling
                    for _, surface_indices in enumerate(surface_names):
                        surface_name = ''.join(map(str, surface_indices))
                        dft_val = dft_surface_energies.get(surface_name, 0.0)

                        for model_name in model_names:
                            val = 0.0
                            if surface_name in results[model_name]['surfaces']:
                                val = results[model_name]['surfaces'][surface_name][
                                    'surface_energy_J_per_m2'
                                ]
                            max_abs_rel = max(max_abs_rel, abs(val - dft_val))

                    if max_abs_rel == 0:
                        max_abs_rel = 1.0

                    for i, surface_indices in enumerate(surface_names):
                        surface_name = ''.join(map(str, surface_indices))
                        dft_val = dft_surface_energies.get(surface_name, 0.0)

                        surface_data = []
                        for j, model_name in enumerate(model_names):
                            val = 0.0
                            if surface_name in results[model_name]['surfaces']:
                                val = results[model_name]['surfaces'][surface_name][
                                    'surface_energy_J_per_m2'
                                ]

                            rel_val = val - dft_val
                            base_color = mlip_colors[j]

                            color = adjust_color_brightness(
                                base_color, brightness_factors[i]
                            )

                            surface_data.append(
                                {
                                    'val': rel_val,
                                    'color': color,
                                    'x_pos': j + i * bar_width if n_surfaces > 1 else j,
                                }
                            )

                        x_positions = [d['x_pos'] for d in surface_data]
                        vals = [d['val'] for d in surface_data]
                        colors = [d['color'] for d in surface_data]

                        bars = ax_rel.bar(
                            x_positions,
                            vals,
                            bar_width,
                            color=colors,
                            edgecolor='#282828',
                            linewidth=1,
                        )

                        for bar, val in zip(bars, vals, strict=True):
                            height = bar.get_height()
                            ax_rel.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height
                                + (
                                    max_abs_rel * 0.02
                                    if val >= 0
                                    else -max_abs_rel * 0.05
                                ),
                                f'{val:.3f}',
                                ha='center',
                                va='bottom' if val >= 0 else 'top',
                                fontsize=8,
                                rotation=90 if n_surfaces > 1 else 0,
                            )

                    if n_surfaces > 1:
                        tick_positions = x_pos + bar_width * (n_surfaces - 1) / 2
                        ax_rel.set_xticks(tick_positions)
                    else:
                        ax_rel.set_xticks(x_pos)
                    ax_rel.set_xticklabels(model_names)
                    ax_rel.set_ylabel('Surf. Energy Diff vs DFT (J/m²)', fontsize=9)
                    ax_rel.set_title(
                        f'{plot_info["metal"]} Surface Energies', fontsize=10
                    )
                    ax_rel.grid(True, linestyle='--', alpha=0.6)

                    # 2. Absolute DFT Energies
                    dft_x_pos = np.arange(1)

                    for i, surface_indices in enumerate(surface_names):
                        surface_name = ''.join(map(str, surface_indices))
                        dft_val = dft_surface_energies.get(surface_name, 0.0)

                        color = adjust_color_brightness(
                            '#282828', brightness_factors[i]
                        )
                        shift = i * bar_width if n_surfaces > 1 else 0
                        pos = dft_x_pos[0] + shift - (bar_width * (n_surfaces - 1) / 2)

                        bar = ax_dft.bar(
                            pos,
                            dft_val,
                            bar_width,
                            color=color,
                            edgecolor='black',
                            linewidth=1,
                        )

                        height = bar[0].get_height()
                        ax_dft.text(
                            pos,
                            height + dft_val * 0.01,
                            f'{dft_val:.3f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                        )

                        if n_surfaces > 1:
                            ax_dft.text(
                                pos,
                                height * 0.5,
                                f'({surface_name})',
                                ha='center',
                                va='center',
                                rotation=90,
                                color='white'
                                if brightness_factors[i] < 0.9
                                else 'black',
                                fontweight='bold',
                                fontsize=8,
                            )

                    ax_dft.set_xticks([])
                    ax_dft.set_xlabel('DFT References', fontsize=9)
                    ax_dft.set_ylabel('Abs Energy (J/m²)', fontsize=9)
                    ax_dft.grid(True, linestyle='--', alpha=0.6)

                else:
                    # STANDARD PLOT MODE (Multiple Surfaces)
                    bar_width = 0.8 / len(surface_names)
                    x_pos = range(len(all_model_names))

                    # Define brightness factors for each surface (centered around 1.0)
                    # For 3 surfaces: [0.7, 1.0, 1.3] gives darker, normal, brighter
                    n_surfaces = len(surface_names)
                    if n_surfaces == 1:
                        brightness_factors = [1.0]
                    elif n_surfaces == 2:
                        brightness_factors = [0.8, 1.2]
                    elif n_surfaces == 3:
                        brightness_factors = [0.7, 1.0, 1.3]
                    else:
                        # For more surfaces, distribute brightness factors evenly
                        brightness_factors = [
                            0.5 + (1.0 * i / (n_surfaces - 1))
                            for i in range(n_surfaces)
                        ]

                    for i, surface_indices in enumerate(surface_names):
                        surface_name = ''.join(map(str, surface_indices))

                        # Collect energies for each model for this surface
                        surface_data = []
                        for j, model_name in enumerate(all_model_names):
                            if model_name == 'DFT':
                                # Get DFT energy for this surface
                                if surface_name in dft_surface_energies:
                                    energy = dft_surface_energies[surface_name]
                                else:
                                    energy = 0
                                # Use dark color for DFT, adjusted for brightness
                                adjusted_color = adjust_color_brightness(
                                    '#282828', brightness_factors[i]
                                )
                            else:
                                # Regular MLIP model
                                if surface_name in results[model_name]['surfaces']:
                                    energy = results[model_name]['surfaces'][
                                        surface_name
                                    ]['surface_energy_J_per_m2']
                                else:
                                    energy = 0
                                # Get base color for this model and adjust brightness
                                mlip_idx = model_names.index(model_name)
                                mlip_colors = get_model_colors_by_names(model_names)
                                base_color = mlip_colors[mlip_idx]

                                adjusted_color = adjust_color_brightness(
                                    base_color, brightness_factors[i]
                                )

                            surface_data.append(
                                {
                                    'energy': energy,
                                    'color': adjusted_color,
                                    'x_pos': j + i * bar_width,
                                    'brightness_factor': brightness_factors[i],
                                }
                            )

                        # Plot bars for this surface
                        x_positions = [data['x_pos'] for data in surface_data]
                        energies = [data['energy'] for data in surface_data]
                        colors = [data['color'] for data in surface_data]

                        bars = ax.bar(
                            x_positions,
                            energies,
                            bar_width,
                            color=colors,
                            edgecolor='#282828',
                            linewidth=1,
                        )

                        # Add surface name labels on bars
                        for bar, data in zip(bars, surface_data, strict=True):
                            bar_height = bar.get_height()

                            # Add energy value label
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                bar_height + max(energies) * 0.02,
                                f'{data["energy"]:.3f}',
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                rotation=90,
                            )

                            # Add label even if bar height is very small
                            # Choose text color based on brightness factor
                            # Use white for darker bars, black for rest
                            if abs(bar_height) > 1e-10:  # Much smaller threshold
                                text_color = (
                                    'white'
                                    if data['brightness_factor'] < 0.9
                                    else 'black'
                                )

                                # Calculate label position based on current y-axis range
                                y_min, y_max = ax.get_ylim()
                                y_range = y_max - y_min
                                # Position label at 10% from bottom of visible range
                                label_y = y_min + (y_range * 0.1)

                                # Add surface name rotated 90 degrees
                                ax.text(
                                    x=bar.get_x() + bar.get_width() / 2.0,
                                    y=label_y,
                                    s=f'({surface_name})',
                                    ha='center',
                                    va='center',
                                    rotation=90,
                                    fontweight='bold',
                                    color=text_color,
                                    fontsize=10,  # Increased font size
                                    zorder=100,  # Much higher z-order
                                    transform=ax.transData,
                                )

                    ax.set_xlabel('Model')
                    ax.set_ylabel('Surface Energy (J/m²)')
                    ax.set_title(f'{plot_info["metal"]} Surface Energies')
                    tick_positions = [
                        x + bar_width * (len(surface_names) - 1) / 2 for x in x_pos
                    ]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(all_model_names)

                    # Rotate x-axis labels if more than 4 bars for better visibility
                    # if len(all_model_names) > 4:
                    #     ax.tick_params(axis='x', labelrotation=45)
                    #     plt.setp(ax.get_xticklabels(), ha='right')

                ax.grid(True, linestyle='--', alpha=0.6)

        elif plot_type == 'high_temp_md':
            # High-temperature MD heatmap
            results = plot_info['results']
            model_names = list(results.keys())

            # Get temperatures from the stored data (fallback to default range)
            if results and model_names:
                first_result = results[model_names[0]]
                if 'defects_detected' in first_result:
                    temperatures = list(range(len(first_result['defects_detected'])))
                else:
                    temperatures = [1000]  # Default high temp
            else:
                temperatures = [1000]  # Default high temp

            # Prepare defect matrix
            defect_matrix = []
            for model_name in model_names:
                if 'defects_detected' in results[model_name]:
                    defect_matrix.append(results[model_name]['defects_detected'])
                else:
                    defect_matrix.append([False] * len(temperatures))

            # Create heatmap
            ax.imshow(
                defect_matrix,
                cmap='RdYlGn_r',
                aspect='auto',
                vmin=0,
                vmax=1,
                interpolation='nearest',
            )

            # Set labels
            ax.set_xticks(range(len(temperatures)))
            ax.set_xticklabels([f'{temp}K' for temp in temperatures], rotation=45)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names)

            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(temperatures)):
                    text = 'FAIL' if defect_matrix[i][j] else 'PASS'
                    color = 'white' if defect_matrix[i][j] else 'black'
                    ax.text(
                        j,
                        i,
                        text,
                        ha='center',
                        va='center',
                        color=color,
                        fontweight='bold',
                    )

            ax.set_xlabel('Temperature')
            ax.set_ylabel('MLIP Model')
            ax.set_title(plot_info['title'])

        elif plot_type == 'database_sizes':
            # Database size comparison bar chart (final sizes only)
            run_names = plot_info['run_names']
            final_sizes = plot_info['final_sizes']

            # Get consistent colors for models
            colors = get_model_colors_by_names(run_names)
            bars = ax.bar(
                run_names,
                final_sizes,
                color=colors,
                alpha=1.0,
                edgecolor='#282828',
                linewidth=1,
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(final_sizes) * 0.01,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

            ax.set_xlabel('AL Run')
            ax.set_ylabel('Number of Structures')
            ax.set_title(plot_info['title'])
            ax.set_xticklabels(run_names)
            ax.grid(True, linestyle='--', alpha=0.6)

        elif plot_type == 'learning_curves':
            # Learning curves plot - RMSE vs training database size
            learning_data = plot_info['data']

            # Use matplotlib's gridspec for proper subplot management
            from matplotlib.gridspec import GridSpecFromSubplotSpec

            # Clear and hide the main axis to prevent background interference
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Create a 2x1 gridspec within this subplot
            gs = GridSpecFromSubplotSpec(
                2, 1, ax.get_subplotspec(), height_ratios=[1, 1], hspace=0.3
            )

            # Create the two subaxes
            ax_energy = ax.figure.add_subplot(gs[0])
            ax_force = ax.figure.add_subplot(gs[1])

            # Plot each AL run
            run_names = list(learning_data.keys())
            run_colors = get_model_colors_by_names(run_names)

            for i, (run_name, data) in enumerate(learning_data.items()):
                color = run_colors[i]
                iterations = data['iterations']
                energy_rmse = data['energy_rmse']
                force_rmse = data['force_rmse']

                # Skip iteration 0 (usually has RMSE = 0)
                if len(iterations) > 1:
                    iterations_plot = iterations[1:]
                    energy_rmse_plot = energy_rmse[1:]
                    force_rmse_plot = force_rmse[1:]
                else:
                    iterations_plot = iterations
                    energy_rmse_plot = energy_rmse
                    force_rmse_plot = force_rmse

                ax_energy.plot(
                    iterations_plot,
                    energy_rmse_plot,
                    'o-',
                    color=color,
                    label=run_name,
                    linewidth=2,
                    markersize=4,
                )
                ax_force.plot(
                    iterations_plot,
                    force_rmse_plot,
                    'o-',
                    color=color,
                    label=run_name,
                    linewidth=2,
                    markersize=4,
                )

            # Configure energy subplot
            ax_energy.set_ylabel('Energy RMSE\n(meV/atom)', fontsize=9, labelpad=2)
            ax_energy.set_title('Active Learning Curves', fontsize=11, pad=10)
            ax_energy.grid(True, linestyle='--', alpha=0.6)
            ax_energy.legend(fontsize=8)
            ax_energy.tick_params(axis='both', which='major', labelsize=8)
            ax_energy.set_xticklabels([])  # Remove x-axis labels from top plot
            ax_energy.set_xlim(left=0)  # Start x-axis at 0

            # Configure force subplot
            ax_force.set_xlabel('Active Learning Iteration', fontsize=10)
            ax_force.set_ylabel('Force RMSE\n(meV/Å)', fontsize=9, labelpad=2)
            ax_force.grid(True, linestyle='--', alpha=0.6)
            ax_force.legend(fontsize=8)
            ax_force.tick_params(axis='both', which='major', labelsize=8)
            ax_force.set_xlim(left=0)  # Start x-axis at 0

        elif plot_type == 'melting_point':
            # Bar chart for melting points
            model_names = plot_info['model_names']
            melting_points = plot_info['values']

            # Get consistent colors for models
            colors = get_model_colors_by_names(model_names)
            bars = ax.bar(
                model_names,
                melting_points,
                color=colors,
                edgecolor='#282828',
                linewidth=1,
            )

            ax.set_xlabel('Model')
            ax.set_ylabel(plot_info['ylabel'])
            ax.set_title(plot_info['title'])
            ax.grid(True, linestyle='--', alpha=0.6)

            # Add value labels
            for bar, temp in zip(bars, melting_points, strict=True):
                if temp > 0:  # Only show label if melting point was determined
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(melting_points) * 0.01,
                        plot_info['value_format'].format(temp),
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )

        elif plot_type == 'md_count':
            # Bar chart for MD calculation counts
            model_names = plot_info['model_names']
            md_counts = plot_info['values']

            # Get consistent colors for models
            colors = get_model_colors_by_names(model_names)
            bars = ax.bar(
                model_names,
                md_counts,
                color=colors,
                edgecolor='#282828',
                linewidth=1,
            )

            # Rotate x-axis labels if needed
            # if len(model_names) > 3:
            #     ax.tick_params(axis='x', rotation=45)

            ax.set_xlabel('Model')
            ax.set_ylabel(plot_info['ylabel'])
            ax.set_title(plot_info['title'])
            ax.grid(True, linestyle='--', alpha=0.6)

            # Add value labels
            for bar, count in zip(bars, md_counts, strict=True):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(md_counts) * 0.01 if max(md_counts) > 0 else 0.1,
                    plot_info['value_format'].format(count),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

        elif plot_type == 'database_evaluation':
            # Database evaluation with energy and force error comparison
            model_names = plot_info['model_names']
            mean_energy_errors = plot_info['mean_energy_errors']
            mean_force_errors = plot_info['mean_force_errors']

            # Clear and hide the main axis to prevent background interference
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Create a 2x1 gridspec within this subplot
            from matplotlib.gridspec import GridSpecFromSubplotSpec

            gs = GridSpecFromSubplotSpec(
                2, 1, ax.get_subplotspec(), height_ratios=[1, 1], hspace=0.4
            )

            # Create the two subaxes
            ax_energy = ax.figure.add_subplot(gs[0])
            ax_force = ax.figure.add_subplot(gs[1])

            # Colors for bars
            colors = get_model_colors_by_names(model_names)

            # Energy errors subplot
            bars_energy = ax_energy.bar(
                model_names,
                mean_energy_errors,
                color=colors,
                edgecolor='#282828',
                linewidth=1,
            )
            ax_energy.set_ylabel('Energy Error\n(meV/atom)', fontsize=9)
            ax_energy.set_title(plot_info['title'], fontsize=11, pad=10)
            ax_energy.grid(True, linestyle='--', alpha=0.6)
            ax_energy.tick_params(axis='both', which='major', labelsize=8)
            ax_energy.set_xticklabels([])  # Remove x-labels from top plot

            # Add value labels on energy bars
            for bar, error in zip(bars_energy, mean_energy_errors, strict=True):
                height = bar.get_height()
                ax_energy.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(mean_energy_errors) * 0.02,
                    f'{error:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

            # Force errors subplot
            bars_force = ax_force.bar(
                model_names,
                mean_force_errors,
                color=colors,
                edgecolor='#282828',
                linewidth=1,
            )
            ax_force.set_xlabel('Model', fontsize=10)
            ax_force.set_ylabel('Force Error\n(eV/Å)', fontsize=9)
            ax_force.grid(True, linestyle='--', alpha=0.6)
            ax_force.tick_params(axis='both', which='major', labelsize=8)

            # Rotate x-axis labels if needed
            # if len(model_names) > 3:
            #     ax.tick_params(axis='x', rotation=45)

            # Add value labels on force bars
            for bar, error in zip(bars_force, mean_force_errors, strict=True):
                height = bar.get_height()
                ax_force.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(mean_force_errors) * 0.02,
                    f'{error:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

        elif plot_type == 'database_evaluation_detailed':
            # Structure-by-structure error plots
            model_names = plot_info['model_names']
            results = plot_info['results']

            # Clear and hide the main axis to prevent background interference
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Create a 2x1 gridspec within this subplot
            from matplotlib.gridspec import GridSpecFromSubplotSpec

            gs = GridSpecFromSubplotSpec(
                2, 1, ax.get_subplotspec(), height_ratios=[1, 1], hspace=0.4
            )

            # Create the two subaxes
            ax_energy = ax.figure.add_subplot(gs[0])
            ax_force = ax.figure.add_subplot(gs[1])

            # Use consistent color scheme
            colors = get_model_colors_by_names(model_names)

            # Plot energy errors for each model
            for i, model_name in enumerate(model_names):
                model_data = results[model_name]
                struct_indices = model_data['structure_indices']
                energy_errors = model_data['energy_errors_meV_per_atom']

                ax_energy.plot(
                    struct_indices,
                    energy_errors,
                    label=model_name,
                    color=colors[i],
                    marker='o',
                    markersize=2,
                    linewidth=1,
                    alpha=0.8,
                )

            ax_energy.set_ylabel('Energy Error\n(meV/atom)', fontsize=9)
            ax_energy.set_title(plot_info['title'], fontsize=11, pad=10)
            ax_energy.legend(fontsize=8)
            ax_energy.grid(True, alpha=0.3)
            ax_energy.set_yscale('log')
            ax_energy.tick_params(axis='both', which='major', labelsize=8)
            ax_energy.set_xticklabels([])  # Remove x-labels from top plot

            # Plot force errors for each model
            for i, model_name in enumerate(model_names):
                model_data = results[model_name]
                struct_indices = model_data['structure_indices']
                force_errors = model_data['force_errors_eV_per_A']

                ax_force.plot(
                    struct_indices,
                    force_errors,
                    label=model_name,
                    color=colors[i],
                    marker='o',
                    markersize=2,
                    linewidth=1,
                    alpha=0.8,
                )

            ax_force.set_xlabel('Structure Index', fontsize=10)
            ax_force.set_ylabel('Force Error\n(eV/Å)', fontsize=9)
            ax_force.legend(fontsize=8)
            ax_force.grid(True, alpha=0.3)
            ax_force.set_yscale('log')
            ax_force.tick_params(axis='both', which='major', labelsize=8)

        elif plot_type == 'magic_cluster':
            # Magic number cluster energies bar chart
            model_names = plot_info['model_names']
            magic_numbers = plot_info['magic_numbers']
            cluster_energies_data = plot_info['cluster_energies_data']

            # Create grouped bar chart for different cluster sizes
            n_clusters = len(magic_numbers)
            n_models = len(model_names)

            if n_clusters == 1:
                # Single cluster size - simple bar chart
                cluster_size = magic_numbers[0]
                energies = cluster_energies_data[cluster_size]

                # Filter out None values and corresponding model names
                valid_data = [
                    (name, energy)
                    for name, energy in zip(model_names, energies, strict=True)
                    if energy is not None
                ]
                if valid_data:
                    valid_names, valid_energies = zip(*valid_data, strict=True)
                else:
                    valid_names, valid_energies = [], []

                # Create colors - DFT gets black/dark color
                colors = []
                mlip_models = [n for n in model_names if n != 'DFT']
                mlip_colors = get_model_colors_by_names(mlip_models)

                for name in valid_names:
                    if name == 'DFT':
                        colors.append('#282828')
                    else:
                        # Find original model index for consistent coloring
                        if name in mlip_models:
                            orig_idx = mlip_models.index(name)
                            colors.append(mlip_colors[orig_idx])
                        else:
                            colors.append('#888888')  # Fallback color

                bars = ax.bar(
                    valid_names,
                    valid_energies,
                    color=colors,
                    edgecolor='#282828',
                    linewidth=1,
                )

                ax.set_title(f'{cluster_size}-Atom {plot_info["title"]}')
                ax.set_xlabel('Model')
                ax.set_ylabel(plot_info['ylabel'])

                # Add value labels with appropriate text color
                for bar, energy, name in zip(
                    bars, valid_energies, valid_names, strict=True
                ):
                    height = bar.get_height()
                    text_color = 'white' if name == 'DFT' else 'black'
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height / 2.0,  # Center vertically in the middle of the bar
                        f'{energy:.3f}',
                        ha='center',
                        va='center',  # Center vertically
                        fontsize=9,
                        color=text_color,
                    )

                # Rotate x-axis labels if needed
                # if len(valid_names) > 4:
                #     ax.tick_params(axis='x', labelrotation=45)
                #     plt.setp(ax.get_xticklabels(), ha='right')

            else:
                # Multiple cluster sizes - grouped bar chart
                bar_width = 0.8 / n_clusters
                x_positions = np.arange(n_models)

                # Define brightness factors for each cluster size
                # Distribute brightness factors evenly across cluster sizes
                # Smaller cluster sizes get lighter colors, larger ones get darker
                if n_clusters == 1:
                    brightness_factors = [1.0]
                elif n_clusters == 2:
                    brightness_factors = [1.3, 0.7]
                elif n_clusters == 3:
                    brightness_factors = [1.4, 1.0, 0.6]
                else:
                    # For more cluster sizes, distribute brightness factors evenly
                    # Start bright for small clusters, end dark for large clusters
                    brightness_factors = [
                        1.5 - (1.0 * k / (n_clusters - 1)) for k in range(n_clusters)
                    ]

                for i, cluster_size in enumerate(magic_numbers):
                    energies = cluster_energies_data[cluster_size]

                    # Filter out None values but keep position consistency
                    plot_energies = []
                    plot_colors = []
                    plot_positions = []

                    for j, (name, energy) in enumerate(
                        zip(model_names, energies, strict=True)
                    ):
                        if energy is not None:
                            plot_energies.append(energy)
                            plot_positions.append(x_positions[j] + i * bar_width)

                            if name == 'DFT':
                                # Use brightness-adjusted dark color for DFT
                                adjusted_color = adjust_color_brightness(
                                    '#282828', brightness_factors[i]
                                )
                                plot_colors.append(adjusted_color)
                            else:
                                # Find original model index for consistent base coloring
                                mlip_models = [n for n in model_names if n != 'DFT']
                                if name in mlip_models:
                                    orig_idx = mlip_models.index(name)
                                    base_color = COLORS[orig_idx % len(COLORS)]
                                    # Apply brightness factor to create shades

                                    adjusted_color = adjust_color_brightness(
                                        base_color, brightness_factors[i]
                                    )
                                    plot_colors.append(adjusted_color)
                                else:
                                    plot_colors.append('#888888')  # Fallback

                    if plot_energies:  # Only plot if we have data
                        bars = ax.bar(
                            plot_positions,
                            plot_energies,
                            bar_width,
                            label=f'{cluster_size} atoms',
                            color=plot_colors,
                            edgecolor='#282828',
                            linewidth=1,
                        )

                        # Add value labels with appropriate text color
                        for bar, energy in zip(bars, plot_energies, strict=True):
                            height = bar.get_height()

                            # Determine text color based on brightness factor
                            # Use white text for darker bars (brightness < 0.8)
                            # Use black text for lighter bars (brightness >= 0.8)
                            brightness = brightness_factors[i]
                            text_color = 'white' if brightness < 0.8 else 'black'

                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height / 2.0,  # Center vertically in middle of bar
                                f'{energy:.3f}',
                                ha='center',
                                va='center',  # Center vertically
                                fontsize=7,
                                rotation=90,
                                color=text_color,
                            )

                # Set labels and ticks
                ax.set_xlabel('Model')
                ax.set_ylabel(plot_info['ylabel'])
                ax.set_title(plot_info['title'])

                # Center the x-tick labels
                tick_positions = x_positions + bar_width * (n_clusters - 1) / 2
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(model_names)

                # Rotate x-axis labels if needed
                # if len(model_names) > 4:
                #     ax.tick_params(axis='x', labelrotation=45)
                #     plt.setp(ax.get_xticklabels(), ha='right')

                # Add legend
                ax.legend(fontsize=8, loc='best')

            ax.grid(True, linestyle='--', alpha=0.6)

        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()

    # Save in both PNG and SVG formats
    base_path = args.output_dir / 'all_benchmarks_summary'
    png_path, svg_path = save_plot_dual_format(base_path, dpi=300)

    custom_print(f"Multi-panel summary plot saved to '{png_path}'", 'success')
    custom_print(f"Multi-panel summary plot saved to '{svg_path}'", 'success')
    plt.close(fig)


def _initialize_log_file():
    """Initialize the log file with a timestamp."""
    global _log_file
    if _log_file is None:
        timestamp = int(time.time())
        log_filename = f'atl_run_mlip_benchmark_{timestamp}.log'
        _log_file = open(log_filename, 'w', encoding='utf-8')  # noqa
        print(f'Logging to file: {log_filename}')
        # Register cleanup function
        atexit.register(_cleanup_log_file)


def _cleanup_log_file():
    """Close the log file properly."""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


def _write_to_log_file(message):
    """Write a message to the log file."""
    global _log_file
    if _log_file:
        # Create a clean version without Rich markup for file logging
        clean_message = re.sub(r'\[/?[a-zA-Z0-9_ ]+\]', '', message)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        _log_file.write(f'[{timestamp}] {clean_message}\n')
        _log_file.flush()


def custom_print(message, level='info'):
    """Custom print function that integrates with Rich UI and logs to file."""
    global _ui_manager, _log_file

    # Initialize log file if not already done
    _initialize_log_file()

    # Write to log file with clean message
    _write_to_log_file(f'{level.upper()}: {message}' if level != 'empty' else message)

    if _ui_manager:
        # Format message with level for Rich UI
        if level == 'info':
            formatted = f'[blue]INFO[/blue]: {message}'
        elif level == 'warn':
            formatted = f'[yellow]WARN[/yellow]: {message}'
        elif level == 'error':
            formatted = f'[red]ERROR[/red]: {message}'
        elif level == 'done':
            formatted = f'[green]DONE[/green]: {message}'
        elif level == 'debug':
            formatted = f'[bright_black]DEBUG[/bright_black]: {message}'
        else:
            formatted = message

        _ui_manager.log(formatted)
    else:
        # Fallback to original function
        _original_custom_print(message, level)


class RichUIManager:
    """Manages the Rich CLI interface for the benchmark suite."""

    def __init__(self, benchmarks_to_run):
        self.console = Console()
        self.benchmarks_to_run = benchmarks_to_run
        self.current_benchmark = None
        self.completed_benchmarks = set()
        self.logs = []
        self.live = None
        self.progress = None
        self.overall_task = None

    def create_layout(self):
        """Create the main layout structure."""
        layout = Layout()

        # Split into header and body
        layout.split_column(Layout(name='header', size=3), Layout(name='body'))

        # Split body into sidebar and main content
        layout['body'].split_row(Layout(name='sidebar', size=30), Layout(name='main'))

        return layout

    def create_progress_panel(self):
        """Create the progress bar panel."""
        if self.progress is None:
            self.progress = Progress(
                TextColumn('[bold blue]{task.description}'),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                console=self.console,
                expand=True,
            )
            self.overall_task = self.progress.add_task(
                'Overall Progress', total=len(self.benchmarks_to_run)
            )

        return Panel(
            self.progress,
            title='[bold cyan]ATLAS - MLIP Benchmark Suite',
            border_style='blue',
            expand=True,
        )

    def create_sidebar_panel(self):
        """Create the sidebar showing benchmark status."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column('Status', width=3)
        table.add_column('Benchmark', min_width=20)

        for benchmark in self.benchmarks_to_run:
            if benchmark in self.completed_benchmarks:
                status = '[green]✓[/green]'
                style = 'green'
            elif benchmark == self.current_benchmark:
                status = '[yellow]►[/yellow]'
                style = 'yellow bold'
            else:
                status = '[dim]○[/dim]'
                style = 'dim'

            table.add_row(status, f'[{style}]{benchmark}[/{style}]')

        return Panel(table, title='[bold]Benchmarks', border_style='green')

    def create_log_panel(self):
        """Create the log display panel."""
        # Calculate available height for logs based on terminal size
        # Account for: header panel (5 lines), sidebar, panel borders, padding
        terminal_height = self.console.size.height
        # Reserve 10 lines for UI elements
        available_height = max(10, terminal_height - 10)

        # Show the most recent logs that fit in the available display space
        total_logs = len(self.logs)
        max_logs = min(total_logs, available_height)
        visible_logs = self.logs[-max_logs:] if max_logs > 0 else self.logs
        log_text = '\n'.join(visible_logs) if visible_logs else 'Waiting for logs...'

        # Add indicator if there are more logs above
        if len(self.logs) > max_logs:
            earlier_count = len(self.logs) - max_logs
            log_text = f'[dim]... ({earlier_count} earlier logs) ...[/dim]\n' + log_text

        benchmark_name = self.current_benchmark or 'Processing'
        title = f'[bold]Logs - {benchmark_name}'

        return Panel(
            Text.from_markup(log_text),
            title=title,
            border_style='cyan',
        )

    def update_display(self):
        """Update the entire display."""
        if self.live is None:
            return

        layout = self.create_layout()
        layout['header'].update(self.create_progress_panel())
        layout['sidebar'].update(self.create_sidebar_panel())
        layout['main'].update(self.create_log_panel())

        self.live.update(layout)

    def start_benchmark(self, benchmark_name):
        """Mark a benchmark as started."""
        self.current_benchmark = benchmark_name
        custom_print('', 'empty')
        custom_print(
            f'Starting benchmark: [bold blue]{benchmark_name}[/bold blue]', 'info'
        )
        self.update_display()

    def complete_benchmark(self, benchmark_name):
        """Mark a benchmark as completed."""
        self.completed_benchmarks.add(benchmark_name)
        if self.progress and self.overall_task is not None:
            self.progress.update(self.overall_task, advance=1)
        custom_print(f'Completed benchmark: {benchmark_name}', 'done')
        self.current_benchmark = None
        self.update_display()

    def log(self, message):
        """Add a log message."""
        self.logs.append(
            f'[dim bright_black]{len(self.logs) + 1:03d}[/dim bright_black] {message}'
        )
        self.update_display()

    def __enter__(self):
        """Enter context manager."""
        layout = self.create_layout()
        layout['header'].update(self.create_progress_panel())
        layout['sidebar'].update(self.create_sidebar_panel())
        layout['main'].update(self.create_log_panel())

        self.live = Live(layout, console=self.console, refresh_per_second=4)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.live:
            self.live.stop()


def _get_structure_format(structure_path: pl.Path) -> str | None:
    """Determines the ASE file format from a path."""
    # Handle common formats
    if '.xyz' in structure_path.suffix:
        return 'extxyz'
    if 'CONTCAR' in structure_path.name or 'POSCAR' in structure_path.name:
        return 'vasp'
    # Default format
    return None
