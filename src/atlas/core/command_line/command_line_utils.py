#!/usr/bin/env python
"""Run an active-learning procedure based on ML-MD using aiida."""

import pathlib as pl
import sys
import tomllib
import warnings

import pandas as pd
import yaml

from atlas.core import ATL_DATA_DIR
from atlas.core import exceptions as atl_exc
from atlas.core.code_utils import custom_print

warnings.filterwarnings('ignore', module='paramiko')

WB_FMT = '[bold yellow]'
WB_END = '[/bold yellow]'
WI_FMT = '[yellow italic]'
WI_END = '[/yellow italic]'
E_FMT = '[bold red]'
E_END = '[/bold red]'
D_FMT = '[dim]'


ATL_LOGO = """
    /.\      |''||''| '||          /.\      .|'''|
    // \\        ||     ||         // \\     ||
   //...\\       ||     ||        //...\\    `|'''|,
  //     \\      ||     ||       //     \\    .   ||
.//       \\.   .||.   .||...| .//       \\.  |...|'
"""


def parse_input_toml(toml_dict: dict, type: str):
    """
    Parses and validates the input TOML dictionary based on the specified type.

    Parameters
    ----------
    toml_dict : dict
        The input dictionary parsed from a TOML file.
    type : str
        The type of configuration to validate. Currently supports "active_learning".

    Raises
    ------
    MissingMandatoryParameterError
        If any mandatory keys are missing from the input TOML dictionary.
    """
    if type == 'active_learning':
        mandatory_keys_list = ['active_learning', 'md', 'committee_eval', 'dft']

    elif type == 'generate_database':
        mandatory_keys_list = ['database', 'phase_diagram', 'generation']

    for key in mandatory_keys_list:
        if key not in list(toml_dict.keys()):
            raise atl_exc.MissingMandatoryParameterError(
                f'Input toml file missing mandatory key: {key}.'
            )


def validate_config_file(
    config_type,
    config_dict=None,
    config_path=None,
    allow_deprecated=False,
    run_mode='workflow',
    logger=None,
):
    """
    Validate a TOML configuration file against the schema.

    Parameters
    ----------
        config_path: str
            Path to the TOML configuration file
        config_dict: dict
            The loaded configuration dictionary
        config_type: str
            Type of configuration.
            One of: database_generation, dft, active_learning)
        schema: dict
            The complete configuration schema
        allow_deprecated: bool
            If True, deprecated keys will only generate warnings
        run_mode: str
            Either 'workflow' or 'script'. Workflow is for function runs at
            the start of a mdb workflow, before proceeding. 'script' is for standalone
            verification script execution.

    Returns
    -------
        tuple: (any_errors_found, errors, warnings)
            where any_errors_found is bool, errors is list of strings
            and warnings is list of dicts
    """
    deprecation_warnings = None
    migrated_config = None

    if config_dict is None and config_path is None:
        return True, ['Either config_dict or config_path must be provided.'], []

    if config_path is not None:
        try:
            # Load the TOML file
            config_path = pl.Path(config_path)
            with open(config_path, 'rb') as f:
                config_data = tomllib.load(f)
        except FileNotFoundError:
            warnings = []
            config_data = None
            errors = [f'Configuration file not found: {config_path}']
        except Exception as e:
            warnings = []
            config_data = None
            errors = [f'Error reading TOML file: {e}']
    elif config_dict is not None:
        config_data = config_dict

    # Get the schema for this config type
    schema = get_schema()
    config_schema = schema.get(config_type)
    if not config_schema:
        return False, [f'Unknown configuration type: {config_type}'], []

    custom_print('Validating TOML input file...', print_type='info', logger=logger)

    # Check for deprecated keys and migrate them
    if config_data:
        migrated_config, deprecation_warnings = check_deprecated_keys(
            config_data, config_schema
        )

    # Show deprecation warnings
    if deprecation_warnings:
        for warning in deprecation_warnings:
            custom_print(warning, print_type='warning', logger=logger)

    # Validate the configuration
    if config_schema is not None and migrated_config is not None:
        errors, warnings = validate_section_recursive(
            migrated_config,
            config_schema,
            root_config_data=migrated_config,
            original_schema_dict=config_schema,
        )

    # Add deprecation warnings as validation errors to fail validation
    if deprecation_warnings and not allow_deprecated:
        errors.extend(
            [
                'Configuration contains deprecated keys. '
                'Please update your file to use the new key names.'
            ]
        )

    if len(warnings) > 0:
        custom_print(
            f'TOML input file validation {WB_FMT}warnings{WB_END}:',
            print_type='warning',
            logger=logger,
        )
        for warning in warnings:
            custom_print(f'  • {warning["msg"]}', print_type='warning')
        print()

    if len(errors) == 0:
        custom_print(
            'TOML input file is [bold green]valid[/bold green]!',
            print_type='done',
            logger=logger,
        )
        if len(warnings) > 0:
            custom_print(
                'However, please take into account the reported warnings'
                ' and act accordingly if necessary.',
                print_type='warning',
                logger=logger,
            )

    else:
        custom_print(
            f'TOML input file validation {E_FMT}failed{E_END}:',
            print_type='error',
            logger=logger,
        )
        for error in errors:
            custom_print(
                f'  • {error}',
                print_type='error',
                logger=logger,
            )

        if run_mode == 'workflow':
            custom_print(
                'Process has not started due to validation errors. '
                'All input errors must be fixed before proceeding.',
                print_type='error',
                logger=logger,
            )
    return len(errors) > 0, errors, warnings


def apply_defaults(config_data, warning_msg_list):
    """
    Apply default values to missing parameters in the configuration.

    Args:
        config_data (dict): The configuration data to check
        warning_msg (str): Warning message to display when applying defaults

    Returns
    -------
        dict: Configuration data with defaults applied
    """
    if not warning_msg_list:
        return config_data

    # Filter messages
    default_messages = [warn for warn in warning_msg_list if 'default_applied' in warn]

    for warning in default_messages:
        # Parsing warning messages
        warning_path = warning['default_applied']['path']

        # Separate the path into the 'parents' and the 'target key'
        path_parts = warning_path.split('.')
        parents = path_parts[:-1]
        target_key = path_parts[-1]

        default_value = warning['default_applied']['default_value']
        default_type = warning['default_applied']['type']

        # Traverse down to the specific dictionary that holds the key
        current_level = config_data

        path_valid = True
        for part in parents:
            # If the path is missing, we create a new dict, or handle error
            if part not in current_level:
                # Create the path if it doesn't exist
                current_level[part] = {}

            current_level = current_level[part]

            # Safety check: ensure we haven't hit a non-dict value mid-path
            if not isinstance(current_level, dict):
                print(f'Error: {part} is not a dictionary. Cannot traverse further.')
                path_valid = False
                break

        # Apply the value to the reference of the parent dictionary
        if path_valid:
            # convert string representing type to actual type
            if default_value in ['None', "'None'", '"None"']:
                default_value = None
            elif default_type == 'int':
                default_value = int(default_value)
            elif default_type == 'float':
                default_value = float(default_value)
            elif default_type == 'bool':
                default_value = bool(default_value)
            elif 'list[' in default_type or default_type == 'list':
                if default_value == '[]':
                    default_value = []
                elif isinstance(default_value, str):
                    # Assume comma-separated values for string representation
                    default_value = [
                        item.strip()
                        for item in default_value.split(',')
                        if item.strip()
                    ]
                else:
                    default_value = list(default_value)
            elif default_type == 'dict':
                default_value = dict(default_value)
            elif default_type == 'str':
                default_value = str(default_value)
            else:
                custom_print(
                    f'Warning: Unknown type {default_type} for {warning_path}',
                    print_type='warning',
                )

            current_level[target_key] = default_value

    return config_data


def check_deprecated_keys(config_data, schema_dict, path='', warnings=None):
    """
    Check for deprecated keys in configuration and provide migration suggestions.

    Args:
        config_data (dict): The configuration data to check
        schema_dict (dict): The schema definition
        path (str): Current path in the configuration
        warnings (list): List to accumulate deprecation warnings

    Returns
    -------
        dict: Migrated configuration data with deprecated keys mapped to new ones
        list: List of deprecation warnings
    """
    if warnings is None:
        warnings = []

    if not isinstance(config_data, dict):
        return config_data, warnings

    migrated_data = config_data.copy()

    # Create reverse lookup for deprecated aliases (section level)
    deprecated_section_map = {}
    for section_name, section_schema in schema_dict.items():
        if isinstance(section_schema, dict):
            aliases = section_schema.get('deprecated_aliases', [])
            for alias in aliases:
                deprecated_section_map[alias] = section_name

    # Check for deprecated section names
    for old_key, new_key in deprecated_section_map.items():
        if old_key in migrated_data:
            full_old_path = f'{path}.{old_key}' if path else old_key
            full_new_path = f'{path}.{new_key}' if path else new_key

            warnings.append(
                f"Deprecated section '{full_old_path}' found. "
                f"Please rename to '{full_new_path}'"
            )

            # Migrate the data
            if new_key not in migrated_data:
                migrated_data[new_key] = migrated_data[old_key]
            del migrated_data[old_key]

    # Check for deprecated parameter names within sections
    for section_name, section_data in migrated_data.items():
        if isinstance(section_data, dict) and section_name in schema_dict:
            section_schema = schema_dict[section_name]
            if isinstance(section_schema, dict) and 'type' not in section_schema:
                # This is a section, check its parameters
                section_path = f'{path}.{section_name}' if path else section_name
                migrated_section, section_warnings = check_deprecated_parameters(
                    section_data, section_schema, section_path
                )
                migrated_data[section_name] = migrated_section
                warnings.extend(section_warnings)

                # Recursively check nested sections
                migrated_nested, nested_warnings = check_deprecated_keys(
                    migrated_section, section_schema, section_path
                )
                migrated_data[section_name] = migrated_nested
                warnings.extend(nested_warnings)

    return migrated_data, warnings


def check_deprecated_parameters(config_data, schema_dict, path=''):
    """
    Check for deprecated parameter names within a section.

    Args:
        config_data (dict): The configuration data to check
        schema_dict (dict): The schema definition for this section
        path (str): Current path in the configuration

    Returns
    -------
        dict: Migrated configuration data
        list: List of deprecation warnings
    """
    warnings = []
    migrated_data = config_data.copy()

    # Create reverse lookup for deprecated parameter aliases
    deprecated_param_map = {}
    for param_name, param_schema in schema_dict.items():
        if isinstance(param_schema, dict) and 'type' in param_schema:
            aliases = param_schema.get('deprecated_aliases', [])
            for alias in aliases:
                deprecated_param_map[alias] = param_name

    # Check for deprecated parameter names
    for old_param, new_param in deprecated_param_map.items():
        if old_param in migrated_data:
            full_old_path = f'{path}.{old_param}'
            full_new_path = f'{path}.{new_param}'

            warnings.append(
                f"Deprecated parameter '{full_old_path}' found. "
                f"Please rename to '{full_new_path}'"
            )

            # Migrate the data
            if new_param not in migrated_data:
                migrated_data[new_param] = migrated_data[old_param]
            del migrated_data[old_param]

    return migrated_data, warnings


def should_validate_mandatory(schema_item, root_config_data):
    """
    Determine if a schema item should be validated as mandatory based on dependencies.

    Args:
        schema_item (dict): Schema definition for parameter or section
        root_config_data (dict): Complete configuration data

    Returns
    -------
        bool: True if item should be validated as mandatory
    """
    if not schema_item.get('mandatory', True):
        return False

    depends_on = schema_item.get('depends_on')

    if depends_on:
        return evaluate_dependency(depends_on, root_config_data)

    return True


def evaluate_dependency(depends_on, root_config_data):
    """
    Evaluate whether a dependency condition is met.

    Args:
        depends_on (dict): The dependency specification from schema
        root_config_data (dict): Complete configuration data for resolving paths

    Returns
    -------
        bool: True if dependency condition is met, False otherwise
    """
    if not depends_on:
        return True

    key_path = depends_on.get('key', '')
    expected_value = depends_on.get('value')

    if not key_path:
        return True

    # Navigate to the dependency value in config
    path_parts = key_path.split('.')
    current_data = root_config_data

    try:
        for part in path_parts:
            if isinstance(current_data, dict):
                current_data = current_data.get(part)
                if isinstance(current_data, dict):
                    valid_check = expected_value in current_data
                else:
                    valid_check = expected_value == current_data
            elif not isinstance(current_data, dict):
                valid_check = current_data == expected_value
            else:
                return False

        # Check if the actual value matches the expected value
        return valid_check

    except (KeyError, TypeError, AttributeError):
        # If we can't find the dependency path, assume dependency is not met
        return False


def validate_parameter(value, param_key, param_schema, path, root_config_data=None):
    """
    Validate a single parameter against its schema definition.

    Args:
        value: The value to validate
        param_key (str): The parameter key name
        param_schema (dict): The parameter schema definition
        path (str): The current path in the configuration for error reporting
        root_config_data (dict): Complete configuration data for dependency resolution

    Returns
    -------
        list: List of validation error messages (empty if valid)
    """
    errors = []
    warnings = []

    if path:
        last_left = path.split('.')[-1]
        first_right = param_key.split('.')[0]

        if last_left == first_right:
            full_path = f'{path}.{".".join(param_key.split(".")[1:])}'
        else:
            full_path = f'{path}.{param_key}'
    else:
        full_path = param_key

    # Check if parameter is mandatory but missing
    if value is None:
        # Check if parameter is actually mandatory considering dependencies
        if should_validate_mandatory(param_schema, root_config_data or {}):
            errors.append(f'Missing mandatory parameter: {full_path}')
        else:
            # If not mandatory, check if default can be applied
            if param_schema.get('default') is not None:
                # Only apply default if dependencies are met, otherwise
                # defaults can be used in situations that lead to incompatible
                # configs, e.g., MACE options being passed to SOAP, leading
                # to a crash.
                dependency_met = False
                depends_on = param_schema.get('depends_on')
                if depends_on:
                    dependency_met = evaluate_dependency(depends_on, root_config_data)
                else:
                    dependency_met = True

                if dependency_met:
                    default_val = param_schema.get('default')
                    if isinstance(default_val, str):
                        default_val_print = f"'{default_val}'"
                    else:
                        default_val_print = f'{default_val}'
                    warnings.append(
                        {
                            'msg': f'Using default: {WI_FMT}{full_path}{WI_END} '
                            f'is missing and has been set to default value: '
                            f'{default_val_print}',
                            'default_applied': {
                                'type': param_schema.get('type'),
                                'default_value': default_val,
                                'warn_type': 'default_applied',
                                'path': full_path,
                            },
                        }
                    )

        return errors, warnings

    # Type validation
    expected_type = param_schema.get('type')
    if expected_type:
        if expected_type == 'str' and not isinstance(value, str):
            errors.append(
                f'Parameter {full_path}: expected string, got {type(value).__name__}'
            )
        elif expected_type == 'int' and not isinstance(value, int):
            errors.append(
                f'Parameter {full_path}: expected integer, got {type(value).__name__}'
            )
        elif expected_type == 'float' and not isinstance(value, (int, float)):
            errors.append(
                f'Parameter {full_path}: expected float, got {type(value).__name__}'
            )
        elif expected_type == 'bool' and not isinstance(value, bool):
            errors.append(
                f'Parameter {full_path}: expected boolean, got {type(value).__name__}'
            )
        elif expected_type.startswith('list[') and not isinstance(value, list):
            errors.append(
                f'Parameter {full_path}: expected list, got {type(value).__name__}'
            )
        elif expected_type == 'dict' and not isinstance(value, dict):
            errors.append(
                f'Parameter {full_path}: expected dict, got {type(value).__name__}'
            )

    # Choice validation
    choices = param_schema.get('choices')
    if choices:
        if isinstance(value, list):
            # For lists, validate each element against choices
            for item in value:
                if item not in choices:
                    errors.append(
                        f"Parameter {full_path}: list item '{item}' not in "
                        f'allowed choices {choices}'
                    )
        else:
            # For non-list values, validate the value directly
            if value not in choices:
                errors.append(
                    f"Parameter {full_path}: value '{value}' not in "
                    f'allowed choices {choices}'
                )

    return errors, warnings


def validate_section_recursive(
    config_data,
    schema_dict,
    path='',
    errors=None,
    warnings=None,
    section_mandatory=True,
    root_config_data=None,
    original_schema_dict=None,
):
    """
    Recursively validate a configuration section against its schema.

    Args:
        config_data (dict): The configuration data to validate
        schema_dict (dict): The schema definition for this section
        path (str): Current path in the configuration
        errors (list): List to accumulate validation errors
        section_mandatory (bool): Whether the current section is mandatory
        root_config_data (dict): Complete configuration data for dependency resolution

    Returns
    -------
        list: List of validation error messages
    """
    if errors is None:
        errors = []
    if warnings is None:
        warnings = []

    # Keep a copy of config_data for removal of validated keys
    # from where they will be popped once validated
    config_data_removal = config_data.copy() if isinstance(config_data, dict) else {}

    # Set root_config_data to the current config_data if not provided
    if root_config_data is None:
        root_config_data = config_data

    if config_data is None:
        if section_mandatory and path.split('.')[-1] not in [
            'depends_on',
            'schema_under_dynamic_keys',
        ]:
            errors.append(f'Missing mandatory section: {path}')
        return errors

    # Separate parameters from nested sections
    params = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' in v
    }
    sections = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' not in v
    }

    # Validate parameters
    for param_key, param_schema in params.items():
        # Handle flattened parameters (dotted notation)
        if '.' in param_key:
            # This should be handled by the flattened section logic
            continue

        value = config_data.get(param_key)
        param_errors, param_warnings = validate_parameter(
            value, param_key, param_schema, path, root_config_data
        )

        errors.extend(param_errors)
        warnings.extend(param_warnings)

        if len(param_errors) == 0:
            config_data_removal.pop(param_key, None)

        # Check for mandatory parameters for job submission
        if param_key == 'computer' and value is not None:
            try:
                from aiida import load_profile, orm
                from aiida.schedulers.plugins.sge import SgeScheduler
                from aiida.schedulers.plugins.slurm import SlurmScheduler

                load_profile()
                loaded_comp = orm.load_computer(value)
                scheduler = loaded_comp.get_scheduler()

                # Default arrangement for scheduler options in config
                scheduler_options_to_check = config_data.get('options')

                # Alternative arrangement under metadata section
                if scheduler_options_to_check is None:
                    scheduler_options_to_check = config_data.get('metadata', {}).get(
                        'options'
                    )
                if scheduler_options_to_check is None:
                    scheduler_options_to_check = config_data.get('options_resources')

                mandatory_keys_for_scheduler = []
                if isinstance(scheduler, SgeScheduler):
                    mandatory_keys_for_scheduler = [
                        'resources.parallel_env',
                        'resources.tot_num_mpiprocs',
                    ]

                    # Flatten dict to ease key checking
                    # Nested keys will use '.' as separator
                    [scheduler_options_to_check] = pd.json_normalize(
                        scheduler_options_to_check, sep='.'
                    ).to_dict(orient='records')

                elif isinstance(scheduler, SlurmScheduler):
                    # TODO: add slurm mandatory keys
                    #  'At least two among `num_machines`, `num_mpiprocs_per_machine`
                    # or `tot_num_mpiprocs` must be specified.'
                    pass

                if scheduler_options_to_check is not None:
                    for key in mandatory_keys_for_scheduler:
                        if key not in scheduler_options_to_check:
                            errors.append(
                                f"Missing mandatory scheduler option '{key}' "
                                f"for scheduler '{type(scheduler).__name__}' "
                                f"in computer '{value}' in {path}.{param_key}."
                            )
                else:
                    errors.append(
                        f"Missing 'options' section for scheduler "
                        f"'{type(scheduler).__name__}' "
                        f"in computer '{value}' in {path}.{param_key}."
                    )

            except Exception as e:
                errors.append(
                    f"Failed to load computer '{value}' in '{path}.{param_key}': {e}"
                )

    # Handle flattened sections
    for section_name, section_schema in sections.items():
        if section_schema.get('flatten'):
            # For flattened sections, look for dotted keys in config_data
            schema_content = section_schema.get('schema', {})
            for sub_key, sub_schema in schema_content.items():
                if isinstance(sub_schema, dict) and 'type' in sub_schema:
                    flattened_key = f'{section_name}.{sub_key}'
                    value = config_data.get(flattened_key)
                    param_errors, param_warnings = validate_parameter(
                        value, flattened_key, sub_schema, path, root_config_data
                    )
                    errors.extend(param_errors)
                    warnings.extend(param_warnings)
                    if len(param_errors) == 0:
                        config_data_removal.pop(param_key, None)
            continue

        if section_schema.get('wildcard_entry') and section_schema.get('wildcard_path'):
            # For wildcard entries, traverse the schema at the specified path
            # and add documentation for each parameter found.
            wildcard_path = section_schema['wildcard_path'].split('.')

            orig_wildcard_section = original_schema_dict
            for part in wildcard_path:
                orig_wildcard_section = orig_wildcard_section.get(part, {})
            if not orig_wildcard_section:
                errors.append(
                    f'Wildcard path in {section_name} not found: '
                    f'{WI_FMT}{wildcard_path}{WI_END}'
                )

            for sub_key, sub_details in orig_wildcard_section.items():
                if sub_key in [
                    'description',
                    'mandatory',
                    'default',
                    'wildcard_path',
                    'wildcard_entry',
                    'flatten',
                    'dynamic_keys',
                    'choices',
                    'type',
                    'depends_on',
                ]:
                    continue
                section_data = config_data.get(section_name, {})
                wildcard_key = '.'.join(wildcard_path + [sub_key])
                if isinstance(sub_details, dict) and 'type' not in sub_details:
                    new_path = f'{path}.{section_name}' if path else section_name

                    # Check if this section should be mandatory considering dependencies
                    is_section_mandatory = (
                        should_validate_mandatory(sub_details, root_config_data)
                        and section_mandatory
                    )

                    validate_section_recursive(
                        config_data=section_data,
                        schema_dict=sub_details,
                        path=new_path,
                        errors=errors,
                        warnings=warnings,
                        section_mandatory=is_section_mandatory,
                        root_config_data=root_config_data,
                        original_schema_dict=original_schema_dict,
                    )
                else:
                    is_section_mandatory = sub_details.get('mandatory', True)

                    if not is_section_mandatory:
                        default_value = sub_details.get('default')
                    else:
                        default_value = None

                    value = section_data.get(sub_key, default_value)

                    param_errors, param_warnings = validate_parameter(
                        value=value,
                        param_key=wildcard_key,
                        param_schema=sub_details,
                        path=path,
                        root_config_data=root_config_data,
                    )
                    errors.extend(param_errors)
                    warnings.extend(param_warnings)
                    if len(param_errors) == 0:
                        config_data_removal[section_name].pop(sub_key, None)
            continue

        # Handle dynamic key sections
        if section_schema.get('dynamic_keys'):
            # Look for keys that match the pattern
            section_prefix = f'{section_name}.'
            for config_key, config_value in config_data.items():
                if config_key.startswith(section_prefix) and isinstance(
                    config_value, dict
                ):
                    # This is a dynamic key entry
                    new_path = f'{path}.{config_key}' if path else config_key
                    is_section_mandatory = (
                        section_schema.get('mandatory', True) and section_mandatory
                    )
                    validate_section_recursive(
                        config_data=config_value,
                        schema_dict=section_schema.get('schema_under_dynamic_keys', {}),
                        path=new_path,
                        errors=errors,
                        warnings=warnings,
                        section_mandatory=is_section_mandatory,
                        root_config_data=root_config_data,
                        original_schema_dict=original_schema_dict,
                    )
            continue

        # Regular nested section
        section_data = config_data.get(section_name)
        new_path = f'{path}.{section_name}' if path else section_name

        # Check if this section should be mandatory considering dependencies
        is_section_mandatory = (
            should_validate_mandatory(section_schema, root_config_data)
            and section_mandatory
        )

        validate_section_recursive(
            config_data=section_data,
            schema_dict=section_schema,
            path=new_path,
            errors=errors,
            warnings=warnings,
            section_mandatory=is_section_mandatory,
            root_config_data=root_config_data,
            original_schema_dict=original_schema_dict,
        )

    # Check for unexpected keys in config_data_removal
    unexpected_keys = set(config_data_removal.keys()) - set(params.keys())
    if unexpected_keys and len(set(config_data_removal.keys())) > 0:
        for key in unexpected_keys:
            full_key_path = f'{path}.{key}' if path else key
            if not key_path_in_schema(full_key_path, original_schema_dict):
                errors.append(f'Unexpected key found: {WI_FMT}{full_key_path}{WI_END}')
    return errors, warnings


def key_path_in_schema(full_path, schema_dict):
    """
    Ensure that a given key is not present in the schema dictionary.

    Args:
        full_path (str): The path to check for absence in the schema.
        schema_dict (dict): The schema dictionary to check against.

    Returns
    -------
        bool: False if the key is not present in the schema, True otherwise.
    """
    keys = full_path.split('.')
    current_dict = schema_dict
    for key in keys:
        if key not in current_dict:
            return False
        current_dict = current_dict[key]
    return True


def get_schema():
    # Load the master schema file
    schema_path = pl.Path(ATL_DATA_DIR) / 'config_schema.yaml'
    try:
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
    except FileNotFoundError:
        custom_print(f'Schema file not found at {schema_path}', print_type='error')
        sys.exit(1)
    except yaml.YAMLError as e:
        custom_print(
            f'Failed to parse schema file {schema_path}: {e}', print_type='error'
        )
        sys.exit(1)

    return schema
