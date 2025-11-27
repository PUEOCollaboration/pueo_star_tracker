"""
Dynamic configuration management system.

This module provides a base class for managing dynamic configuration parameters
that can be updated at runtime and persisted to a file.

Usage in your main application:
    # Initialize config (now with dynamic capabilities)
    cfg = Config('conf/config.ini')

    # Access config values (dynamic values override static ones)
    print(f'solver: {cfg.solver}')
    print(f'lab_best_focus: {cfg.lab_best_focus}')

    # Update dynamic values (automatically saves to dynamic.ini)
    cfg.solver = 'solver2'
    cfg.lab_best_gain = 150
    cfg.autogain_enabled = True

    # The changes are immediately available throughout your application
    print(f'Updated solver: {cfg.solver}')

    # Reload both main config and dynamic config
    cfg.reload()  # This will reload main config and re-apply dynamic overrides

    # Set multiple dynamic parameters
    cfg.set_dynamic(
        solver='new_solver',
        threshold=0.75,
        autogain_enabled=False
    )

    # Get all dynamic parameters
    dynamic_params = cfg.get_all_dynamic_params()
    print(dynamic_params)


"""

import configparser
import os
from typing import Any, Dict, Type, get_type_hints
from contextlib import suppress

class Dynamic:
    """
    A base class to manage dynamic configuration parameters that can be updated
    at runtime and automatically persisted to a file.

    This class handles boolean, string, integer, and float value types.
    """

    # Define dynamic parameters with type hints
    # [LENS_FOCUS_CONSTANTS]
    lab_best_aperture_position: int = 0     # OK
    lab_best_focus: int = 8355              # OK
    lab_best_gain: int = 125                # OK
    lab_best_exposure: int = 100000         # OK
    # autogain_enabled: bool = False          # Cannot change via CLI
    # threshold: float = 0.5                  # Cannot change via CLI

    # [CAMERA]
    autogain_mode: str = 'gain'             # off, gain, both

    # [SOURCES]
    level_filter: int = 9                   # OK

    # GENERAL
    flight_mode: str = 'flight'             # OK
    solver: str = 'solver1'                 # OK
    time_interval: int = 5000000            # OK (cadence)
    # max_processes: int = 1
    # operation_timeout: int = 60
    # current_timeout: int = 200

    #
    # run_autofocus: bool = False             # Would require restart or change of Telemetry class code - it impacts only the INITIAL SERVER STARTUP run autofocus
    # enable_autogain_with_autofocus: bool = False  # NA - no command
    run_autonomous: bool = False            # OK
    # run_telemetry: bool = True              # Would require restart or change of Telemetry class code
    run_chamber: bool = False               # OK
    # run_test: bool = False                  # NA - special test only mode for startup (runs test instead of server)


    def __init__(self, dynamic_config_file: str):
        """
        Initialize the dynamic configuration manager.

        Args:
            dynamic_config_file: Path to the dynamic configuration file
        """
        self._log = None
        # Set initialization flag first
        self._is_initialized = False
        self._dynamic_config_file = dynamic_config_file
        self._dynamic_config_parser = configparser.ConfigParser()

        # Get dynamic parameters from class type hints
        self._dynamic_params = {}
        type_hints = get_type_hints(self.__class__)

        for attr_name in type_hints:
            if not attr_name.startswith('_'):
                # Get current value from instance or class default
                current_value = getattr(self, attr_name, None)
                if current_value is None:
                    current_value = getattr(self.__class__, attr_name)

                self._dynamic_params[attr_name] = {
                    'value': current_value,
                    'type': type_hints[attr_name]
                }

    def _load_dynamic_from_file(self) -> None:
        """Load dynamic parameters from configuration file."""
        if not os.path.exists(self._dynamic_config_file):
            return

        # self._dynamic_config_parser.read(self._dynamic_config_file)
        # Create a temporary parser to read the file without affecting current state
        temp_parser = configparser.ConfigParser()
        temp_parser.read(self._dynamic_config_file)

        # if 'DYNAMIC' in self._dynamic_config_parser:
        if 'DYNAMIC' in temp_parser:
            for param_name in self._dynamic_params:
                if temp_parser.has_option('DYNAMIC', param_name):
                    value_str = temp_parser['DYNAMIC'][param_name]
                    param_type = self._dynamic_params[param_name]['type']

                    try:
                        # Convert string value to appropriate type
                        if param_type == bool:
                            value = temp_parser.getboolean('DYNAMIC', param_name)
                        elif param_type == int:
                            value = int(value_str)
                        elif param_type == float:
                            value = float(value_str)
                        else:
                            value = value_str

                        # Update the dynamic params
                        self._dynamic_params[param_name]['value'] = value
                        # print(f'{param_name:<32} : {value}')
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid dynamic value for {param_name}: {value_str}. Using default.")

    def _save_dynamic_to_file(self) -> None:
        """Save current dynamic parameters to configuration file."""
        # Clear existing sections and create DYNAMIC section
        self._dynamic_config_parser.clear()
        self._dynamic_config_parser.add_section('DYNAMIC')

        # Add all dynamic parameters
        for param_name, param_info in self._dynamic_params.items():
            self._dynamic_config_parser['DYNAMIC'][param_name] = str(param_info['value'])

        # Write to file
        os.makedirs(os.path.dirname(self._dynamic_config_file), exist_ok=True)
        with open(self._dynamic_config_file, 'w') as configfile:
            self._dynamic_config_parser.write(configfile)
        with suppress(AttributeError):
            self._log.debug(f'Updated dynamic {self._dynamic_config_file}')

    def __setattr__(self, name: str, value: Any) -> None:
        """Set dynamic parameter value and automatically save to file."""
        # Handle regular attributes during initialization
        if not hasattr(self, '_is_initialized') or not self._is_initialized or name.startswith('_'):
            super().__setattr__(name, value)
            return

        # Handle dynamic parameters
        if name in self._dynamic_params:
            # Validate and convert value type
            expected_type = self._dynamic_params[name]['type']
            try:
                if expected_type == bool:
                    # Handle various boolean representations
                    if isinstance(value, str):
                        value = value.lower() in ('true', 'yes', '1', 'on')
                    else:
                        value = bool(value)
                else:
                    value = expected_type(value)

                # Update both the attribute and dynamic params
                old_value = self._dynamic_params[name]['value']
                self._dynamic_params[name]['value'] = value
                super().__setattr__(name, value)

                with suppress(AttributeError):
                    self._log.debug(f'Setting dynamic: {name} <- {value} was: {old_value}')
                self._save_dynamic_to_file()  # Auto-save on change
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {name}: {value}. Expected {expected_type.__name__}.")
        else:
            super().__setattr__(name, value)

    def update_from_dynamic(self) -> None:
        """
        Update the configuration object with values from dynamic parameters.

        If dynamic file doesn't exist, create it using current config values.
        """
        if os.path.exists(self._dynamic_config_file):
            # Load existing dynamic values
            print(f'Loading dynamic file: {self._dynamic_config_file}')
            self._load_dynamic_from_file()
            print(f'  Updating config values (from dynamic):')
            # Apply dynamic values to instance attributes
            for param_name, param_info in self._dynamic_params.items():
                # Update Dynamic (no need)
                # Update the instance attribute directly, not through super()
                param_value_curr = getattr(self, param_name)
                param_value = param_info['value']
                if param_value_curr != param_value:
                    print(f'{param_name:>32} <- {param_info["value"]}')
                    # Set directly in instance dict to avoid __setattr__ recursion
                    self.__dict__[param_name] = param_value
                    # super().__setattr__(param_name, param_value)
                    # setattr(self, param_name, param_value)
        else:
            # First time - create dynamic file using current instance values
            print(f'First time - create dynamic file using current instance values: {self._dynamic_config_file}')
            for param_name in self._dynamic_params:
                if hasattr(self, param_name):
                    # Use the current instance value
                    current_value = getattr(self, param_name)
                    self._dynamic_params[param_name]['value'] = current_value

            self._save_dynamic_to_file()

    def get_all_dynamic_params(self) -> Dict[str, Any]:
        """
        Get all dynamic parameters and their current values.

        Returns:
            Dictionary of parameter names and their values
        """
        return {name: info['value'] for name, info in self._dynamic_params.items()}

    def set_dynamic(self, **kwargs) -> None:
        """
        Set multiple dynamic parameters at once.

        Args:
            **kwargs: Parameter names and values to set
        """
        for param_name, value in kwargs.items():
            if param_name in self._dynamic_params:
                setattr(self, param_name, value)
            else:
                print(f"Warning: Unknown dynamic parameter '{param_name}'")

    def reload_dynamic(self) -> None:
        """Reload dynamic parameters from file."""
        if os.path.exists(self._dynamic_config_file):
            self._load_dynamic_from_file()
            self.update_from_dynamic()

    def __str__(self) -> str:
        """Return string representation of dynamic parameters."""
        params_str = "\n".join([f"  {name}: {info['value']} ({info['type'].__name__})"
                                for name, info in self._dynamic_params.items()])
        return f"Dynamic parameters:\n{params_str}"