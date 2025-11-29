"""
commands.py

This module defines the set of commands that can be sent between the GUI
and the backend application. Each command corresponds to an operation
that the backend can perform.

Author: Milan Stubljar <info@stubljar.com>
Date: 2024-11-03
"""
# Standard imports
from enum import Enum
import datetime
import logging

# Custom imports
from lib.common import convert_to_timestamp


class Commands(Enum):
    """
        Key Available Commands:
        camera_take_image
        resume_operation
        pause_operation
        run_autofocus:
           run_autofocus_sequence_contrast <start_position> <stop_position> <stepcount>
           run_autofocus_sequence_diameter <start_position> <stop_position> <stepcount>
             start_position: int (0..9000)
             stop_position: int (0..9000)
             stepcount: int (2..15)
           run_autofocus_sequence_twostep <position1> <position2> <focus_coefficient>
              focus_coefficient:float (0.0 .. 100.0)

        set_gain <gain>
           gain: int (0 .. 512)
        set_aperture <aperture_position>
           aperture_position: int (0 .. 28)
        set_focus_position <focus_position>
           focus_position: int (0 .. 9000)

        delta_focus_position <focus_delta>
           focus_delta: int (-100 .. 100)

        run_autofocus <desired_max_pixel_value>
        # TODO
           ## Note mix_pixel_value is a TYPO shall be max_pixel_value
           ## mix_pixel_value : int (0 ..16383)

           ##NEXT ARE TO BE CHECKED!!!!
           init_gain_value : int (120..570)        config: best_gain_value
             -> This value should be used only on startup but then previous (last) value used.
           best_exposure_time :     config: best_exposure_time
           desired_max_pix_value    config: desired_max_pix_value
           pixel_saturated_value    config: pixel_saturated_value
           pixel_count_tolerance    config; pixel_count_tolerance

        set_exposure_time <exposure_time>
           exposure_time: int (32 .. 5s~5000000) (1, 2, 4, 1/250, 1/500, 1/1000) microseconds
           100*1000 ~ 100 milliseconds
        enable_distortion_correction
        disable_distortion_correction <distortion_parameter1> <distortion_parameter2>
          fov: float (0..100), distortion_parameter1, distortion_parameter2: float (-1.0 .. 1.0), precision 2 decimals
        input_gyro_rates <omega_x> <omega_y> <omega_z>
          omega_x: float (-5.0 .. 5.0) degree per second
          Flight computer -
        update_time <new_time>
          new_time: timestamp : '2024-10-29 10:12:23.123456'
            """

    CHECK_STATUS = 0

    TAKE_IMAGE = 1
    RESUME_OPERATION = 2
    PAUSE_OPERATION = 3
    RUN_AUTOFOCUS = 4

    SET_GAIN = 5
    SET_APERTURE = 6
    SET_APERTURE_POSITION = 7
    SET_FOCUS_POSITION = 8
    DELTA_FOCUS_POSITION = 9

    RUN_AUTOGAIN = 10
    RUN_AUTOEXPOSURE = 11
    SET_EXPOSURE_TIME = 12
    ENABLE_DISTORTION_CORRECTION = 13
    DISABLE_DISTORTION_CORRECTION = 14
    INPUT_GYRO_RATES = 15
    UPDATE_TIME = 16

    POWER_CYCLE = 17
    POWER_SWITCH = 18
    HOME_LENS = 19
    CHECK_LENS = 20

    SET_LEVEL_FILTER = 21

    SET_AUTOGAIN_MODE = 22

    CHAMBER_MODE = 97
    GET = 98
    FLIGHT_MODE = 99
    FLIGHT_TELEMETRY = 100

    @classmethod
    def from_string(cls, string: str):
        for member in cls:
            if member.name.lower() == string:
                return member
        raise ValueError(f"Invalid Suit string: {string}")


class Command:
    focus_method = None
    start_position = None
    stop_position = None
    step_count = None
    enable_autogain = None
    focus_coefficient = None
    gain = None
    aperture = None
    aperture_position = None
    focus_position = None
    focus_delta = None
    desired_max_pixel_value = None
    exposure_time = None
    fov = None
    distortion_parameter1 = None
    distortion_parameter2 = None
    omega_x = None
    omega_y = None
    omega_z = None
    new_time = None
    limit = 0
    method = None
    mode = None
    param = None
    metadata: bool = None
    device: str = None
    power: bool = None

    settings = {
        Commands.CHECK_STATUS.value: {
            'params': {}
        },
        Commands.TAKE_IMAGE.value: {
            'params': {
                'mode': {
                    'type': 'list',
                    'values': ['raw', 'solver1', 'solver2', 'solver3']
                },
                'focus_position': {
                    'type': 'int',
                    'min': 0,
                    'max': 9000
                },
                'aperture': {
                    'type': 'list',
                    'values': ['open', 'close']
                },
                'aperture_position': {
                    'type': 'inst',
                    'min': 0,
                    'max': 32
                },
                'exposure_time': {
                    'type': 'int',
                    'min': 0,
                    'max': 5000000
                },
            }
        },
        Commands.RESUME_OPERATION.value: {
            'params': {
                'solver': {
                    'type': 'list',
                    'values': ['solver1', 'solver2', "solver3"]
                },
                'cadence': {
                    'type': 'float',
                    'min': 0,
                    'max': 3600 * 24  # like one day
                },
            }
        },
        Commands.PAUSE_OPERATION.value: {
            'params': {}
        },
        Commands.RUN_AUTOFOCUS.value: {
            'params': {
                'focus_method': {
                    'type': 'list',
                    'values': ['sequence_contrast', 'sequence_diameter', 'sequence_twostep']
                },
                'start_position': {
                    'type': 'int',
                    'min': 0,
                    'max': 9000
                },
                'stop_position': {
                    'type': 'int',
                    'min': 0,
                    'max': 9000
                },
                'step_count': {
                    'type': 'int',
                    'min': 0,
                    'max': 9000
                },
                'focus_coefficient': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 100.0
                },
                'enable_autogain': {
                    'type': 'bool'
                }
            }
        },
        Commands.SET_GAIN.value: {
            'params': {
                'gain': {
                    'type': 'int',
                    'min': 0,
                    'max': 570
                }
            }
        },
        Commands.SET_APERTURE.value: {
            'params': {
                'aperture': {
                    'type': 'list',
                    'values': ['closed', 'opened']
                }
            }
        },
        Commands.SET_APERTURE_POSITION.value: {
            'params': {
                'aperture_position': {
                    'type': 'int',
                    'min': 0,
                    'max': 32
                }
            }
        },
        Commands.SET_FOCUS_POSITION.value: {
            'params': {
                'focus_position': {
                    'type': 'int',
                    'min': 0,
                    'max': 9000
                }
            }
        },
        Commands.DELTA_FOCUS_POSITION.value: {
            'params': {
                'focus_delta': {
                    'type': 'int',
                    'min': -100,
                    'max': 100
                }
            }
        },
        Commands.RUN_AUTOGAIN.value: {
            'params': {
                'desired_max_pixel_value': {
                    'type': 'int',
                    'min': 0,
                    'max': 65532
                }
            }
        },
        Commands.RUN_AUTOEXPOSURE.value: {
            'params': {
                'desired_max_pixel_value': {
                    'type': 'int',
                    'min': 0,
                    'max': 65532
                }
            }
        },
        Commands.SET_EXPOSURE_TIME.value: {
            'params': {
                'exposure_time': {
                    'type': 'int',
                    'min': 0,
                    'max': 5000000
                }
            }
        },
        Commands.ENABLE_DISTORTION_CORRECTION.value: {
            'params': {
                # TODO: Check with Windell the FOV param
                'fov': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 100.0
                },
                'distortion_parameter1': {
                    'type': 'float',
                    'min': -1.0,
                    'max': 1.0
                },
                'distortion_parameter2': {
                    'type': 'float',
                    'min': -1.0,
                    'max': 1.0
                }
            }
        },
        Commands.DISABLE_DISTORTION_CORRECTION.value: {
            'params': {}
        },
        Commands.INPUT_GYRO_RATES.value: {
            'params': {
                'omega_x': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0
                },
                'omega_y': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0
                },
                'omega_z': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0
                }
            }
        },
        Commands.UPDATE_TIME.value: {
            'params': {
                'new_time': {
                    'type': 'timestamp',
                }
            }
        },
        Commands.POWER_CYCLE.value: {
            'params': {}
        },
        Commands.POWER_SWITCH.value: {
            'params': {
                'device': {
                    'type': 'list',
                    'values': ['camera', 'focuser']
                },
                'power': {
                    'type': 'list',
                    'values': ['on', 'off'],
                    'default': 'on'
                }
            }
        },
        Commands.HOME_LENS.value: {
            'params': {}
        },
        Commands.CHECK_LENS.value: {
            'params': {}
        },
        Commands.SET_LEVEL_FILTER.value: {
            'params': {
                'level': {
                    'type': 'int',
                    'min': 5,
                    'max': 199,
                    'odd': True,
                    'default': 9
                }
            }
        },
        Commands.SET_AUTOGAIN_MODE.value: {
            'params': {
                'mode': {
                    'type': 'list',
                    'values': ['off', 'gain', 'both'],
                    'default': 'gain'
                },
                'desired_max_pixel_value': {
                    'type': 'int',
                    'min': 0,
                    'max': 65532,
                    'default': 32767
                },
            }
        },
        Commands.CHAMBER_MODE.value: {
            'params': {
                'method': {
                    'type': 'list',
                    'values': ['get', 'set'],
                    'default': 'get'
                },
                'mode': {
                    'type': 'list',
                    'values': [True, False]
                },
            },
        },
        Commands.GET.value: {
            'params': {
                'param': {
                    'type': 'list',
                    'values': ['aperture', 'aperture_position', 'exposure', 'gain', 'focus', 'settings', 'level_filter', 'autogain_mode'],
                },
            },
        },
        Commands.FLIGHT_MODE.value: {
            'params': {
                'method': {
                    'type': 'list',
                    'values': ['get', 'set'],
                    'default': 'get'
                },
                'mode': {
                    'type': 'list',
                    'values': ['flight', 'preflight']
                },
            },
        },
        Commands.FLIGHT_TELEMETRY.value: {
            'params': {
                'limit': {
                    'type': 'int',
                    'min': 0,
                    'max': 100,
                    'default': 0
                },
                'metadata': {
                    'type': 'bool',
                    'default': False
                }
            }
        }
    }

    def __init__(self, command_data=None):
        self.log = logging.getLogger('root')
        self.command_data = command_data
        self.command_name = None
        self.command = None
        self.data = None
        self.define(command_data)

    def define(self, command_data):
        if command_data:
            self.command_data = command_data
            self.command_name = command_data.get('command')
            self.command = Commands.from_string(self.command_name)
            self.data = command_data.get('data')  # get('data')

            self.add_attributes()

    def add_attributes(self):
        """
        Validations are only applicable for commands that have parameters.
        :return:
        :rtype:
        """
        validation = self.settings[self.command.value]
        if self.command == Commands.TAKE_IMAGE:
            self.add_attribute('mode', self.data['mode'], validation['params'])
        elif self.command == Commands.RESUME_OPERATION:
            self.add_attribute('solver', self.data['solver'], validation['params'])
            self.add_attribute('cadence', self.data['cadence'], validation['params'])
        elif self.command == Commands.RUN_AUTOFOCUS:
            self.add_attribute('focus_method', self.data['focus_method'], validation['params'])
            self.add_attribute('start_position', self.data['start_position'], validation['params'])
            self.add_attribute('stop_position', self.data['stop_position'], validation['params'])
            if self.focus_method == 'sequence_twostep':
                self.add_attribute('focus_coefficient', self.data['focus_coefficient'], validation['params'])
            else:
                self.add_attribute('step_count', self.data['step_count'], validation['params'])
            self.add_attribute('enable_autogain', self.data['enable_autogain'], validation['params'])

        elif self.command == Commands.SET_GAIN:
            self.add_attribute('gain', self.data['gain'], validation['params'])
        elif self.command == Commands.SET_APERTURE:
            self.add_attribute('aperture', self.data['aperture'], validation['params'])
        elif self.command == Commands.SET_APERTURE_POSITION:
            self.add_attribute('aperture_position', self.data['aperture_position'], validation['params'])
        elif self.command == Commands.SET_FOCUS_POSITION:
            self.add_attribute('focus_position', self.data['focus_position'], validation['params'])

        elif self.command == Commands.DELTA_FOCUS_POSITION:
            self.add_attribute('focus_delta', self.data['focus_delta'], validation['params'])

        elif self.command == Commands.RUN_AUTOGAIN:
            self.add_attribute('desired_max_pixel_value', self.data['desired_max_pixel_value'], validation['params'])
        elif self.command == Commands.RUN_AUTOEXPOSURE:
            self.add_attribute('desired_max_pixel_value', self.data['desired_max_pixel_value'], validation['params'])

        elif self.command == Commands.SET_EXPOSURE_TIME:
            self.add_attribute('exposure_time', self.data['exposure_time'], validation['params'])
        elif self.command == Commands.ENABLE_DISTORTION_CORRECTION:
            self.add_attribute('fov', self.data['fov'], validation['params'])
            self.add_attribute('distortion_parameter1', self.data['distortion_parameter1'], validation['params'])
            self.add_attribute('distortion_parameter2', self.data['distortion_parameter2'], validation['params'])
        elif self.command == Commands.INPUT_GYRO_RATES:
            self.add_attribute('omega_x', self.data['omega_x'], validation['params'])
            self.add_attribute('omega_y', self.data['omega_y'], validation['params'])
            self.add_attribute('omega_z', self.data['omega_z'], validation['params'])
        elif self.command == Commands.UPDATE_TIME:
            self.add_attribute('new_time', self.data['new_time'], validation['params'])
        elif self.command == Commands.POWER_SWITCH:
            self.add_attribute('device', self.data['device'], validation['params'])
            if 'power' not in self.data:
                self.data['power'] = validation['params']['power']['default']
            self.add_attribute('power', self.data['power'], validation['params'])

        # CHAMBER_MODE ~ id 97
        elif self.command == Commands.CHAMBER_MODE:
            if self.data is None:
                self.data = {}
            if 'method' not in self.data:
                self.data['method'] = validation['params']['method']['default']
            self.add_attribute('method', self.data['method'], validation['params'])
            # The chamber_mode is applicable for the set method
            if self.method == 'set':
                self.add_attribute('mode', self.data['mode'], validation['params'])
        elif self.command == Commands.SET_LEVEL_FILTER:
            self.add_attribute('level', self.data['level'], validation['params'])
        elif self.command == Commands.SET_AUTOGAIN_MODE:
            self.add_attribute('mode', self.data['mode'], validation['params'])
            if 'desired_max_pixel_value' not in self.data:
                self.data['desired_max_pixel_value'] = validation['params']['desired_max_pixel_value']['default']
            self.add_attribute('desired_max_pixel_value', self.data['desired_max_pixel_value'], validation['params'])
        # ...
        elif self.command == Commands.GET:
            self.add_attribute('param', self.data['param'], validation['params'])
        # FLIGHT_MODE ~ id 99
        elif self.command == Commands.FLIGHT_MODE:
            if self.data is None:
                self.data = {}
            if 'method' not in self.data:
                self.data['method'] = validation['params']['method']['default']
            self.add_attribute('method', self.data['method'], validation['params'])
            # The flight_mode is applicable for the set method
            if self.method == 'set':
                self.add_attribute('mode', self.data['mode'], validation['params'])

        # FLIGHT_TELEMETRY ~ id 100
        elif self.command == Commands.FLIGHT_TELEMETRY:
            if self.data is None:
                self.data = {}
            if 'limit' not in self.data:
                self.data['limit'] = validation['params']['limit']['default']
            self.add_attribute('limit', self.data['limit'], validation['params'])
            if 'metadata' not in self.data:
                self.data['metadata'] = validation['params']['metadata']['default']
            self.add_attribute('metadata', self.data['metadata'], validation['params'])


    def add_attribute_old(self, name, value, validation=None):
        if validation is not None:
            rules = validation[name]
            var_type = rules['type']
            if var_type == 'list' and value not in rules['values']:
                raise ValueError(f'Invalid value: name: {name}: {value} allowed values: {rules["values"]}')
            elif var_type == 'int' and (int(value) < rules['min'] or rules['max'] < int(value) or
                                        rules.get['odd', False] and int(value)%2==0):
                raise ValueError(f'Invalid value: name: {name}: {value} range: {rules["min"]} .. {rules["max"]}')
            elif var_type == 'float' and (float(value) < rules['min'] or rules['max'] < float(value)):
                raise ValueError(f'Invalid value: name: {name}: {value} range: {rules["min"]} .. {rules["max"]}')
            elif var_type == 'timestamp':
                value = convert_to_timestamp(value)

            setattr(self, name, value)
            # if var_type == 'int':
            #     self.__annotations__[name] = int
            # if var_type == 'float':
            #     self.__annotations__[name] = float

    def add_attribute(self, name, value, validation=None):
        """
        Dynamically adds an attribute to the object with optional validation.

        Args:
            name (str): The name of the attribute to add.
            value (Any): The value to assign to the attribute.
            validation (dict, optional): A dictionary containing validation rules.
                Expected structure:
                {
                    "name": {
                        "type": "int" | "float" | "list" | "timestamp",
                        "min": int | float,  # Optional (for int/float)
                        "max": int | float,  # Optional (for int/float)
                        "odd": bool,         # Optional (for int, requires odd values)
                        "values": list,     # Required (for list, allowed values)
                    }
                }

        Raises:
            ValueError: If validation fails (invalid value, out of range, etc.).
            KeyError: If the validation rule for `name` is missing required keys.

        Example:
            >>> obj = MyClass()
            >>> validation_rules = {"age": {"type": "int", "min": 0, "max": 120}}
            >>> obj.add_attribute("age", 30, validation_rules)  # Valid
            >>> obj.add_attribute("age", 150, validation_rules) # Raises ValueError
        """
        if validation is None:
            setattr(self, name, value)
            return

        # Get validation rules for the attribute
        rules = validation.get(name)
        if not rules:
            raise KeyError(f"Validation rule for '{name}' not found.")

        var_type = rules.get('type')
        if var_type is None:
            raise KeyError(f"Missing 'type' in validation rules for '{name}'.")

        try:
            # --- List Validation ---
            if var_type == 'list':
                allowed_values = rules.get('values', [])
                if value not in allowed_values:
                    raise ValueError(
                        f"Invalid value for '{name}': {value}. Allowed values: {allowed_values}"
                    )

            # --- Integer Validation ---
            elif var_type == 'int':
                value_int = int(value)
                min_val = rules.get('min', float('-inf'))  # Default: no lower bound
                max_val = rules.get('max', float('inf'))  # Default: no upper bound

                if value_int < min_val or value_int > max_val:
                    raise ValueError(
                        f"Value for '{name}' ({value_int}) out of range. "
                        f"Allowed: {min_val}..{max_val}"
                    )
                if rules.get('odd', False) and value_int % 2 == 0:
                    raise ValueError(f"Value for '{name}' must be odd (got {value_int}).")

            # --- Float Validation ---
            elif var_type == 'float':
                value_float = float(value)
                min_val = rules.get('min', float('-inf'))
                max_val = rules.get('max', float('inf'))

                if value_float < min_val or value_float > max_val:
                    raise ValueError(
                        f"Value for '{name}' ({value_float}) out of range. "
                        f"Allowed: {min_val}..{max_val}"
                    )

            # --- Timestamp Conversion ---
            elif var_type == 'timestamp':
                value = convert_to_timestamp(value)  # Assumes this function exists

            # Set the attribute if validation passes
            setattr(self, name, value)

        except (ValueError, TypeError) as e:
            raise ValueError(f"Validation failed for '{name}': {str(e)}")

    def __setattr__(self, name, value):
        if name in self.__dict__:
            super().__setattr__(name, value)
        else:
            # print(f"Adding new attribute: {name}")
            super().__setattr__(name, value)

    def check_status(self):
        command_data = {'command': Commands.CHECK_STATUS.name.lower(), 'data': {}}
        self.define(command_data)
        return self.command_data

    def take_image(self, mode: str):
        command_data = {
            'command': Commands.TAKE_IMAGE.name.lower(),
            'data': {
                'mode': mode
                # TODO: Include other
            }
        }
        self.define(command_data)
        return self.command_data

    def resume_operation(self, solver: str, cadence: float):
        """cadence: float in seconds"""
        command_data = {
            'command': Commands.RESUME_OPERATION.name.lower(),
            'data': {
                'solver': solver,
                'cadence': cadence
            }
        }
        self.define(command_data)
        return self.command_data

    def pause_operation(self):
        command_data = {'command': Commands.PAUSE_OPERATION.name.lower(), 'data': {}}
        self.define(command_data)
        return self.command_data

    def run_autofocus(self, focus_method: str, start_position: int, stop_position: int, step_count: int = 5,
                      focus_coefficient: float = 1.0, enable_autogain: bool = True):
        command_data = {
            'command': Commands.RUN_AUTOFOCUS.name.lower(),
            'data': {
                'focus_method': focus_method,
                'start_position': start_position,
                'stop_position': stop_position,
                'enable_autogain': enable_autogain
            }
        }
        if focus_method == 'sequence_twostep':
            command_data['data']['focus_coefficient'] = focus_coefficient
        else:
            command_data['data']['step_count'] = step_count
        self.define(command_data)
        return self.command_data

    def set_gain(self, gain: int):
        command_data = {'command': Commands.SET_GAIN.name.lower(), 'data': {'gain': gain}}
        self.define(command_data)
        return self.command_data

    def set_aperture(self, aperture: int):
        command_data = {'command': Commands.SET_APERTURE.name.lower(), 'data': {'aperture': aperture}}
        self.define(command_data)
        return self.command_data

    def set_aperture_position(self, aperture_position: int):
        command_data = {'command': Commands.SET_APERTURE_POSITION.name.lower(),
                        'data': {'aperture_position': aperture_position}}
        self.define(command_data)
        return self.command_data

    def set_focus_position(self, focus_position: int):
        command_data = {'command': Commands.SET_FOCUS_POSITION.name.lower(), 'data': {'focus_position': focus_position}}
        self.define(command_data)
        return self.command_data

    def delta_focus_position(self, focus_delta: int):
        command_data = {'command': Commands.DELTA_FOCUS_POSITION.name.lower(), 'data': {'focus_delta': focus_delta}}
        self.define(command_data)
        return self.command_data

    def run_autogain(self, max_pixel_value: int):
        command_data = {'command': Commands.RUN_AUTOGAIN.name.lower(),
                        'data': {'desired_max_pixel_value': max_pixel_value}}
        self.define(command_data)
        return self.command_data

    def run_autoexposure(self, max_pixel_value: int):
        command_data = {'command': Commands.RUN_AUTOEXPOSURE.name.lower(),
                        'data': {'desired_max_pixel_value': max_pixel_value}}
        self.define(command_data)
        return self.command_data

    def set_exposure_time(self, exposure_time: int):
        command_data = {'command': Commands.SET_EXPOSURE_TIME.name.lower(), 'data': {'exposure_time': exposure_time}}
        self.define(command_data)
        return self.command_data

    def set(self, command, value, *extra_values):
        """
        Dispatch 'set' commands.

        Minimal helper wrapper. `command` and `value` are mandatory.
        Optional additional positional values may be supplied and will be
        forwarded only where explicitly used here (currently only for
        'autogain_mode').

        Missing extra values are intentionally NOT handled here (caller/upper
        layer is responsible for providing/validating them).
        """
        if command in ['aperture', 'set_aperture']:
            return self.set_aperture(value)
        if command in ['aperture_position', 'set_aperture_position']:
            return self.set_aperture_position(value)
        elif command in ['exposure', 'set_exposure_time']:
            return self.set_exposure_time(value)
        elif command in ['focus', 'set_focus_position']:
            return self.set_focus_position(value)
        elif command in ['gain', 'set_gain']:
            return self.set_gain(value)
        elif command in ['level_filter', 'set_level_filter']:
            return self.set_level_filter(value)
        elif command in ['focuser_power', 'set_focuser_power']:
            return self.power_switch('focuser', value)
        elif command in ['camera_power', 'set_camera_power']:
            return self.power_switch('camera', value)
        elif command in ['autogain_mode', 'set_autogain_mode']:
            # NOTE: if caller did not supply an extra value, this will raise IndexError,
            # which is acceptable per your instruction ("missing extra VALUE is OK to fail").
            return self.set_autogain_mode(value, extra_values[0])  # CHANGED

        raise ValueError('Invalid command.')

    def enable_distortion_correction(self, fov: float, distortion_parameter1: float, distortion_parameter2: float):
        command_data = {'command': Commands.ENABLE_DISTORTION_CORRECTION.name.lower(),
                        'data': {'fov': fov, 'distortion_parameter1': distortion_parameter1,
                                 'distortion_parameter2': distortion_parameter2}}
        self.define(command_data)
        return self.command_data

    def disable_distortion_correction(self):
        command_data = {'command': Commands.DISABLE_DISTORTION_CORRECTION.name.lower(), 'data': {}}
        self.define(command_data)
        return self.command_data

    def input_gyro_rates(self, omega_x: float, omega_y: float, omega_z: float):
        command_data = {'command': Commands.INPUT_GYRO_RATES.name.lower(),
                        'data': {'omega_x': omega_x,
                                 'omega_y': omega_y,
                                 'omega_z': omega_z}}
        self.define(command_data)
        return self.command_data

    def update_time(self, new_time: datetime):
        new_time_ts = convert_to_timestamp(new_time)
        command_data = {'command': Commands.UPDATE_TIME.name.lower(), 'data': {'new_time': new_time_ts}}
        self.define(command_data)
        return self.command_data

    def power_cycle(self):
        command_data = {'command': Commands.POWER_CYCLE.name.lower(), 'data': {}}
        self.define(command_data)
        return self.command_data

    def power_switch(self, device: str, power: bool = True):
        command_data = {'command': Commands.POWER_SWITCH.name.lower(), 'data': {'device': device, 'power': power}}
        self.define(command_data)
        return self.command_data

    def home_lens(self):
        command_data = {'command': Commands.HOME_LENS.name.lower(), 'data': {}}
        self.define(command_data)
        return self.command_data

    def check_lens(self):
        command_data = {'command': Commands.CHECK_LENS.name.lower(), 'data': {}}
        self.define(command_data)
        return self.command_data

    def set_level_filter(self, level: int):
        command_data = {'command': Commands.SET_LEVEL_FILTER.name.lower(), 'data': {'level': level}}
        self.define(command_data)
        return self.command_data

    def set_autogain_mode(self, mode: str, desired_max_pixel_value: int):
        command_data = {'command': Commands.SET_AUTOGAIN_MODE.name.lower(), 'data': {'mode': mode, 'desired_max_pixel_value': desired_max_pixel_value}}
        self.define(command_data)
        return self.command_data

    def chamber_mode(self, method: str, mode: bool = None):
        command_data = {'command': Commands.CHAMBER_MODE.name.lower(), 'data': {'method': method}}
        if method == 'set':
            command_data = {'command': Commands.CHAMBER_MODE.name.lower(), 'data': {'method': method, 'mode': mode}}
        self.define(command_data)
        return self.command_data

    def get(self, param: str):
        command_data = {'command': Commands.GET.name.lower(), 'data': {'param': param}}
        self.define(command_data)
        return self.command_data

    def flight_mode(self, method: str, mode: str = None):
        command_data = {'command': Commands.FLIGHT_MODE.name.lower(), 'data': {'method': method}}
        if method == 'set':
            command_data = {'command': Commands.FLIGHT_MODE.name.lower(), 'data': {'method': method, 'mode': mode}}
        self.define(command_data)
        return self.command_data

    def flight_telemetry(self, limit: int = 0, metadata: bool = False):
        command_data = {'command': Commands.FLIGHT_TELEMETRY.name.lower(), 'data': {'limit': limit, 'metadata': metadata}}
        self.define(command_data)
        return self.command_data


if __name__ == "__main__":
    print(Commands.TAKE_IMAGE.value)
    print(Commands.TAKE_IMAGE)

    cmd = Command()
    command_data1 = cmd.run_autofocus('sequence_twostep', 300, 400, focus_coefficient=25.2, enable_autogain=True)
    command_data1_raw = {
        'command': 'run_autofocus',
        'data': {
            'focus_method': 'sequence_diameter',
            'start_position': 300,
            'stop_position': 400,
            'step_count': 5,
            'enable_autogain': False
        }
    }

    cmds = [
        command_data1, command_data1_raw,
        cmd.resume_operation('solver1', 10),  # start/resume autonomous mode: solver1|solver2, cadence in seconds
        cmd.pause_operation(),  # stop autonomous mode

        cmd.update_time(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')),
        {
            'command': 'update_time',
            'data': {
                'new_time': '2024-10-29 10:12:23.123456'
            }
        },
        cmd.flight_telemetry(),
        {
            'command': 'flight_telemetry',
        },
        cmd.flight_mode('get'),
        cmd.flight_mode('set', 'preflight'),
        cmd.take_image('solver1'),  # 'raw', 'solver1', 'solver2
        cmd.get('focus'),
        cmd.get('aperture_position'),
        cmd.chamber_mode('set', True),
        cmd.chamber_mode('get'),
        cmd.set('level_filter', 5),
        cmd.set('camera_power', 'on'),
        cmd.set('focuser_power', 'off'),
        cmd.set('autogain_mode', 'both', 1000)
        # cmd.get('level_filter'),
    ]
    for cmd_dict in cmds:
        cmd = Command(cmd_dict)
        print(cmd_dict)
        if cmd.command == Commands.RUN_AUTOFOCUS:
            print(cmd.focus_method)
            print(cmd.start_position)
            print(cmd.stop_position)
            if cmd.focus_method == 'sequence_twostep':
                print(cmd.focus_coefficient)
            else:
                print(cmd.step_count)
            print(cmd.enable_autogain)
        elif cmd.command == Commands.UPDATE_TIME:
            print(cmd.new_time)
        elif cmd.command == Commands.FLIGHT_TELEMETRY:
            print(f'{cmd.limit}, {cmd.metadata}, {cmd.command_name}, {cmd.data}')

        elif cmd.command == Commands.FLIGHT_MODE:
            print(f'{cmd.method}, {cmd.command_name}, {cmd.data}')
        elif cmd.command == Commands.POWER_SWITCH:
            print(f'{cmd.device}, {cmd.power}, {cmd.data}')
        elif cmd.command == Commands.SET_AUTOGAIN_MODE:
            print(f'{cmd.mode}')
        else:
            print(f'{cmd.command} -> {cmd_dict} ... {cmd.param}')

        # cmd.add_attribute('exposure_time', 1)
        # print(cmd.exposure_time)
