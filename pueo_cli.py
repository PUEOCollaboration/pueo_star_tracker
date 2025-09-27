#!/usr/bin/env python3
"""
pueo-cli.py - Command line interface for PUEO server communication

Usage:
  python pueo-cli.py <command> [<args>...]

Available commands:
  start                     Start/resume autonomous mode
  stop                      Stop autonomous mode
  home_lens                 Perform focuser homing
  power_cycle               Powercycle and initialize camera and focuser
  auto_focus <start> <end> <steps> Run autofocus routine
  auto_gain                 Run autogain routine
  auto_exposure             Run autoexposure routine
  take_image [type]         Take image (raw, solver1, solver2, solver3 - default from config)
  get_chamber_mode          Get current chamber mode
  set_chamber_mode <mode>   Set chamber mode (true, false)
  get_flight_mode           Get current flight mode
  set_flight_mode <mode>    Set flight mode (preflight, flight)
  get_aperture              Get current aperture value
  set_aperture <value>      Set aperture value
  get_aperture_position     Get current aperture position
  set_aperture_position <value> Set  aperture position (opened, closed)
  get_focus                 Get current focus value
  set_focus <value>         Set focus value
  get_exposure              Get current exposure value
  set_exposure <value>      Set exposure value
  get_gain                  Get current gain value
  set_gain <value>          Set gain value
  get_settings              Get Pueo settings values
  set_level_filter          Get level filter value
  get_level_filter <value>  Set level filter value
"""

import sys
import json
import time
import socket
import logging
import argparse
from typing import Optional, Tuple, Any

# Custom Imports
from lib.commands import Command, Commands
from lib.common import load_config, get_dt, current_timestamp, logit
from lib.config import Config

__program__ = "pueo-cli"
__version__ = "1.0.0"


class PueoSocketClient:
    """
    A client for communicating with the PUEO server via sockets.
    Handles command sending and response receiving with timeout support.

    Attributes:
        server_ip (str): IP address of the PUEO server
        port (int): Port number for communication
        socket_timeout (float): Timeout for socket operations in seconds
        cfg (Config): Configuration object
    """

    def __init__(self, config_file: str = 'conf/config.ini'):
        """
        Initialize the PUEO socket client.

        Args:
            config_file (str): Path to configuration file (default: 'conf/config.ini')
        """
        self.cfg = cfg
        self.server_ip = self.cfg.server_ip
        self.port = self.cfg.port
        self.socket_timeout = self.cfg.socket_timeout
        self.socket = None
        # self.log = logging.getLogger(__name__)
        self.log = logging.getLogger('pueo')
        self.setup_logging()

    def setup_logging(self):
        """Configure logging to display timestamps and messages."""
        # 2. Create a StreamHandler for the console (screen)
        console_handler = logging.StreamHandler(sys.stdout)  # You can also use sys.stderr for error output
        # Set the logging level for the console handler if you want it different from the file handler
        # For example, if you only want INFO and above on console, but DEBUG in file:
        console_handler.setLevel(logging.INFO)

        # 3. Create a formatter for the console output (optional, but good practice)
        # You might want a simpler format for the screen compared to the file.
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.log.addHandler(console_handler)

    def connect(self) -> bool:
        """Establish connection to the server."""
        try:
            self.log.debug(f"Connecting to server at {self.server_ip}:{self.port}")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.socket_timeout)
            self.socket.connect((self.server_ip, self.port))
            self.log.info(f"Connected to server at {self.server_ip}:{self.port}")
            return True
        except socket.timeout:
            self.log.error("Connection timed out")
            return False
        except ConnectionRefusedError:
            self.log.error("Connection refused - is the server running?")
            return False
        except Exception as e:
            self.log.error(f"Connection failed: {str(e)}")
            return False

    def send_command(self, command: dict) -> Tuple[Optional[dict], Optional[float]]:
        """
        Send a JSON command to the server and wait for response.

        Args:
            command (dict): Command to send as dictionary

        Returns:
            tuple: (response_data, response_time_ms) or (None, None) if failed
        """
        try:
            # Prepare and send command
            cmd_str = json.dumps(command, ensure_ascii=False) + '\n'
            start_time = time.perf_counter()
            self.socket.sendall(cmd_str.encode('utf-8'))

            # Wait for response
            response = self.socket.recv(8196)
            response_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            if response:
                # Decode response and parse JSON
                decoded_response = response.decode('utf-8').strip()
                return json.loads(decoded_response), response_time
            return None, response_time

        except (json.JSONDecodeError, UnicodeDecodeError):
            self.log.error("Invalid JSON response from server")
            return None, None
        except socket.timeout:
            self.log.error("Server response timed out")
            return None, None
        except Exception as e:
            self.log.error(f"Command failed: {str(e)}")
            return None, None

    @staticmethod
    def str2bool(v):
        """Converts a string or boolean input to a boolean value.

        This method is useful for parsing boolean values from configuration files,
        CLI arguments, or environment variables where the input might be a string
        (e.g., 'true', 'false') instead of a native boolean.

        Args:
            v (str | bool): Input value to convert. Accepts:
                - Native Python booleans (returned as-is).
                - Case-insensitive strings:
                    - True values: 'yes', 'true', 't', 'y', '1'
                    - False values: 'no', 'false', 'f', 'n', '0'

        Returns:
            bool: Parsed boolean value.

        Raises:
            argparse.ArgumentTypeError: If input is neither a boolean nor a recognized string.

        Examples:
            >>> str2bool(True)   # Returns True
            >>> str2bool('YES')   # Returns True
            >>> str2bool('0')     # Returns False
            >>> str2bool('maybe') # Raises argparse.ArgumentTypeError
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse_cmd(self) -> dict:
        """
        Parse command line arguments and generate appropriate command dictionary.

        Returns:
            dict: Command dictionary to send to server

        Raises:
            ValueError: If invalid command or arguments are provided
        """
        parser = argparse.ArgumentParser(
            add_help=False,
            description='PUEO Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""Examples:
  python pueo-cli.py start raw
  python pueo-cli.py stop
  python pueo-cli.py take_image solver1
  python pueo-cli.py set_focus 150
  python pueo-cli.py get_exposure"""
        )

        parser.add_argument('--full-help', action='store_true', help='Show detailed help for all commands')

        # Custom full help logic
        def show_full_help():
            print("FULL HELP:\n")
            parser.print_help()  # Show main parser help
            print("\n--- COMMAND DETAILS ---")
            for name, subparser in subparsers.choices.items():
                print(f"\nCommand: '{name}'")
                subparser.print_help()

        subparsers = parser.add_subparsers(dest='command', title='Commands')

        # Start command
        start_parser = subparsers.add_parser('start', help='Start/resume autonomous mode')
        start_parser.add_argument('solver', nargs='?', choices=['solver1', 'solver2', 'solver3'], default=self.cfg.solver,
                                      help='Solver value (default: %(default)s)')
        start_parser.add_argument('cadence', nargs='?', type=float, default=float(self.cfg.time_interval/1.e6),
                                      help='Cadence (default: %(default)s)')

        # Custom validation
        def start_validate_args(args):
            if args.cadence is not None and args.solver not in ['solver1', 'solver2', 'solver3']:
                start_parser.error("Cadence requires explicit solver (solver1/solver2/solver3)")

        start_parser.set_defaults(validator=start_validate_args)

        # Stop command
        stop_parser = subparsers.add_parser('stop', help='Stop autonomous mode')

        # Home lens command
        home_lens_parser = subparsers.add_parser('home_lens', help='Perform focuser homing')

        # Home lens command
        check_lens_parser = subparsers.add_parser('check_lens', help='Perform focuser check lens')

        # Power cycle command
        power_parser = subparsers.add_parser('power_cycle', help='Powercycle and initialize camera and focuser')

        # Auto focus command
        autofocus_parser = subparsers.add_parser('auto_focus', help='Run autofocus routine', description='Positional args: [start] [stop] [step] (or none for defaults)')
        autofocus_parser.add_argument('start_position', nargs='?', type=int, default=self.cfg.autofocus_start_position, help='Start position value (default: %(default)s)')
        autofocus_parser.add_argument('stop_position', nargs='?', type=int, default=self.cfg.autofocus_stop_position, help='Stop position value (default: %(default)s)')
        autofocus_parser.add_argument('step_count', nargs='?', type=int, default=self.cfg.autofocus_step_count, help='Step count value (default: %(default)s)')
        autofocus_parser.add_argument('enable_autogain', nargs='?', type=self.str2bool, default=self.cfg.enable_autogain_with_autofocus,help='Enable Autogain (default: %(default)s) (True/False, true/false, 0/1, yes/no)')

        # Custom validation
        def autofocus_validate_args(args):
            # Check which args were EXPLICITLY provided (even if they match defaults)
            provided = [
                hasattr(args, '_start_position'),  # True if user typed --start_position
                hasattr(args, '_stop_position'),  # Same for other args
                hasattr(args, '_step_count'),
                hasattr(args, '_enable_autogain')
            ]

            if any(provided) and not all(provided):
                autofocus_parser.error(
                    "Either provide ALL positional arguments (start_position, stop_position, step_count, enable_autogain) or NONE to use defaults.")

        autofocus_parser.set_defaults(validator=autofocus_validate_args)

        # Auto gain command
        autogain_parser = subparsers.add_parser('auto_gain', help='Run autogain routine')
        autogain_parser.add_argument('desired_max_pixel_value', nargs='?', type=int, default=self.cfg.autogain_desired_max_pixel_value, help='Auto gain desired max pixel value (default: %(default)s)')

        # Auto exposure command
        autoexposure_parser = subparsers.add_parser('auto_exposure', help='Run autoexposure routine')
        autoexposure_parser.add_argument('desired_max_pixel_value', nargs='?', type=int, default=self.cfg.autoexposure_desired_max_pixel_value, help='Auto exposure desired max pixel value (default: %(default)s)')

        # Take image command
        take_image_parser = subparsers.add_parser('take_image', help='Take image (default: solver from config)')
        take_image_parser.add_argument('type', nargs='?', default=self.cfg.solver,
                                      choices=['raw', 'solver1', 'solver2', 'solver3'],
                                      help='Image type (default: %(default)s)')

        # Chamber mode commands
        subparsers.add_parser('get_chamber_mode', help='Get current chamber mode')

        set_chamber_parser = subparsers.add_parser('set_chamber_mode', help='Set chamber mode')
        set_chamber_parser.add_argument('mode', type=self.str2bool,
                                        help='Chamber mode to set (True/False, true/false, 0/1, yes/no)')

        # Flight mode commands
        subparsers.add_parser('get_flight_mode', help='Get current flight mode')

        set_flight_parser = subparsers.add_parser('set_flight_mode', help='Set flight mode')
        set_flight_parser.add_argument('mode', choices=['preflight', 'flight'], help='Flight mode to set')

        # Flight Telemetry
        get_flight_telemetry_parser = subparsers.add_parser('get_flight_telemetry', help='Get flight telemetry data',
                                                            description='Positional args: [limit]')
        get_flight_telemetry_parser.add_argument('limit', nargs='?', type=int, default=0,
                                                 help='Number of last solutions (default: %(default)s)')
        get_flight_telemetry_parser.add_argument('metadata', nargs='?', type=bool, default=False,
                                                 help='Include metadata (default: %(default)s)')

        # PUEO Server Status
        get_status_parser = subparsers.add_parser('get_status', help='Get PUEO Server status')

        # Parameter get commands
        get_list = ['aperture', 'aperture_position', 'focus', 'exposure', 'gain', 'level_filter', 'settings']
        set_list = ['aperture', 'aperture_position', 'focus', 'exposure', 'gain', 'level_filter']
        for param in get_list:
            subparsers.add_parser(f'get_{param}', help=f'Get current {param} value')

        # Parameter set commands
        for param in set_list:
            set_parser = subparsers.add_parser(f'set_{param}', help=f'Set {param} value')
            set_parser.add_argument('value', type=str, help=f'{param.capitalize()} value to set')

        # Custom full help logic
        def show_full_help():
            print("FULL HELP:\n")
            parser.print_help()  # Show main parser help
            print("\n--- COMMAND DETAILS ---")
            for name, subparser in subparsers.choices.items():
                print(f"\nCommand: '{name}'")
                subparser.print_help()

        # Parse args
        args, _ = parser.parse_known_args()
        if args.full_help:
            show_full_help()
            exit()
        else:
            args = parser.parse_args()  # Proceed normally

        if hasattr(args, 'validator'):  # Check if validator exists
            args.validator(args)  # Run validation
        cmd = Command()

        try:
            if args.command == 'start':
                return cmd.resume_operation(args.solver, args.cadence)  # Default solver and cadence
            elif args.command == 'stop':
                return cmd.pause_operation()
            elif args.command == 'home_lens':
                return cmd.home_lens()
            elif args.command == 'check_lens':
                return cmd.check_lens()
            elif args.command == 'power_cycle':
                return cmd.power_cycle()
            elif args.command == 'auto_focus':
                return cmd.run_autofocus(self.cfg.autofocus_method, args.start_position,args.stop_position, args.step_count, enable_autogain=args.enable_autogain)
            elif args.command == 'auto_gain':
                return cmd.run_autogain(args.desired_max_pixel_value)
            elif args.command == 'auto_exposure':
                return cmd.run_autoexposure(args.desired_max_pixel_value)
            elif args.command == 'take_image':
                return cmd.take_image(args.type)
            elif args.command == 'get_chamber_mode':
                return cmd.chamber_mode('get')
            elif args.command == 'set_chamber_mode':
                return cmd.chamber_mode('set', args.mode)
            elif args.command == 'get_flight_mode':
                return cmd.flight_mode('get')
            elif args.command == 'set_flight_mode':
                return cmd.flight_mode('set', args.mode)
            elif args.command == 'get_flight_telemetry':
                return cmd.flight_telemetry(args.limit, args.metadata)
            elif args.command == 'get_status':
                return cmd.check_status()

            elif args.command.startswith('get_'):
                param = args.command[4:]  # Remove 'get_' prefix
                return cmd.get(param)
            elif args.command.startswith('set_'):
                param = args.command[4:]  # Remove 'set_' prefix
                # Handle set_aperture, set_aperture_position, set_exposure, set_focus, set_gain, set_level_filter
                return cmd.set(param, args.value)
            else:
                raise ValueError(f"Unknown command: {args.command}")
        except AttributeError:
            raise ValueError(f"Invalid command or parameter: {args.command}")

    def run(self):
        """Main execution method that parses commands and communicates with server."""
        try:
            cmd_dict = self.parse_cmd()

            if not self.connect():
                self.log.error("Failed to connect to server")
                sys.exit(1)

            self.log.info(f'Sending command: {cmd_dict}')
            response, rtt = self.send_command(cmd_dict)

            if response is not None:
                # Use ensure_ascii=False to properly display Unicode characters
                formatted_response = json.dumps(response, indent=2, ensure_ascii=False)
                self.log.info(f"Response time: {rtt:.1f}ms Response:\n{formatted_response}")
            else:
                self.log.warning("No response received from server")
                sys.exit(1)

        except ValueError as e:
            self.log.error(f"Command error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            self.log.error(f"Unexpected error: {str(e)}")
            sys.exit(1)
        finally:
            if hasattr(self, 'socket') and self.socket:
                self.socket.close()
            self.log.info("Session completed")


def init() -> Config:
    """
    Initializes core settings and configuration for the application.

    Returns:
        Config: Loaded configuration object

    This function handles:
    - Loading the configuration and setting up logging.
    - Initializing global variables.
    """
    program_name = f'{__program__} v{__version__}'
    logit(program_name, attrs=['bold'], color='cyan')

    log = load_config(name='pueo', config_file='config.ini', log_file_name_token="client_log")
    log.debug(f'Main - {program_name}')

    return Config('conf/config.ini')


if __name__ == "__main__":
    cfg = init()
    client = PueoSocketClient(cfg)
    client.run()

# last line