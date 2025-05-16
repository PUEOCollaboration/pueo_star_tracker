"""
File: star_comm_bridge.py
Description: This module implements the StarCommBridge class, enabling socket-based communication
             between the Star Tracker GUI and Pueo Star Operation App. This class supports both client
             and server modes to send/receive JSON-encoded commands and responses.

Classes:
    - StarCommBridge: Main class for managing socket communication.
    - Status: Defines error codes and methods for standard status_code messages.

Usage:
    StarCommBridge can be initialized in client or server mode. The client periodically sends
    status_code check commands, while the server responds with the current status_code.

Author: Milan Stubljar <info@stubljar.com>
Created: 2024-11-03
Version: 1.0

Dependencies:
    - socket, json, threading, logging, time
    - commands.py (Commands enum for supported commands)
    - status_code.py (Status class for response codes and messages)

Configuration:
    Uses settings from a config object, including
    - client_log and server_log (paths for log files)
    - pueo_server_ip and port (server network configuration)
    - server_ip and port (for client access, network configuration)
    - max_retry, retry_delay (connection settings)

Notes:
    - Ensure both `commands.py` and `status_code.py` are present in the same directory.
    - Edit config.ini to adjust socket and logging settings as needed.

"""
# Standard Imports
import socket
import json
from threading import Thread, Event
import logging
import time
from contextlib import suppress
import os
import shutil
import random
from types import SimpleNamespace
from queue import Queue
import errno
import traceback
from datetime import datetime

# Custom Imports
from lib.commands import Command, Commands
from lib.common import get_dt, current_timestamp, logit
from lib.status import Status
from lib.messages import MessageHandler


class StarCommBridge:
    """
    Class to handle socket-based communication between Star Tracker GUI and Pueo Star Operation App.

    This class supports both client and server modes to enable bidirectional communication.
    Commands are sent as JSON strings, and responses include a JSON-encoded status_code.
    """

    def __init__(self, is_client: bool, config, camera=None):
        """
        Initialize the StarCommBridge.

        Args:
            is_client (bool): Specifies if this instance is acting as the client or server.
            config: Configuration object to access settings, such as server IP, port, and log file paths.
        """
        self.is_client = is_client
        self.config = config
        self.camera = camera
        self.max_retry = config.max_retry or 5
        self.pueo_server_ip = config.pueo_server_ip
        self.server_ip = config.server_ip
        self.port = config.port
        self.retry_delay = config.retry_delay or 2

        self.log = logging.getLogger('pueo')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.is_connecting = False
        self.is_processing = False
        self.command =  None    # Command
        self.command_t0 = time.monotonic()

        role = 'client' if self.is_client else 'server'
        self.log.debug(f'StarCommBridge initialized as {role}.')

        self.messages = MessageHandler(msg_max_size=self.config.msg_max_size)
        self.server = None # link to the PueoStarCameraOperation instance
        self.is_running = False
        self.client_threads = []  # Store client threads for cleanup
        self.shutdown_event = Event()

    def connect(self):
        """Attempt to connect to the server as a client, with retry logic."""
        if self.is_connecting:
            self.log.warning('Already connecting.')
            return

        # Update GUI
        self.camera.set_camera_status(self.camera.INIT, 'Initializing.')

        self.is_connecting = True
        for attempt in range(1, self.max_retry + 1):
            try:
                if self.connected:
                    with suppress(Exception):
                        self.log.debug("self.socket.shutdown(socket.SHUT_RDWR)")
                        self.socket.shutdown(socket.SHUT_RDWR)
                        self.log.debug("self.socket.close()")
                        self.socket.close()
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                self.socket.connect((self.server_ip, self.port)) # Connecting as client to the server
                self.connected = True
                self.log.info("Successfully connected to server on attempt %d", attempt)
                self.camera.set_camera_status_text(f"Connected.")
                break
            except socket.error as e:
                self.log.error("Connection attempt %d failed: %s", attempt, e)
                self.camera.set_camera_status_text(f"Retry #{attempt}. Error: {e.errno}")
                time.sleep(self.retry_delay)
        else:
            self.log.critical("Failed to connect after %d attempts", self.max_retry)

        self.is_connecting = False

    def send(self, command: str, data: dict = None, is_check=False):
        """
        Send a command and optional data to the server or client.

        Args:
            command (str): Command to send, as defined in Commands.
            data (dict, optional): Additional data for the command.
            is_check (bool, optional): Send type, normal/check.
        Returns:
            dict: The server response in JSON format.
        """
        if not self.connected:
            self.log.error("Not connected to server, unable to send command")
            return Status.get_status(Status.ERROR, "Not connected")

        payload = None
        if isinstance(command, str):
            payload = json.dumps({"command": command, "data": data})
        elif isinstance(command, dict) and 'command' in command:
            payload = json.dumps(command)
        if payload is None:
            return
        try:
            self.socket.sendall(payload.encode('utf-8'))
            self.log.debug("Sent command: %s", payload)
            if is_check:
                res = self._receive()
                response_dict = None
                try:
                    response_dict = json.loads(res)
                except Exception as e:
                    self.log.error(e)
                    self.log.debug(res)
                return response_dict
            else:
                return Status.get_status(Status.SUCCESS, f"Command sent: {command['command']}")
        except socket.error as e:
            self.log.error("Error sending command '%s': %s", command, e)
            # TODO: Cleanup
            # [WinError 10053] An established connection was aborted by the software in your host computer.
            # [WinError 10054] An existing connection was forcibly closed by the remote host
            # [WinError 10057] A request to send or receive data was disallowed because the socket is not connected and (when sending on a datagram socket using a sendto call) no address was supplied
            # [Unix        32] A broken pipe error occurred,
            if e.errno in  (10053, 10054, 10057, 32):
                time.sleep(1)
                self.connect()
            return Status.get_status(Status.ERROR, f"Send failed: {e}")

    def _receive_old(self):
        """
        Receive a response from the other end of the socket.

        Returns:
            str: The received response in JSON format.
        """
        try:
            res = self.socket.recv(4096).decode('utf-8')
            self.log.debug("Received response: %s", res)
            return res
        except socket.error as e:
            self.log.error("Error receiving data: %s", e)
            return json.dumps(Status.get_status(Status.ERROR, "Receive failed"))

    def _receive(self):
        """
        Receive a complete response from the other end of the socket.

        The function reads the data stream until all data has been received,
        indicated by the socket closing the connection or a pre-determined
        end-of-message marker.

        Returns:
            str: The received response in JSON format.
        """
        try:
            buffer = b""  # Initialize a buffer to store the received bytes
            chunk_size = 16384
            self.log.debug(f'Receiving message.')
            t0 = time.monotonic()
            chunk_cnt = 0
            while True:
                chunk = self.socket.recv(chunk_size)  # Read up to 4096 bytes at a time
                chunk_cnt += 1
                if not chunk:  # No more data, connection closed by peer
                    break
                buffer += chunk
                if len(chunk) < chunk_size:
                    break

            res = buffer.decode('utf-8')  # Decode the complete message
            if len(res) <= chunk_size:
                self.log.debug(f"Received complete response: {res}")
            else:
                self.log.debug(f"Received complete response: {str(res)[:512]}...")
            self.log.debug(f'Received in {get_dt(t0)} chunks: {chunk_cnt}  chunk_size: {chunk_size}')
            return res
        except socket.error as e:
            self.log.error("Error receiving data: %s", e)
            return json.dumps(Status.get_status(Status.ERROR, "Receive failed"))

    def handle_client(self, client_socket):
        """Handle incoming client commands as a server."""

        def process_command():
            cmd_json = str(command_json).strip()
            # Remove leading/trailing quotes if any
            if cmd_json and (cmd_json.startswith("'") or cmd_json.startswith('"')):
                cmd_json = cmd_json[1:-1]
            self.log.debug(f"Received command from client: {cmd_json} mode: {self.server.flight_mode} queues: msg: {self.messages.message_queue.qsize()} telemetry: {self.server.telemetry_queue.qsize()} positions: {self.server.positions_queue.qsize()}")
            # logit(f"Received command from client: {cmd_json}", color='magenta')
            try:
                command_data = json.loads(cmd_json)
            except json.decoder.JSONDecodeError as e:
                return

            if command_data.get('disconnect', False):
                client_socket.close()
                self.connected = False
                self.log.debug('Disconnecting.')
            res = self._process_command(command_data)
            if res is not None:
                with suppress(Exception):
                    client_socket.sendall(json.dumps(res).encode('utf-8'))

        try:
            while self.is_running:
                try:
                    command_json = client_socket.recv(4096).decode('utf-8')
                    if not command_json:  # Empty string means client disconnected
                        self.log.debug('Client closed connection')
                        break

                    _thread = Thread(target=process_command, daemon=True)
                    _thread.start()

                except ConnectionResetError:
                    self.log.debug('Client connection reset')
                    break
                except socket.timeout:
                    # Handle timeout if you've set a timeout on the socket
                    continue
                except UnicodeDecodeError:
                    self.log.error('Received malformed data from client')
                    continue

        except socket.error as e:
            self.log.error("Error in client handler: %s", e)
        finally:
            client_socket.close()  # Close socket after disconnection
            self.connected = False
            self.log.warning('Client disconnected.')
        self.log.warning('Exiting handle_client.')

    def _process_command(self, command_data):
        """
        Process a command and generate a response.

        Args:
            command_data (dict): Parsed command data.

        Returns:
            dict: Response status_code as JSON-encoded dictionary.
        """
        cmd = Command(command_data)
        commands = cmd.command

        # Remember what command has been started
        if commands == Commands.CHECK_STATUS:
            # TODO: Remove time.sleep()
            # time.sleep(1)
            message = 'Idle'
            if self.is_processing:
                message = f'Busy: {self.command.command_name} [{get_dt(self.command_t0)}]'
            elif not self.server.is_running:
                message = f'Server not running.'
            # self.log.debug(f'Status check response: {message}')
            data = {'mode': self.server.flight_mode}
            res = Status.get_status(Status.SUCCESS, message, data)
            res['messages'] = self.messages.get_messages()
            return res

        # Remember what command has been started
        if commands == Commands.FLIGHT_TELEMETRY:
            # TODO: Remove time.sleep()
            # time.sleep(1)
            message = 'Idle'
            if self.is_processing:
                message = f'Busy: {self.command.command_name} [{get_dt(self.command_t0)}]'
            elif not self.server.is_running:
                message = f'Server not running.'
            # self.log.debug(f'Status check response: {message}')
            # res = Status.get_status(Status.SUCCESS, message)
            limit = cmd.limit
            position_elements = self.server.positions_queue.get_all(limit)
            telemetry_elements = self.server.telemetry_queue.get_all(limit)
            ts = datetime.now().isoformat()
            data = {
                'mode': self.server.flight_mode,
                'position': {
                    'timestamp': ts,
                    'size': len(position_elements),
                    'data': position_elements
                },
                'telemetry': {
                    'timestamp': ts,
                    'size': len(telemetry_elements),
                    'data': telemetry_elements
                }
            }
            return Status.success(data, message)

        if not self.server.is_running:
            return None

        self.is_processing = True
        self.command = cmd
        self.command_t0 = time.monotonic()

        logit(f'Running: {commands.name}', color='green')
        try:
            if commands == Commands.TAKE_IMAGE:
                # Execute the take image command here
                if self.server.operation_enabled is not None and not self.server.operation_enabled:
                    self.server.camera_take_image(cmd)
                    ret = Status.get_status(Status.SUCCESS, "Image captured.")
                else:
                    self.server.server.messages.write('Warning: Cannot take image, autonomous operation is enabled.', 'warning')
                    ret = Status.get_status(Status.ERROR, "'Warning: Cannot take image, autonomous operation is enabled.")
            elif commands == Commands.RESUME_OPERATION: # Start
                self.server.camera_resume(cmd)
                ret = Status.get_status(Status.SUCCESS, "Resumed.")
            elif commands == Commands.PAUSE_OPERATION: # Stop
                self.server.camera_pause()
                ret = Status.get_status(Status.SUCCESS, "Paused.")
            elif commands == Commands.RUN_AUTOFOCUS:
                self.server.camera_run_autofocus(cmd)
                ret = Status.get_status(Status.SUCCESS, "Autofocus completed.")
            elif commands == Commands.SET_GAIN:
                self.server.camera_set_gain(cmd)
                ret = Status.get_status(Status.SUCCESS, "Gain set.")
            elif commands == Commands.SET_APERTURE_POSITION:
                self.server.camera_set_aperture_position(cmd)
                ret = Status.get_status(Status.ERROR, "Command camera_set_aperture not implemented.")
            elif commands == Commands.SET_FOCUS_POSITION:
                self.server.camera_set_focus_position(cmd)
                ret = Status.get_status(Status.SUCCESS, "Focus position set.")
            elif commands == Commands.DELTA_FOCUS_POSITION:
                self.server.camera_delta_focus_position(cmd)
                ret = Status.get_status(Status.SUCCESS, "Focus delta position set.")
            elif commands == Commands.RUN_AUTOGAIN:
                self.server.camera_run_autogain(cmd)
                ret = Status.get_status(Status.SUCCESS, "Autogain completed.")
            elif commands == Commands.SET_EXPOSURE_TIME:
                self.server.camera_set_exposure_time(cmd)
                ret = Status.get_status(Status.SUCCESS, "Exposure time set.")
            elif commands == Commands.ENABLE_DISTORTION_CORRECTION:
                self.server.camera_enable_distortion_correction(cmd)
                ret = Status.get_status(Status.SUCCESS, "Distortion correction enabled.")
            elif commands == Commands.DISABLE_DISTORTION_CORRECTION:
                self.server.camera_disable_distortion_correction()
                ret = Status.get_status(Status.SUCCESS, "Distortion correction disabled.")
            elif commands == Commands.INPUT_GYRO_RATES:
                self.server.camera_input_gyro_rates(cmd)
                ret = Status.get_status(Status.SUCCESS, "New Gyro rates set.")
            elif commands == Commands.UPDATE_TIME:
                self.server.camera_update_time(cmd)
                ret = Status.get_status(Status.SUCCESS, "Time updated set.")
            elif commands == Commands.POWER_CYCLE:
                self.server.camera_power_cycle(cmd)
                ret = Status.get_status(Status.SUCCESS, "Powercycle not implemented.")
            elif commands == Commands.HOME_LENS:
                self.server.camera_home_lens(cmd)
                ret = Status.get_status(Status.SUCCESS, "Camera Home lens completed.")
            elif commands == Commands.FLIGHT_MODE:
                if cmd.method == 'set':
                    self.server.flight_mode = cmd.mode
                ret = Status.success({'data': {'mode': self.server.flight_mode}}, f"Flight mode {cmd.method}.")
            else:
                self.log.warning("Unknown command received: %s", commands)
                return Status.get_status(Status.ERROR, f"Unknown command: {commands}")
        except Exception as e:
            ret = Status.get_status(Status.ERROR, f"Error running command: {commands}")
            for line in traceback.format_exception(e):
                self.server.logit(line.strip(), 'error') # .strip()
                # logit(line, color='red')

        self.server.camera_get_values()
        self.server.server.messages.write(f'Command {commands} status: {ret}')
        # Command completed
        self.is_processing = False
        if commands in [Commands.FLIGHT_MODE]:
            return ret
        return None

    def start_server(self, pueo_server):
        """Start the server to listen for incoming client connections."""
        self.server = pueo_server
        self.is_running = True
        try:
            self.socket.bind((self.pueo_server_ip, self.port))  # Binding/Starting Pueo Server
            self.socket.listen()
            self.log.info("Pueo Server listening on %s:%d", self.pueo_server_ip, self.port)
            # In your while loop:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    client_socket, _ = self.socket.accept()
                    client_socket.settimeout(1.0)  # 1 second timeout
                    self.connected = True
                    self.log.info("Accepted new client connection")
                    client_thread = Thread(target=self.handle_client, args=(client_socket,))
                    client_thread.daemon = True  # Mark as daemon thread
                    client_thread.start()
                    self.client_threads.append(client_thread)  # Track the thread
                except socket.timeout as e:
                    # Timeout occurred, check if the server is still running
                    continue
                except socket.error as e:
                    if self.is_running:
                        self.log.critical(f"Server error: {e}")
                    break
        except socket.error as e:
            self.log.critical(f"Server error: {e}")
        finally:
            self.stop_server()

        self.log.debug('Exiting: start_server')

    def stop_server(self):
        """Stop the server and clean up resources."""
        self.is_running = False
        self.log.debug("Stopping server...")

        # Close the server socket to unblock the accept() call
        if self.socket:
            self.socket.close()
            self.log.debug("Server socket closed")

        # Wait for all client threads to finish
        for thread in self.client_threads:
            thread.join(timeout=5)
        self.log.debug("All client threads terminated")
        self.log.debug("Server stopped")

    def send_command(self, command: Command):
        def __send_command():
            self.camera.set_camera_status(self.camera.SENDING, command.command_name)
            res = self.send(command.command_data)
            res = SimpleNamespace(**res)
            if res.error_code == Status.SUCCESS:
                self.camera.set_camera_status(self.camera.OK, res.error_message)
            else:
                self.camera.set_camera_status(self.camera.ERROR, res.error_message)

        if self.connected and not self.is_connecting:
            self.log.info('Starting command')
            Thread(target=__send_command, daemon=True).start()
        # TODO: This next two lines are to be deleted. It was for testing only.
        # else:
        #     self.log.warning('Skipping send command: not connected or connecting.')

    def start_status_check(self, interval=2):
        def check_status():
            self.log.debug(f'start_status_check: ready: {self.connected}')
            command = Command()
            command_data = command.check_status()
            self.log.debug(f'command_data: {command_data}')

            while True:
                while self.connected and not self.is_connecting:
                    # self.camera.set_camera_status(self.camera.SENDING, f"Status check.")
                    self.camera.set_camera_status(self.camera.SENDING)
                    res = self.send(command_data, is_check=True)

                    with suppress(TypeError):
                        res = SimpleNamespace(**res)

                    if res is None:
                        # Communication socket error, no result
                        self.camera.set_camera_status(self.camera.ERROR, 'Error receiving server response.')
                    elif res.error_code == Status.SUCCESS:
                        self.camera.set_camera_status(self.camera.OK, res.error_message)
                    else:
                        self.camera.set_camera_status(self.camera.ERROR, res.error_message)

                    # Process messages
                    if hasattr(res, 'messages'):
                        messages = res.messages
                        self.messages.process_messages(messages)

                    time.sleep(interval)
                time.sleep(interval)

        if self.is_client:
            self.log.info('Starting start_status_check')
            Thread(target=check_status, daemon=True).start()

    def write(self, item, level='info', data_type='message', dst_filename=None):
        """Write saves the item to the message queue.
        It handles different types such as 'message', 'file'.
        It saves the data as dict that is used to send over a socket connection to the client
        """
        if self.connected:
            self.messages.write(item, level, data_type, dst_filename)

        # If not connected to the client, delete images in preflight mode.
        if data_type.endswith('_file') and not self.server.is_flight:
            with suppress(FileNotFoundError, PermissionError, Exception):
                file_path = item
                os.remove(file_path)
                self.log.warning(f"File {file_path} deleted successfully. flight_mode: {self.server.flight_mode}")
        # self.log.warning('Not connected.')

    def close(self):
        """Close the socket connection gracefully."""
        self.log.warning('Closing.')
        with suppress(Exception):
            self.socket.close()
            self.connected = False
            self.log.info("Connection closed")


class DummyPueoServer:
    """Testing Dummy Server"""
    def __init__(self, server):
        self.log = logging.getLogger('pueo')
        self.server = server
        for i in range(3):
            self.server.messages.write(f'Test info messages {i}: Krneki {i}', 'info')
            self.server.messages.write(f'Test warning messages {i}: Krneki {i}', 'warning')
            self.server.messages.write(f'Test error messages {i}: Krneki {i}', 'error')
            self.server.messages.write(f'Test error messages {i}: Krneki {i}', 'debug')
            self.server.messages.write(f'Test error messages {i}: Krneki {i}', 'critical')

        self.images_dir = '../output/downsampled'
        self.images = self.get_files(self.images_dir)
        self.image_idx = 0
        self.tokens_to_randomize = ['CPU Temperature', 'exposure time (us)']
        self.operation_enabled = False
        self.solver = 'solver1'

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.log.warning(f"Dummy Server Method '{name}' called with args: {args}, kwargs: {kwargs}")
            self.server.messages.write(f'DUMMY message from client: Method: {name} with args: {args}')
            if name == 'camera_take_image':
                image_file = self.get_next_image()
                info_filename = 'dummy_info_file.txt'
                info_file = f'{image_file[:-4]}.txt'
                src_info_file = f'{str(self.images_dir)}/{info_filename}'
                self.log.debug(f'Copying info dummmy file: {src_info_file} -> {info_file}')
                shutil.copy(src_info_file, info_file)
                self.dummy_info(info_file, self.tokens_to_randomize)
                self.server.messages.write(image_file, data_type='image_file')
                self.server.messages.write(info_file, data_type='info_file')
        return method

    def write(self, msg, level='info', data_type='message'):
        self.server.messages.write(msg, level, data_type)

    def camera_resume(self, cmd):
        self.operation_enabled = True
        self.solver = cmd.solver

    def camera_pause(self):
        self.operation_enabled = False

    def get_next_image(self):
        if self.image_idx >= len(self.images):
            self.image_idx = 0
        image = self.images[self.image_idx]
        self.image_idx += 1
        return image

    def get_files(self, path, ext='.png') -> list:
        """
        Retrieves a list of files with the specified extension from the given path.

        Args:
        path (str): The path to the directory to search.
        ext (str, optional): The file extension to filter for. Defaults to '.png'.

        Returns:
        list: A list of file paths with the specified extension.
        """
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
              if filename.endswith(ext):
                files.append(os.path.join(root, filename))
        self.log.debug(f'Found images: {len(files)}')
        return files

    def dummy_info(self, filename, tokens: list):
        """
        Randomizes the values of specific tokens in a text file.

        Args:
            self: The object instance (unused in this function).
            filename (str): The path to the text file.
            tokens (list): A list of tokens to randomize.
        """

        self.log.debug(f'Dummy editing file: {filename}')
        with open(filename, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)  # Move the file pointer to the beginning

            for line in lines:
                for token in tokens:
                    if token in line:
                        # Extract the current value
                        value = line.split(':')[1].strip()
                        # Randomize the value (adjust the randomization range as needed)
                        new_value = str(random.uniform(0, 1000))
                        if token == 'CPU Temperature':
                            _value = random.uniform(0, 100)
                            new_value = f'{_value:.1f} °C'
                        elif token == 'exposure time (us)':
                            _value = random.uniform(1000, 100000)
                            new_value = f'{_value:.0f}'
                        # Replace the old value with the new one
                        self.log.debug(f'replacing: {value} -> {new_value}')
                        line = line.replace(value, new_value)
                f.write(line)
            f.truncate()  # Remove any extra lines at the end

if __name__ == "__main__":
    from types import SimpleNamespace
    from common import load_config
    log = load_config(config_file='../../conf/config.ini', log_file_name_token="server_log")  # Initialise logging, config.ini, ...
    log.debug('Main - star_comm_bridge')

    mode = 'Server'
    if mode == 'Client':
        # Test Client
        config = {'pueo_server_ip': 'localhost', 'port': 5555, 'retry_delay': 2, 'max_retry': 5}
        config = SimpleNamespace(**config)
        client = StarCommBridge(is_client=True, config=config)
        # client.start_status_check()
        command = Command().run_autofocus('sequence_contrast', 1000, 5000, 200)
        response = client.send(command)
        print(f'Response: {response}')
    elif mode == 'Server':
        # Test SERVER
        print('Starting TEST Server')
        config = {'pueo_server_ip': 'localhost', 'port': 5555, 'retry_delay': 2, 'max_retry': 5}
        config = SimpleNamespace(**config)
        server = StarCommBridge(is_client=False, config=config)
        pueo_server = DummyPueoServer(server)
        server.start_server(pueo_server)

    time.sleep(1000)


