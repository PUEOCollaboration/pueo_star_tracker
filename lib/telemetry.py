from contextlib import suppress
import subprocess
import psutil
import queue
import serial
import threading
import random
import time
import datetime
import platform

import subprocess
import re
from datetime import datetime
from typing import Dict, List, Tuple

from lib.common import load_config, logit, logpair, cprint, get_os_type, DroppingQueue

class DummySerial:
    """
    A dummy serial port class for testing.

    Simulates a serial port that sends a line of 10 random temperature values
    separated by commas every second.
    """

    def __init__(self, timeout=1.0, items=10):
        """
        Initializes the dummy serial port.

        Args:
            port (str): The name of the dummy port (not used in this implementation).
            baudrate (int): The baud rate (not used in this implementation).
        """
        self.timeout = timeout
        self.items = items
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._generate_data, daemon=True)
        self.thread.start()

    def _generate_data(self):
        """
        Thread function to generate and queue simulated serial data.
        """
        while self.running:
            temperatures = [f'S{idx+1}:{round(random.uniform(10, 100), 2)}' for idx in range(self.items)]
            line = ",".join(map(str, temperatures)) + "\n"
            self.queue.put(line.encode())
            time.sleep(self.timeout)

    def readline(self):
        """
        Reads a line of simulated data from the dummy serial port.

        Returns:
            bytes: A bytes object containing the simulated data.
        """
        try:
            return self.queue.get(timeout=1)
        except queue.Empty:
            return b''

    def close(self):
        """
        Stops the dummy serial port.
        """
        self.running = False
        self.thread.join()


class Sensors:
    """Sensors Class Implementation"""

    dummy_sensor_data = """drivetemp-scsi-0-0
Adapter: SCSI adapter
temp1:        +28.0°C  (low  =  +0.0°C, high = +70.0°C)
                       (crit low =  +0.0°C, crit = +70.0°C)
                       (lowest = +23.0°C, highest = +28.0°C)

acpitz-acpi-0
Adapter: ACPI interface
temp1:        +29.0°C  

drivetemp-scsi-1-0
Adapter: SCSI adapter
temp1:        +27.0°C  (low  =  +0.0°C, high = +70.0°C)
                       (crit low =  +0.0°C, crit = +70.0°C)
                       (lowest = +22.0°C, highest = +27.0°C)

coretemp-isa-0000
Adapter: ISA adapter
Package id 0:  +30.0°C  (high = +110.0°C, crit = +110.0°C)
Core 0:        +29.0°C  (high = +110.0°C, crit = +110.0°C)
Core 1:        +30.0°C  (high = +110.0°C, crit = +110.0°C)
Core 2:        +29.0°C  (high = +110.0°C, crit = +110.0°C)
Core 3:        +29.0°C  (high = +110.0°C, crit = +110.0°C)"""

    def __init__(self, log):
        self.log = log
        self.os_type = platform.system()
        if self.os_type == 'Windows':
            self.log.warning('Windows DETECTED. Using Dummy DATA!')
        else:
            self.log.info('Sensors initialized.')

        self.success_cnt = 0  # Success counter, only log success once.
        self.error_cnt = 0    # Error logging counter
        self.error_max = 5    # MAX error counter (do not log errors to log after error_cnt > error_max

        _, self.field_names = self.get_sensor_data()

        self.log.info(f'Sensors: {", ".join(self.field_names)}')

    def parse_sensors_output(self, output: str) -> List[Tuple[str, str, str, float]]:
        """
        Properly parse sensors output into structured data

        Args:
            output: The full text output from 'sensors' command

        Returns:
            List of tuples: (adapter_name, adapter_type, sensor_name, temperature)
        """
        results = []
        current_adapter = None
        current_type = None

        # Regular expression patterns
        adapter_pattern = re.compile(r'^([^:]+):$')  # "adapter-name:"
        type_pattern = re.compile(r'^Adapter:\s*(.*)$')  # "Adapter: type"
        temp_pattern = re.compile(r'^([^:]+):\s+\+([\d.]+)°C')  # "sensor_name: +32.1°C"

        for line in output.split('\n'):
            line = line.strip()
            if not line:
                current_adapter = None
                current_type = None
                continue

            # Check for adapter name (first line of sensor block)
            if not current_adapter:
                adapter_match = line # .strip() # adapter_pattern.match(line)
                if adapter_match:
                    current_adapter = adapter_match # .group(1)
                    continue

            # Check for adapter type (second line of sensor block)
            if current_adapter and not current_type:
                type_match = type_pattern.match(line)
                if type_match:
                    current_type = type_match.group(1)
                    continue

            # Check for temperature readings (subsequent lines)
            if current_adapter and current_type:
                temp_match = temp_pattern.match(line)
                if temp_match:
                    sensor_name = temp_match.group(1).strip()
                    temperature = float(temp_match.group(2))
                    results.append((current_adapter, current_type, sensor_name, temperature))

        return results

    def get_sensor_data(self) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Get sensor data by querying the system's sensor information.

        On Linux systems, uses the 'sensors' command to retrieve hardware sensor data.
        On Windows systems, returns dummy data for testing purposes.
        Handles cases where no sensors are found or the command fails.

        Returns:
            Tuple containing:
            - sensor_data: Nested dictionary structure {adapter_name: {sensor_name: temperature_value}}
            - field_names: List of all field names for CSV header, formatted consistently

        Example:
            sensor_data, field_names = get_sensor_data()
            # sensor_data: {'coretemp-isa-0000': {'Package id 0': 45.0, 'Core 0': 42.0}}
            # field_names: ['coretemp_package_id_0_temp', 'coretemp_core_0_temp']
        """
        sensor_data = {}
        field_names = []

        if self.os_type == 'Windows':
            # Use dummy data for Windows systems
            self.log.debug("Using dummy sensor data for Windows system")
            output = self.dummy_sensor_data
        else:
            try:
                # Execute sensors command and capture output
                # self.log.debug("Executing 'sensors' command to retrieve hardware data")
                output = subprocess.check_output(["sensors"], text=True, stderr=subprocess.DEVNULL)

                # Check if no sensors were found (common error case)
                if "No sensors found!" in output:
                    self.log.warning("No sensors detected. Try running 'sensors-detect' to configure sensors.")
                    return {}, []

            except subprocess.CalledProcessError as e:
                # Command failed (sensors not installed or other error)
                self.error_cnt += 1
                if self.error_cnt < self.error_max:
                    self.log.error(f"Failed to execute sensors command: {e}")
                return {}, []
            except FileNotFoundError:
                # sensors command not available on system
                self.error_cnt += 1
                if self.error_cnt < self.error_max:
                    self.log.error("'sensors' command not found. Install lm-sensors package.")
                return {}, []
            except Exception as e:
                # Catch any other unexpected errors
                self.error_cnt += 1
                if self.error_cnt < self.error_max:
                    self.log.error(f"Unexpected error reading sensor data: {e}")
                return {}, []

        # Parse the sensor output
        results = self.parse_sensors_output(output)

        # Log successful sensor detection
        if results:
            pass
            # self.log.debug(f"Successfully parsed data from {len(results)} sensors")
        else:
            self.error_cnt += 1
            if self.error_cnt < self.error_max:
                self.log.warning("No sensor data parsed from output")
            return {}, []

        # Process parsed results into structured data
        for adapter, _type, sensor, temp in results:
            # Initialize adapter dictionary if it doesn't exist
            if adapter not in sensor_data:
                sensor_data[adapter] = {}

            # Add sensor reading to the adapter
            sensor_data[adapter][sensor] = temp

            # Generate consistent field name for CSV header
            if adapter.startswith('coretemp'):
                # For coretemp adapters, use sensor name in field
                field_name = f"{adapter}_{sensor}_temp".lower().replace(' ', '_').replace('-', '_')
            else:
                # For other adapters, use adapter name in field
                field_name = f"{adapter}_temp".lower().replace(' ', '_').replace('-', '_')

            field_names.append(field_name)

        # Log successful data collection
        if sensor_data:
            if self.success_cnt == 0:
                self.success_cnt += 1
                self.log.info(f"Collected sensor data from {len(sensor_data)} adapters")
                # For critical success message, also log to console with color
                logit(f"Successfully retrieved sensor data from {len(sensor_data)} hardware adapters", color='green')
        else:
            self.error_cnt += 1
            if self.error_cnt < self.error_max:
                self.log.warning("No sensor data collected")

        return sensor_data, field_names

class Telemetry:
    """
    A class to handle serial port communication and logging telemetry data.
    """
    last_error = None

    def __init__(self, cfg, telemetry_queue=None, t_lock=None):
        """
        Initializes the serial port and logging configuration.

        Args:
            cfg (str): Config object
        """
        self.cfg = cfg
        self.is_enabled = self.cfg.run_telemetry
        self.port = self.cfg.telemetry_port
        self.baud_rate = self.cfg.telemetry_baud_rate
        self.dummy = None
        self.config_file = '../conf/config.ini'
        self.cnt = 0

        # log = logging.getLogger(__name__)
        self.log = load_config(
            name='pueo-telemetry',
            config_file=self.config_file,
            log_file_name_token="telemetry_log",
            is_global=False
        )  # Initialise logging, config.ini, ...

        self.log.info(f'Telemetry enabled: {self.is_enabled}')
        logpair('telemetry', self.is_enabled, color='green')
        if not self.is_enabled:
            return

        self.os_type = platform.system()
        self.arduino_serial = None
        self.init_arduino()

        self.sensors = Sensors(self.log)
        self.headers = []

        self.telemetry_queue = telemetry_queue
        self.t_lock = t_lock

        # Thread for reading serial data
        self.running = True
        self.read_thread = threading.Thread(target=self.run, daemon=True)

        # Start the reading thread in the background
        self.read_thread.start()

        time.sleep(5)
        logit(f'Telemetry initialized.', color='green')

    def init_arduino(self, is_silent=False):
        """
        Initializes or reinitializes the serial connection to the Arduino device.

        This method handles both initial setup and recovery scenarios when the connection is lost.
        If the serial port is already open, it will be properly closed before reinitialization.
        On failure, it will attempt to use a dummy serial interface for testing on Windows systems.

        Behavior:
        - If serial port exists but is closed/invalid, closes it and attempts to reconnect
        - On successful connection, logs the port details
        - On failure, logs the error and either uses a dummy interface (Windows) or gives up (other OS)

        Exceptions:
        - Logs SerialException errors but doesn't propagate them

        Side Effects:
        - May modify self.arduino_serial
        - Writes to log and console output
        """
        # Close existing connection if it exists
        if self.arduino_serial is not None:
            try:
                if hasattr(self.arduino_serial, 'is_open') and self.arduino_serial.is_open:
                    self.arduino_serial.close()
                    self.log.info("Closed existing serial connection before reinitialization")
                elif isinstance(self.arduino_serial, DummySerial):
                    self.log.info("Closing dummy serial interface before reinitialization")
            except Exception as e:
                self.log.error(f"Error while closing existing serial port: {e}")

        # Attempt to establish new connection
        try:
            self.arduino_serial = serial.Serial(self.port, self.baud_rate)
            self.log.info(f"Serial port connected: {self.port} @ {self.baud_rate} baud")
            return True
        except (serial.SerialException, PermissionError) as e:
            if not is_silent:
                self.log.error(f"Failed to connect to serial port: {e}; arduino temperature telemetry will not be captured.")
                cprint(f"Failed to connect to serial port: {e}; arduino temperature telemetry will not be captured.", color='red')

            if self.os_type == 'Windows':
                cprint('Using Dummy Serial for TESTING.', color='red')
                self.log.warning('Using Dummy Serial for TESTING.')
                self.arduino_serial = DummySerial()
                return True
            else:
                self.arduino_serial = None
                return False

    def get_cpu_loads(self):
        """Get CPU usage per core (0-100%)"""
        return psutil.cpu_percent(interval=1, percpu=True)

    def get_arduino_data(self):
        """Read temperature from Arduino (adjust based on your Arduino code)"""
        if not self.arduino_serial:
            timeout = 1.0 # Hearthbeat
            start = time.monotonic()
            is_connected = self.init_arduino(is_silent=True)

            # Pad remaining time to ensure exactly 1-second heartbeat spacing
            elapsed = time.monotonic() - start
            if elapsed < timeout:
                time.sleep(timeout - elapsed)

        self.last_error = None
        is_header = False
        try:
            max_retry = 11
            retry = 0
            while retry < max_retry:
                line = self.arduino_serial.readline().decode('utf-8').strip()  # Read and decode line
                if line:
                    item, *_ = str(line).split(' ', 1)
                    if item in ['Sensor', 'Number']:
                        self.log.info(f'Arduino: {line}')
                        is_header = True
                        continue
                    return line
                else:
                    # Skip empty line if we already received any header lines.
                    if not is_header:
                        break
                    pass
                retry += 1

            return None
        except UnicodeDecodeError as ue:
            if ue is not None and self.last_error is not None and self.last_error != ue:
                self.log.warning("Failed to decode serial data (may contain non-UTF-8 characters)")
            last_error = ue
        except Exception as e:
            if e is not None and self.last_error is not None and self.last_error != e:
                self.log.error(f"Error reading serial port: {e}")
            self.last_error = e
            self.init_arduino(is_silent=True)
        return None

    def collect_data(self):
        """Collect all available metrics"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [ts, ]

        # Temperature  metrics
        sensor_data, field_names = self.sensors.get_sensor_data()
        data += sensor_data
        self.headers = field_names

        # Dummy Temperature (for testing)
        if False:
            data += ['123.4 °K']
            self.headers += ['dummy_temp']

        # CPU Loads
        cpu_load_data = self.get_cpu_loads()
        data += [ f'{v} %' for v in cpu_load_data]
        cpu_load_names = [f'core{idx}_load' for idx, _ in enumerate(cpu_load_data)]
        self.headers += cpu_load_names

        # Arduino data
        # Telemetry data: S1:29.87,S2:24.44,S3:28.87,S4:28.37,S5:22.62,S6:24.31
        #             or: Error: No DS18B20 sensors found!
        #             or: Sensor 1 Address: 288C6B4000000008
        #             or: empty line. (None)
        arduino_temp = self.get_arduino_data()
        if arduino_temp is not None and str(arduino_temp).startswith('S'):
            arduino_raw_data = arduino_temp.split(',')
            arduino_data = [f'{s.split(":")[1]} °C' for s in arduino_raw_data]
            arduino_temp_names = [s.split(":")[0] for s in arduino_raw_data]
            data += arduino_data
            self.headers += arduino_temp_names
        return data, self.headers

    def log_data(self, filename="system_monitor_log.csv"):
        """Record all metrics in a CSV file"""
        data, headers = self.collect_data()

        data_line = ', '.join(data)
        header_line = ', '.join(headers)

        # Prepare Telemetry entry dict
        data_dict = {
            'timestamp': datetime.now().isoformat(),
            'headers': ['capture_time', ] + headers,
            'data': data,
            # 'data_pairs': dict(zip(headers, data)),
        }
        # Save Telemetry to the Queue for flight computer telemetry feed.
        if self.telemetry_queue:
            # self.log.debug(f'Putting to queue: size: {self.telemetry_queue.qsize()}/{self.telemetry_queue.maxsize}')
            self.telemetry_queue.put(data_dict)
            # self.log.debug(f'Queue size: {self.telemetry_queue.qsize()}/{self.telemetry_queue.maxsize}')

        # print(f'Logged data at {data[0]}')
        is_print = False

        if self.cnt % 25 == 0:
            if is_print:
                cprint(f'  Header: {header_line}', color='cyan')

            self.log.info(f'Telemetry header: {header_line}')  # Log telemetry data
        self.log.info(f'Telemetry data: {data_line}')  # Log telemetry data

        if is_print:
            cprint(f'    Data: {data_line}', color='white')

        self.cnt += 1

    def run(self):
        """Run continuous monitoring"""
        self.log.info("Starting system monitoring...")
        headers = ','.join(self.headers)
        self.log.info(f'Available metrics: {headers}')

        # Run logging every 5 seconds
        while self.running:
            self.log_data()

    def close(self):
        """
        Shuts down the serial port and stops the reading thread.
        """
        self.log.warning('Closing.')
        with suppress(Exception):
            self.running = False
            self.arduino_serial.close()
            self.log.info("Serial port closed")
            self.dummy.close()


# Example usage
if __name__ == "__main__":
    telemetry = None
    try:
        class cfg:
            run_telemetry = True
            telemetry_port = '/dev/ttyUSB0'
            telemetry_baud_rate = 115200
        telemetry_queue = DroppingQueue(maxsize=12)
        t_lock = threading.Lock()
        telemetry = Telemetry(cfg, telemetry_queue, t_lock)  # Replace with your port and baudrate
        # ... (your application logic) ...
    except Exception as e:
        print(f"Error: {e}")
    finally:
        time.sleep(50)
        if telemetry:
            telemetry.close()

# Last line
