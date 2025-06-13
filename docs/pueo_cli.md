# PUEO Command Line Interface (CLI) Reference
**Version:** 1.0 (Preliminary)
**Last Updated:** 2025-05-23
**Contact:** [Milan Stubljar](mailto:info@stubljar.com)

## Overview
The `pueo-cli.py` tool provides command-line control for the PUEO server. It supports various commands for system control, focus operations, camera settings, and more.

## Basic Usage Using python3 or Bash Wrapper
```bash
python pueo-cli.py <command> [<args>...]

# or using bash wrapper
# Basic command
./pc.sh <command> [<args>...]

# Example:
./pc.sh start

```

## Command Reference

### System Control Commands

| Command        | Description                    | Arguments                                                                                                           |
|----------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `start`        | Start/resume autonomous mode   | `[solver]` (solver1, solver2), `[cadence]` (Cadence in seconds)<br/> (all optional, but must provide first or both) |
| `stop`         | Stop autonomous mode           | None                                                                                                                |
| `power_cycle`  | Power cycle camera and focuser | None                                                                                                                |

### Focus Operations

| Command      | Description                                 | Arguments                                                                                        |
|--------------|---------------------------------------------|--------------------------------------------------------------------------------------------------|
| `home_lens`  | Perform focuser homing                      | None                                                                                             |
| `check_lens` | Perform focuser check lens (MZ, MZ, MI, MI) | None (```stop``` autonomous before running ```check_lens```)                                    |
| `auto_focus` | Run autofocus routine                       | `[start_position] [stop_position] [step_count]`<br/>(all optional, but must provide all or none) |

### Camera Settings

| Command          | Description                          | Arguments                     |
|------------------|--------------------------------------|-------------------------------|
| `auto_gain`      | Run autogain routine                | `[desired_max_pixel_value]` (optional) |
| `auto_exposure`  | Run autoexposure routine            | `[desired_max_pixel_value]` (optional) |
| `take_image`     | Capture image                       | `[type]` (optional: raw, solver1, solver2) |

### Chamber Mode Control

| Command            | Description              | Arguments             |
|--------------------|--------------------------|-----------------------|
| `get_chamber_mode` | Get current chamber mode | None                  |
| `set_chamber_mode` | Set chamber mode         | `<mode>` (True/False) |

Note: Chamber mode is used for testing in a dark chamber, to generate heat by normal operation yet getting viable test images for solving from **test_images** files. Make sure to have this set to False in real flight.

### Flight Mode Control

| Command             | Description                          | Arguments                     |
|---------------------|--------------------------------------|-------------------------------|
| `get_flight_mode`   | Get current flight mode              | None                          |
| `set_flight_mode`   | Set flight mode                      | `<mode>` (preflight or flight) |

Note: In preflight mode the images are not saved to sd/ssd while all other operations are the same.

### Flight Telemetry Data

| Command                | Description               | Arguments                                   |
|------------------------|---------------------------|---------------------------------------------|
| `get_flight_telemetry` | Get flight telemetry data | `[limit]` (Number of solutions, 0 for all.) |

Note: Number of solutions kept by server is defined in ```config.ini```, default 1. 
```ini
# Max number of solutions available/to keep for flight computer API exchange
fq_max_size = 1
```

### Parameter Access Commands

#### Get Commands
```bash
pueo-cli.py get_aperture
pueo-cli.py get_aperture_position
pueo-cli.py get_focus
pueo-cli.py get_exposure
pueo-cli.py get_gain
pueo-cli.py get_settings
```

#### Set Commands
```bash
pueo-cli.py set_aperture <value>
pueo-cli.py set_aperture_position <value>
pueo-cli.py set_focus <value>
pueo-cli.py set_exposure <value>
pueo-cli.py set_gain <value>
```

## Examples

### Basic Operations
```bash
# Start autonomous mode
python pueo-cli.py start

# Stop autonomous mode
python pueo-cli.py stop

# Take an image and run solver1
python pueo-cli.py take_image solver1

# Take a raw image and skip astro solving.
python pueo-cli.py take_image raw
```

### Focus Control
```bash
# Run autofocus with defaults
python pueo-cli.py auto_focus

# Run autofocus with custom parameters
python pueo-cli.py auto_focus 100 200 10

# Home the lens
python pueo-cli.py home_lens
```

### Camera Configuration
```bash
# Set focus position
python pueo-cli.py set_focus 150

# Get current exposure
python pueo-cli.py get_exposure

# Run autoexposure
python pueo-cli.py auto_exposure
```

## Notes
1. When using `auto_focus`, either provide all three parameters or none (to use defaults)
2. Most set commands require root privileges
3. Default values are loaded from the configuration file

For interactive help, use:
```bash
python pueo-cli.py --full-help
```

## Example of runs

### 1. System Control
```bash
$ python pueo_cli.py stop
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'pause_operation', 'data': {}}
INFO: Response time: 1004.8ms Response:
{
  "error_code": 0,
  "error_message": "Paused."
}
INFO: Session completed
```

```bash
$ python pueo_cli.py start
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'resume_operation', 'data': {'solver': 'solver2', 'cadence': 5.0}}
INFO: Response time: 3.5ms Response:
{
  "error_code": 0,
  "error_message": "Resumed."
}
INFO: Session completed
```

### 2. Camera Settings
```bash
$ python pueo_cli.py set_gain 150
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'set_gain', 'data': {'gain': '150'}}
INFO: Response time: 4.4ms Response:
{
  "error_code": 0,
  "error_message": "Gain set."
}
INFO: Session completed
```

```bash
$ python pueo_cli.py get_gain
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'get', 'data': {'param': 'gain'}}
INFO: Response time: 4.8ms Response:
{
  "error_code": 0,
  "error_message": "Get gain.",
  "data": {
    "data": {
      "param": "gain",
      "value": 120
    }
  }
}
INFO: Session completed
```

### 3. Exposure Control
```bash
$ python pueo_cli.py set_exposure_time 200000
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'set_exposure_time', 'data': {'exposure_time': '200000'}}
INFO: Response time: 4.3ms Response:
{
  "error_code": 0,
  "error_message": "Exposure time set."
}
INFO: Session completed
```

```bash
$ python pueo_cli.py get_exposure
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'get', 'data': {'param': 'exposure'}}
INFO: Response time: 2.3ms Response:
{
  "error_code": 0,
  "error_message": "Get exposure.",
  "data": {
    "data": {
      "param": "exposure",
      "value": 30000
    }
  }
}
INFO: Session completed
```

### 4. System Status
```bash
$ python pueo_cli.py get_settings
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'get', 'data': {'param': 'settings'}}
INFO: Response time: 4.4ms Response:
{
  "error_code": 0,
  "error_message": "Get settings.",
  "data": {
    "data": {
      "param": "settings",
      "value": {
        "aperture": "closed",
        "aperture_pos": 0,
        "aperture_f_val": "??",
        "focus": null,
        "chamber_mode": false,
        "flight_mode": "preflight",
        "run_test": false,
        "run_telemetry": true,
        "autonomous": false,
        "solver": "solver2",
        "cadence": 5.0,
        "Gain": 120,
        "Exposure": 30000,
        "Offset": 0,
        "BandWidth": 100,
        "Flip": 0,
        "AutoExpMaxGain": 285,
        "AutoExpMaxExpMS": 30000,
        "AutoExpTargetBrightness": 100,
        "HighSpeedMode": 1,
        "Temperature": 228,
        "GPS": 0
      }
    }
  }
}
INFO: Session completed
```

### 5. Auto Functions (Note: These may timeout)
```bash
$ python pueo_cli.py auto_exposure
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'run_autoexposure', 'data': {'desired_max_pixel_value': 55702}}
ERROR: Server response timed out
WARNING: No response received from server
INFO: Session completed
```

**Note:** Auto functions (`auto_exposure`, `auto_gain`, `auto_focus`) may timeout as they typically take longer (∼25 seconds) than the default socket timeout (5 seconds) to complete. This is expected behavior.

## Error Examples

### 1. Invalid Focus Position
```bash
$ python pueo_cli.py set_focus 10000
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
ERROR: Command error: Invalid value: name: focus_position: 10000 range: 0 .. 9000
INFO: Session completed
```

### 2. Invalid Exposure Value
```bash
$ python pueo_cli.py set_exposure -10000
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
ERROR: Command error: Invalid value: name: exposure_time: -10000 range: 0 .. 5000000
INFO: Session completed
```

### 3. Partial Auto Focus Arguments
```bash
$ python pueo_cli.py auto_focus 100 200
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
ERROR: Command error: Either provide ALL positional arguments (start_position, stop_position, step_count) or NONE to use defaults.
Provided: 2/3 arguments.
INFO: Session completed
```

### 4. Invalid Flight Mode
```bash
$ python pueo_cli.py set_flight_mode invalid_mode
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
ERROR: Command error: argument mode: invalid choice: 'invalid_mode' (choose from 'preflight', 'flight')
INFO: Session completed
```

### 5. Server Timeout (Long-running Operations)
```bash
$ python pueo_cli.py auto_exposure
```
```
pueo-cli v1.0.0
Reading config file: conf/config.ini
INFO: Connected to server at 127.0.0.1:5555
INFO: Sending command: {'command': 'run_autoexposure', 'data': {'desired_max_pixel_value': 55702}}
ERROR: Server response timed out
WARNING: No response received from server
INFO: Session completed
```

**Note:** The error outputs show:
- Clear error messages with parameter validation details
- Valid value ranges when applicable
- Consistent formatting with the successful command outputs
- Timeout warnings for long-running operations
- Session completion messages in all cases

The errors maintain the same logging format as successful commands, making them easy to parse programmatically.