# PUEO Flight Computer Integration Guide

## Overview

This guide provides comprehensive instructions for integrating with the PUEO Flight Computer system. The PUEO system provides real-time star tracking, attitude determination, and telemetry monitoring capabilities through both socket-based and serial interfaces.

## 1. Real-Time Telemetry via Socket Interface

The primary method for obtaining current flight solutions is through the socket interface, which provides JSON-formatted telemetry data containing both positional solutions and system health information.

### Command Line Interface (CLI)

```bash
# Get the most recent flight telemetry
$ python pueo_cli.py get_flight_telemetry

# Alternative using the shell script wrapper
$ ./pc.sh get_flight_telemetry
```

### Python Code Example

```python
import json
import socket
import logging

class PueoClient:
    def __init__(self, host='127.0.0.1', port=5555, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Establish connection to PUEO server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.logger.info(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False
            
    def get_flight_telemetry(self, limit=0, metadata=False):
        """Request flight telemetry data"""
        from lib.commands import Command
        
        cmd = Command()
        command = cmd.flight_telemetry(limit=limit, metadata=metadata)
        cmd_str = json.dumps(command) + '\n'
        
        try:
            self.socket.sendall(cmd_str.encode('utf-8'))
            response = self.socket.recv(8192)
            return json.loads(response.decode('utf-8').strip())
        except Exception as e:
            self.logger.error(f"Telemetry request failed: {str(e)}")
            return None
            
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()

# Usage example
client = PueoClient()
if client.connect():
    telemetry = client.get_flight_telemetry(limit=1, metadata=False)
    if telemetry and telemetry.get('error_code') == 0:
        position_data = telemetry['data']['position']['data'][0]
        print(f"Current attitude: {position_data['astro_position']}")
        print(f"RMSE: {position_data['RMSE']}")
        print(f"Matched stars: {position_data['matched_stars']}")
    client.close()
```

### Response Structure

The telemetry response contains two main sections:

1. **Position Data**: Star tracker solution with attitude information
2. **Telemetry Data**: System health monitoring data

Key fields in the position solution:
- `astro_position`: [RA, Dec, Roll] in degrees
- `RMSE`: Root Mean Square Error of the solution
- `RMS`: Per-axis Root Mean Square errors
- `FOV`: Field of View in degrees
- `sources`: Number of detected stars
- `matched_stars`: Number of matched stars
- `probability`: Solution confidence probability

For detailed metadata information, refer to the [position_meta.py](https://github.com/windell747/pueo_star_tracker/blob/main/lib/position_meta.py) documentation.

## 2. Serial Interface for Flight Computer Integration

For flight computers that require serial communication, PUEO provides a JSON output stream that can be accessed by monitoring the astrometry log file.

### Implementation

The flight computer can connect via serial to a shell that continuously outputs the contents of the astrometry solution file:

```bash
# On the PUEO system, run:
$ tail -f /path/to/astro.json
```

### Data Format

The serial interface outputs JSON objects with the following structure:

```json
{
    "timestamp": "2025-09-03T21:05:23.490259",
    "solver": "solver1",
    "astro_position": [
        78.72695681386688,
        7.723743286263546,
        71.86314403595408
    ],
    "FOV": 10.804807808196644,
    "RMSE": 16.0852992272869,
    "RMS": [
        1477.4262478101796,
        8832.829487514024,
        54611.08512287879
    ],
    "sources": 28,
    "matched_stars": 20,
    "probability": 9.230307830297849e-34,
    "angular_velocity": [
        NaN,
        NaN,
        NaN
    ]
}
```

### Position Metadata Definition

The astrometry telemetry follows a well-defined metadata structure:

| Field | Description | Units | Type | Required |
|-------|-------------|-------|------|----------|
| `timestamp` | Timestamp of the position measurement in UTC | ISO 8601 datetime string | string | Yes |
| `solver` | Name of the astrometry solver algorithm | none | string | Yes |
| `astro_position` | Astronomical orientation coordinates | degrees | array[3] | Yes |
| `FOV` | Field of View - angular extent | degrees | float | Yes |
| `RMSE` | Root Mean Square Error | arcseconds | float | Yes |
| `RMS` | Per-axis Root Mean Square errors | arcseconds | array[3] | Yes |
| `sources` | Number of star sources detected | count | integer | Yes |
| `matched_stars` | Number of stars matched to catalog | count | integer | Yes |
| `probability` | Probability estimate of solution correctness | dimensionless (0-1) | float | Yes |
| `angular_velocity` | Angular velocity components | degrees per second | array[3] | No |

### Detailed Field Information

**astro_position elements:**
- Index 0: RA (Right Ascension) - celestial longitude coordinate (0-360°)
- Index 1: Dec (Declination) - celestial latitude coordinate (-90 to 90°)
- Index 2: Roll - rotation around the pointing axis (0-360°)

**RMS elements:**
- Index 0: RA_RMS - Right Ascension axis RMS error (arcseconds)
- Index 1: Dec_RMS - Declination axis RMS error (arcseconds)
- Index 2: Roll_RMS - Roll axis RMS error (arcseconds)

**angular_velocity elements (if available):**
- Index 0: roll_rate - Angular velocity around roll axis (deg/s)
- Index 1: az_rate - Angular velocity around azimuth axis (deg/s)
- Index 2: el_rate - Angular velocity around elevation axis (deg/s)

### Configuration

The serial interface uses the same data format as found in `lib/astro.json`. Baud rate and other serial parameters can be configured in the `conf/config.ini` file.

## 3. Inspection Images Access

PUEO captures and stores inspection images for diagnostic purposes. These images are available for download via the web interface or direct file access.

### Location
- Web access: `/inspection_images`
- File system: Path configured in `conf/config.ini` (typically in the `images` directory)

### Naming Convention
Images follow the format: `YYMMDD_HHMMSS.ssssss-raw-ds.jpg`

Example: `250903_190523.490259-raw-ds.jpg`
- Date: September 3, 2025 (25/09/03)
- Time: 19:05:23.490259
- Type: Raw downsampled image

### Retention
The system maintains the last 100 images by default. This value can be adjusted in the configuration.

## 4. Camera Control Operations

### Starting and Stopping Autonomous Mode

```bash
# Stop autonomous operation
$ ./pc.sh stop

# Start with specific solver
$ ./pc.sh start solver1  # ESA Tetra3 solver
$ ./pc.sh start solver2  # Cedar Tetra3 solver
$ ./pc.sh start solver3  # Astrometry.net solver

# Start with specific cadence (capture interval in seconds)
$ ./pc.sh start solver1 1.5  # Capture every 1.5 seconds
```

### Python Code for Operation Control

```python
from lib.commands import Command

cmd = Command()

# Stop autonomous operation
stop_command = cmd.pause_operation()

# Start autonomous operation with solver1 and 2-second cadence
start_command = cmd.resume_operation("solver1", 2.0)

# Send commands to server (using socket client as shown earlier)
```

## 5. Camera Parameter Configuration

PUEO allows dynamic adjustment of camera parameters through both CLI and programmatic interfaces.

### Common Parameter Commands

```bash
# Gain control
$ ./pc.sh set_gain 230
$ ./pc.sh get_gain

# Exposure control
$ ./pc.sh set_exposure 500000  # microseconds
$ ./pc.sh get_exposure

# Focus control
$ ./pc.sh set_focus 150
$ ./pc.sh get_focus

# Autogain control
$ ./pc.sh set_autogain true
$ ./pc.sh set_autogain false

# Get all current settings
$ ./pc.sh get_settings
```

### Programmatic Control Example

```python
from lib.commands import Command

cmd = Command()

# Set camera parameters
gain_command = cmd.set("gain", 230)
exposure_command = cmd.set("exposure", 500000)
focus_command = cmd.set("focus", 150)
autogain_command = cmd.set("autogain", True)

# Get current values
get_gain = cmd.get("gain")
get_exposure = cmd.get("exposure")
get_settings = cmd.get("settings")  # Returns all settings
```

### Help Documentation

For detailed parameter information and additional commands:

```bash
# Show help for specific command
$ ./pc.sh set_gain --help
$ ./pc.sh get_settings --help

# Comprehensive documentation
# See: https://github.com/windell747/pueo_star_tracker/blob/main/docs/pueo_cli.md
```

## 6. Web Access to Latest Images
**Web server** is listening on **port 8000**. Using browser open `localhost:8000`. The latest files are exposed:
- astro.json
- last_final_overlay_image_downscaled.png
- last_info_file.txt
- last_inspection_image.jpg.

The web root folder is `pcc/web`.

## 7. Advanced Socket API Integration

For direct socket communication, refer to the complete [PUEO API documentation](https://github.com/windell747/pueo_star_tracker/blob/main/docs/pueo_api.md).

### Connection Protocol

1. Establish TCP connection to configured IP and port (default: 127.0.0.1:5555)
2. Send JSON-formatted command followed by newline
3. Receive JSON response

### Example Command Structure

```python
{
    "command": "flight_telemetry",
    "data": {
        "limit": 0,          # Number of historical solutions (0 = most recent only)
        "metadata": False    # Whether to include metadata
    }
}
```

## 8. Troubleshooting

### Common Issues

1. **Connection refused**: Ensure PUEO server is running
2. **No response**: Check firewall settings and server configuration
3. **Invalid command**: Verify command format and parameters

### Getting Help

- Use `./pc.sh <command> --help` for command-specific information
- Consult the [main documentation](https://github.com/windell747/pueo_star_tracker/blob/main/docs/pueo_cli.md)
- Review the [pueo_cli.py source](https://github.com/windell747/pueo_star_tracker/blob/main/pueo_cli.py) for implementation details

## Support

For additional support or questions regarding PUEO integration, please refer to the project documentation or create an issue in the GitHub repository.