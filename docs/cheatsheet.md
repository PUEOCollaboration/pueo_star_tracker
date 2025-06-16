# PUEO Star Tracker Cheatsheet  
**Version:** 1.0 (Preliminary)  
**Last Updated:** 2025-06-09  
**Contact:** [Milan Stubljar](mailto:info@stubljar.com)
---

## Notes  
- **Paths:** Always navigate to the script’s directory before execution.  
  ```bash
  # Example:
  cd ~/Projects/pcc/logs/
  ./cleanup_data.sh
  ```

---

## Server Management  
### Autostart  
The server starts automatically on boot via:  
```bash
~/scripts/startup_commands.sh
```  
**Processes:**  
1. `CEDAR Detect Server`  
2. `PUEO Star Tracker Server`  

### Logs & PIDs  
| Component          | Console Output                  | Debug Logs                     | PID File                     |
|--------------------|---------------------------------|--------------------------------|------------------------------|
| PUEO Server        | `~/Projects/pcc/logs/pueo-console.log` | `~/Projects/pcc/logs/debug-server.log` | `~/Projects/pcc-gui/logs/pueo.pid` |
| CEDAR Server       | `~/Projects/pcc/logs/cedar_console.log` | N/A                            | `~/Projects/pcc-gui/logs/cedar.pid` |
| PUEO CLI           | N/A                             | `~/Projects/pcc/logs/debug-client.log` | N/A                          |
| PUEO GUI           | N/A                             | `~/Projects/pcc-gui/logs/debug-client.log` | N/A                          |

### Automatic Start/Stop/Restart/Status using PUEO Server Status Tool
```bash
# Get Status:
~/Projects/pcc/logs/status.sh
# or 
~/Projects/pcc/logs/status.sh status
# Example output:
VL Install: Installed & Enabled (Success)
CEDAR Detect: Running (PIDs: 39977)
PUEO Server: Running (PIDs: 39980,40181)

# Start - will check automatically if the server is running already, stop it and startit.
~/Projects/pcc/logs/status.sh start

# Stop 
~/Projects/pcc/logs/status.sh stop

# Restart
~/Projects/pcc/logs/status.sh restart
# Example output:
Stopping services (delay: 2s)...
  CEDAR Detect: Terminated PIDs: 38631
  PUEO Server: Terminated PIDs: 38634
38844
Starting services...
  Executed: /home/pst/scripts/startup_commands.sh
Waiting 5s for servers to initialize...

Final Status:
VL Install: Installed & Enabled (Success)
CEDAR Detect: Running (PIDs: 39977)
PUEO Server: Running (PIDs: 39980)

# Help:
~/Projects/pcc/logs/status.sh -h
PUEO Server Status Tool
Usage: ./status.sh [command]
Commands:
  status       Show current status (default)
  start        Stop (if running) and start services
  stop         Kill all CEDAR/PUEO processes
  restart      Stop + Start services
  -h, --help   Show this help

Reports VL installation state and CEDAR/PUEO process status with PIDs
```

### Manual Restart  
1. **Stop existing processes:**  
   ```bash
   ps aux | grep -E "cedar-detect-server|pueo_star_camera_operation"  
   kill -9 <PID>  # Kill all listed PIDs
   ```  
2. **Update configs** (e.g., `conf/config.ini`).  
3. **Restart:**  
   ```bash
   ~/Projects/scripts/startup_commands.sh  # No output; verify via PID/log files
   ```  
   *Monitor progress with:*  
   ```bash
   tail -f ~/Projects/pcc/logs/debug-server.log
   ```

---

## Interfaces  
### CLI  
```bash
~/Projects/pcc/pc.sh  # Standard CLI
~/Projects/pcc/pci.sh  # Interactive CLI
```

### GUI  
```bash
~/Projects/pcc-gui/run.sh
```  
**Remote Connection:**  
1. Set `server_ip = <SERVER_IP>` in `~/Projects/pcc-gui/conf/config.ini`.  
2. Restart GUI after server changes (unless running on Windows).  

---

## Maintenance Tools  
- **Cleanup:**  
  ```bash
  ~/Projects/pcc/logs/cleanup_data.sh  # Confirms before deletion
  ```  
- **Status Check:**  
  ```bash
  ~/Projects/pcc/logs/status.sh  # Lists active processes
  ```

---

## Use Cases  
### Chamber Mode  
1. **Edit `conf/config.ini`:**  
   ```ini
   run_chamber = True  
   flight_mode = flight
   run_autonomous = True  
   ```  
2. **Cleanup:**  
   ```bash
   ~/Projects/pcc/logs/cleanup_data.sh
   ```  
3. **Add test images** to `~/Projects/pcc/test_images/` (`.png` files).  
4. **Restart server** (follow [Manual Restart](#manual-restart)).  

### Mission Mode  
1. **Edit `conf/config.ini`:**  
   ```ini
   run_chamber = False  
   flight_mode = preflight
   run_autonomous = True  
   ```  
2. **Cleanup and restart server.**  
3. **Start data recording during mission:**  
   ```bash
   ~/Projects/pcc/pc.sh set_flight_mode flight
   ```  
   *Or send JSON payload:*  
   ```json
   {"command": "flight_mode", "data": {"method": "set", "mode": "flight"}}
   ```  
4. **Stop server post-mission.**  

### Manual Autofocus/Autogain
1) How to perform a **manual autofocus**. Where to find file generated/images:

**Ensure:**
- autonomous: stopped (pasued)
- chamber_mode: False
- flight_mode: flight

```bash
# CLI:
./pc.sh stop 
./pc.sh set_flight_mode flight
./pc.sh set_chamber_mode False
./pc.sh auto_focus
```

Note: The following **auto_focus** options are available.
```bash
$ python pueo_cli.py auto_focus -h
pueo-cli v1.0.0
Reading config file: conf/config.ini
usage: pueo_cli.py auto_focus [-h] [start_position] [stop_position] [step_count]

Positional args: [start] [stop] [step] (or none for defaults)

positional arguments:
  start_position  Start position value (default: 5000)
  stop_position   Stop position value (default: 6000)
  step_count      Step count value (default: 10)

options:
  -h, --help      show this help message and exit
```


The **resulting images** (flight_mode='flight') and files will reside in the "**autogain**" subfolder: ```~/Projects/pcc/autogain```

```bash
pst@erin-03:~/Projects/pcc/autogain$ ll
total 376
drwxr-xr-x 5 pst  pst  364544 Jun 16 06:36 ./
drwxr-xr-x 6 root root   4096 Jun  2 03:26 ../
drwxrwxr-x 2 pst  pst    4096 Jun 16 06:36 250616_063404.062499_auto_gain_exposure_images/
    -rw-rw-r-- 1 pst pst    17508 Jun 16 06:34 250616_063406.175537_eg120_histogram.jpg
    -rw-rw-r-- 1 pst pst 11090925 Jun 16 06:36 250616_063406.175537_eg120.png
    -rw-rw-r-- 1 pst pst      240 Jun 16 06:34 250616_063406.175537_eg120.txt
drwxrwxr-x 2 pst  pst    4096 Jun 16 06:36 250616_063429.005674_coarse_focus_images/
    -rw-rw-r-- 1 pst pst 3876249 Jun 16 06:36 250616_063431.171664_f8242.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:34 250616_063431.171664_f8242.txt
    -rw-rw-r-- 1 pst pst 3883078 Jun 16 06:36 250616_063440.798067_f8266.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:34 250616_063440.798067_f8266.txt
    -rw-rw-r-- 1 pst pst 3886129 Jun 16 06:36 250616_063450.140885_f8291.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:34 250616_063450.140885_f8291.txt
    -rw-rw-r-- 1 pst pst 3887147 Jun 16 06:36 250616_063459.389541_f8315.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063459.389541_f8315.txt
    -rw-rw-r-- 1 pst pst 3889653 Jun 16 06:36 250616_063508.757726_f8340.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063508.757726_f8340.txt
    -rw-rw-r-- 1 pst pst 3895876 Jun 16 06:36 250616_063517.927965_f8364.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063517.927965_f8364.txt
    -rw-rw-r-- 1 pst pst 3901938 Jun 16 06:36 250616_063527.106097_f8389.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063527.106097_f8389.txt
    -rw-rw-r-- 1 pst pst 3905954 Jun 16 06:36 250616_063536.366798_f8413.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063536.366798_f8413.txt
    -rw-rw-r-- 1 pst pst 3902918 Jun 16 06:36 250616_063545.533872_f8438.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063545.533872_f8438.txt
    -rw-rw-r-- 1 pst pst 3901951 Jun 16 06:36 250616_063554.712080_f8462.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:35 250616_063554.712080_f8462.txt
    -rw-rw-r-- 1 pst pst 3914891 Jun 16 06:36 250616_063604.297356_f8352.png
    -rw-rw-r-- 1 pst pst     240 Jun 16 06:36 250616_063604.297356_f8352.txt
    -rw-rw-r-- 1 pst pst   45871 Jun 16 06:36 focus_score.png
drwxrwxr-x 2 pst  pst    4096 Jun 16 06:36 250616_063639.671767_auto_gain_exposure_images/
```


3) How to perform a **manual autogain**. Where to find files generated images?
Same as autofocus but the CLI command is auto_gain:
```bash
$ python pueo_cli.py auto_gain -h
pueo-cli v1.0.0
Reading config file: conf/config.ini
usage: pueo_cli.py auto_gain [-h] [desired_max_pixel_value]

positional arguments:
  desired_max_pixel_value
                        Auto gain desired max pixel value (default: 55702)

options:
  -h, --help            show this help message and exit
```

### Manual TAKE IMAGE
1) How to **take a single image** at exposure time and gain. Where to find it?

**Really Windell?**

```bash
./pc.sh stop 
./pc.sh set_flight_mode flight
./pc.sh set_chamber_mode False
./pc.sh set_exposure 125
./pc.sh set_gain 355
./pc.sh set_focus 8200
./pc.sh take_image

# or ... 
./pc.sh get_settings
#  To fetch all current settings, gain, exposure, focus, modes...

# Use ./pc.sh take_image -h
$ python pueo_cli.py take_image -h
pueo-cli v1.0.0
Reading config file: conf/config.ini
usage: pueo_cli.py take_image [-h] [{raw,solver1,solver2}]

positional arguments:
  {raw,solver1,solver2}
                        Image type (default: solver2)

options:
  -h, --help            show this help message and exit
# or refer to pueo_cli.md documentation in docs subfolder.
```