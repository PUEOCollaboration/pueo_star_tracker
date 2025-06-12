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
~/Projects/scripts/startup_commands.sh
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

