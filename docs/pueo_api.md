# Pueo Star Tracker API Specification
**Version:** 1.0
**Last Updated:** 2025-10-07
**Contact:** [Milan Stubljar](mailto:info@stubljar.com)

## Overview
The Pueo Star Tracker API enables programmatic interaction with the star tracker system via JSON messages over TCP socket (port 5555). This document specifies the flight_telemetry command for retrieving flight telemetry data.

---

## Connection Methods

### **Linux/macOS (nc/netcat)**
```bash
echo '{"command": "flight_telemetry"}' | nc localhost 5555
```
- If `nc` is not installed, you can get it via:
  - **Debian/Ubuntu**: `sudo apt install netcat`
  - **macOS**: Usually preinstalled.

### **Windows (using `ncat` from Nmap)**
Since Windows doesn’t have `nc` by default, you can use `ncat` (from **Nmap**):
1. **Install Nmap** (if not installed): [Download Nmap](https://nmap.org/download.html)
2. Run:
   ```powershell
   echo '{"command": "flight_telemetry"}' | ncat localhost 5555
   ```
   (or in **cmd**):
   ```cmd
   echo {"command": "flight_telemetry"} | ncat localhost 5555
   ```

### **Alternative (PowerShell on Windows)**
If you don’t want to install `ncat`, you can use **PowerShell**:
```powershell
$msg = '{"command": "flight_telemetry"}'
$socket = New-Object System.Net.Sockets.TcpClient("localhost", 5555)
$stream = $socket.GetStream()
$writer = New-Object System.IO.StreamWriter($stream)
$writer.WriteLine($msg)
$writer.Flush()
$socket.Close()
```

### **Notes**
- Make sure your server is running (`localhost:5555`) before sending the message.
- If your server expects a **newline** at the end, add `\n` (Linux) or use `echo -e`:
  ```bash
  echo -e '{"command": "flight_telemetry"}\n' | nc localhost 5555
  ```
- If you need **verbose output**, use `-v`:
  ```bash
  echo '{"command": "flight_telemetry"}' | nc -v localhost 5555
  ```

---

### **Command Specification**  
**Command:** `flight_telemetry`  
**Purpose:** Retrieve a list of recent flight telemetry records.  

#### **Parameters**  
| Name   | Type    | Description                                                                 | Default | Constraints       |  
|--------|---------|-----------------------------------------------------------------------------|---------|-------------------|  
| `limit`| `int`   | Number of records to return. `0` returns all records (since last call).     | `0`     | `0 ≤ limit ≤ 100` |  

**Note:** The maximum length of returned data depends on two factors:  
1. **Available Data** – The number of existing solutions and telemetry entries.  
2. **Configured Limit** – The value set in `config.ini` under `[STAR_COMM_BRIDGE]/fq_max_size`.  

Example configuration:  
```ini
# Max number of solutions retained for flight computer API exchange  
fq_max_size = 1  
```

#### **Example Request**  
```json
{
  "command": "flight_telemetry",
  "data": {
    "limit": 5
  }
}
```

#### **Example Response**  
```json
{
  "error_code": 0,
  "error_message": "Idle",
  "data": {
    "mode": "flight",
    "position": {
      "timestamp": "2025-04-22T16:45:43.617632",
      "size": 1,
      "data": [
        {
          "timestamp": "2025-04-22T16:45:38.727641",
          "solver": "solver2",
          "solver_name": "Cedar Tetra3",
          "astro_position": [
            78.72341647554903,
            7.712427624830612,
            71.83045121390376
          ],
          "FOV": 10.804579396546963,
          "RMSE": 14.250483723479707,
          "RMS": [
            19492.219293473838,
            107.51525421665598,
            646769.7752664452
          ],
          "sources": 30,
          "matched_stars": 18,
          "probability": 2.446187512831069e-22,
          "angular_velocity": [
            NaN,
            NaN,
            NaN
          ]
        }
      ]
    },
    "telemetry": {
      "timestamp": "2025-04-22T16:45:43.617632",
      "size": 1,
      "data": [
        {
          "timestamp": "2025-04-22T16:45:42.967302",
          "headers": [
            "capture_time",
            "drivetemp-scsi-0-0_temp",
            "acpitz-acpi-0_temp",
            "drivetemp-scsi-1-0_temp",
            "package_id_0_temp",
            "core_0_temp",
            "core_1_temp",
            "core_2_temp",
            "core_3_temp",
            "core0_load",
            "core1_load",
            "core2_load",
            "core3_load",
            "core4_load",
            "core5_load",
            "core6_load",
            "core7_load",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10"
          ],
          "data": [
            "2025-04-22 16:45:41",
            "28.0 \u00b0C",
            "29.0 \u00b0C",
            "27.0 \u00b0C",
            "30.0 \u00b0C",
            "29.0 \u00b0C",
            "30.0 \u00b0C",
            "29.0 \u00b0C",
            "29.0 \u00b0C",
            "39.4 %",
            "26.6 %",
            "39.1 %",
            "43.8 %",
            "21.9 %",
            "15.6 %",
            "37.5 %",
            "34.4 %",
            "22.17 \u00b0C",
            "62.02 \u00b0C",
            "50.4 \u00b0C",
            "69.87 \u00b0C",
            "58.29 \u00b0C",
            "39.13 \u00b0C",
            "76.36 \u00b0C",
            "29.62 \u00b0C",
            "47.86 \u00b0C",
            "72.89 \u00b0C"
          ]
        }
      ]
    },
    "filesystem": {
      "timestamp": "2025-10-06T22:43:01.702627",
      "size": 1,
      "data": [
        {
          "root": {
            "status": "normal",
            "is_critical": false,
            "levels": {
              "warning": 95.0,
              "critical": 99.0
            },
            "current": {
              "ts": 1759826529,
              "folders": 0,
              "files": 0,
              "size_mb": 0.0,
              "used_mb": 80187.375,
              "total_mb": 128371.004,
              "used_pct": 62.47,
              "free_pct": 37.53
            },
            "trend": {
              "1h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "6h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "24h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "week": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "month": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "all": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              }
            },
            "forecast": {
              "critical": null,
              "warning": null
            }
          },
          "ssd": {
            "status": "normal",
            "is_critical": false,
            "levels": {
              "warning": 95.0,
              "critical": 99.0
            },
            "current": {
              "ts": 1759826529,
              "folders": 2,
              "files": 682,
              "size_mb": 1439.732,
              "used_mb": 80187.375,
              "total_mb": 128371.004,
              "used_pct": 62.47,
              "free_pct": 37.53
            },
            "trend": {
              "1h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "6h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "24h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "week": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "month": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "all": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              }
            },
            "forecast": {
              "critical": null,
              "warning": null
            }
          },
          "sd_card": {
            "status": "normal",
            "is_critical": false,
            "levels": {
              "warning": 95.0,
              "critical": 99.0
            },
            "current": {
              "ts": 1759826529,
              "folders": 2,
              "files": 682,
              "size_mb": 366.969,
              "used_mb": 80187.375,
              "total_mb": 128371.004,
              "used_pct": 62.47,
              "free_pct": 37.53
            },
            "trend": {
              "1h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "6h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "24h": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "week": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "month": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              },
              "all": {
                "folders": 0,
                "files": 0,
                "size_mb": 0.0,
                "start_ts": 1759825689,
                "end_ts": 1759826529
              }
            },
            "forecast": {
              "critical": null,
              "warning": null
            }
          }
        }
      ]
    }
  }
}
```  

#### **Notes**  
- **`data`** in the response will contain an array of camera telemetry/solution/filesystem records. Number of elements of the telemetry is dynamic and can vary based on the actual sensors reported by systems.
- **`limit=0`** fetches all available records since the last query.  
- Error codes:  
  - `0`: Success.  
  - Non-zero: Error (details in `error_message`).  

---

### **Command Specification**  
**Command:** `flight_mode`  
**Purpose:** Get or set the current flight mode (e.g., `preflight` or `flight`).  

#### **Parameters**  
| Name     | Type    | Description                                                                 | Required For  | Constraints                     |  
|----------|---------|-----------------------------------------------------------------------------|---------------|---------------------------------|  
| `method` | `str`   | Operation to perform: `"get"` (retrieve current mode) or `"set"` (change mode). | Always        | Must be `"get"` or `"set"`.     |  
| `mode`   | `str`   | Target flight mode (only required if `method = "set"`).                     | `method=set`  | Must be `"preflight"` or `"flight"`. |  

**Notes:**  
1. **Mode Handling**  
   - Input is case-insensitive (e.g., `"PreFlight"` → normalized to `"preflight"`).  
   - Invalid modes raise an error (see **Example Error Response**).  

2. **State Persistence**  
   - The mode persists until explicitly changed or system reset.  

#### **Example Request (Get Current Mode)**  
```json
{
  "command": "flight_mode",
  "data": {
    "method": "get"
  }
}

//Simplified:

{
  "command": "flight_mode",
}
```

#### **Example Request (Set Mode to Flight)**  
```json
{
  "command": "flight_mode",
  "data": {
    "method": "set",
    "mode": "flight"
  }
}
```

#### **Example Response (Success)**  
```json
{
  "error_code": 0,
  "error_message": "Flight mode set.",
  "data": {
    "mode": "flight"  // Returned for both get/set
  }
}
```


---
