#!/bin/bash
# Pueo Startup Script by Milan (info@stubljar.com)
#   File: ~/scripts/startup_commands.sh

LOG_FILE=~/Projects/pcc/logs/startup.log
PUEO_CONSOLE_FILE=~/Projects/pcc/logs/pueo_console.log
CEDAR_CONSOLE_FILE=~/Projects/pcc/logs/cedar_console.log

# Log script execution
printf "[%s] Executed ~/scripts/startup_commands.sh\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# Start Cedar Detect
printf "[%s] Starting Cedar Detect:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd ~/Projects/pcc/lib/cedar_detect/python
cargo run --release --bin cedar-detect-server > "$CEDAR_CONSOLE_FILE" 2>&1 &
CEDAR_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$CEDAR_PID" >> "$LOG_FILE"

# Start Pueo Star Tracker Server
printf "[%s] Starting Pueo Star Tracker Server:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd ~/Projects/pcc
.venv/bin/python pueo_star_camera_operation_code.py > "$PUEO_CONSOLE_FILE" 2>&1 &
# ./.venv/bin/python pueo_star_camera_operation_code.py > "~/Projects/pcc/logs/pueo-console.log" 2>&1 &

PUEO_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$PUEO_PID" >> "$LOG_FILE"

# Optional: Store PIDs in a file for later reference
echo "$CEDAR_PID" > ~/Projects/pcc/logs/cedar.pid
echo "$PUEO_PID" > ~/Projects/pcc/logs/pueo.pid