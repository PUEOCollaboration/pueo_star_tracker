#!/bin/bash
# Pueo Startup Script (Generic User)
# File: ~/scripts/startup_commands.sh

# Allow background processes to start without failing the script
set +e
set -euo pipefail

# Always print the script title
echo "PUEO Star Tracker Startup Script v2.1"

# --- Handle help option ---
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo
    echo "Usage: $0 [mode]"
    echo
    echo "mode: Optional startup mode (default: production)"
    echo "  production   Start all services normally (default)"
    echo "  development  Skip PUEO Star Tracker Server; start remaining services"
    exit 0
fi

# Detect current user home
USER_HOME="$HOME"

# Log directory
LOG_DIR="$USER_HOME/Projects/pcc/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/startup.log"
PUEO_CONSOLE_FILE="$LOG_DIR/pueo_console.log"
CEDAR_CONSOLE_FILE="$LOG_DIR/cedar_console.log"
WEB_CONSOLE_FILE="$LOG_DIR/web_console.log"

# Web server configuration
WEB_PORT=8000
WEB_DIRECTORY="$USER_HOME/Projects/pcc/web"
VENV_PYTHON="$USER_HOME/Projects/pcc/.venv/bin/python"

# Optional: ensure unbuffer exists
command -v /usr/bin/unbuffer >/dev/null || { echo "unbuffer not found"; exit 1; }

# Read mode parameter (default to "production")
MODE="${1:-production}"

# Validated input mode param
if [[ "$MODE" != "production" && "$MODE" != "development" ]]; then
    echo "Invalid mode '$MODE'. Allowed values: production, development."
    exit 1
fi

# Log script execution
printf "[%s] Executed %s/scripts/startup_commands.sh\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$USER_HOME" >> "$LOG_FILE"

# Start Cedar Detect
printf "[%s] Starting Cedar Detect:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd "$USER_HOME/Projects/pcc/lib/cedar_detect/python"
/usr/bin/unbuffer /usr/bin/stdbuf -oL -eL "$USER_HOME/Projects/pcc/lib/cedar_detect/target/release/cedar-detect-server" > "$CEDAR_CONSOLE_FILE" 2>&1 &
CEDAR_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$CEDAR_PID" >> "$LOG_FILE"

# Start Pueo Star Tracker Server
if [[ "$MODE" == "production" ]]; then
  printf "[%s] Starting Pueo Star Tracker Server:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
  cd "$USER_HOME/Projects/pcc"
  /usr/bin/stdbuf -oL -eL "$VENV_PYTHON" pueo_star_camera_operation_code.py > "$PUEO_CONSOLE_FILE" 2>&1 &
  PUEO_PID=$!
  printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$PUEO_PID" >> "$LOG_FILE"
else
    # Development mode: skip PUEO server
    printf "[%s] Development mode; start PUEO Star Tracker Server in IDE (PyCharm)\n" \
        "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    PUEO_PID=0
fi

# Start Python HTTP Server
printf "[%s] Starting Python HTTP Server:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd "$USER_HOME/Projects/pcc"
"$VENV_PYTHON" -m http.server "$WEB_PORT" --directory "$WEB_DIRECTORY" > "$WEB_CONSOLE_FILE" 2>&1 &
WEB_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$WEB_PID" >> "$LOG_FILE"
printf "[%s]  Serving directory: %s on port: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$WEB_DIRECTORY" "$WEB_PORT" >> "$LOG_FILE"

# Store PIDs in files for later reference
echo "$CEDAR_PID" > "$LOG_DIR/cedar.pid"
echo "$PUEO_PID" > "$LOG_DIR/pueo.pid"
echo "$WEB_PID" > "$LOG_DIR/web.pid"
