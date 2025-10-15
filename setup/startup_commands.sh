#!/bin/bash
# Pueo Startup Script by Milan (info@stubljar.com)
#   File: ~/scripts/startup_commands.sh

LOG_FILE=~/Projects/pcc/logs/startup.log
PUEO_CONSOLE_FILE=~/Projects/pcc/logs/pueo_console.log
CEDAR_CONSOLE_FILE=~/Projects/pcc/logs/cedar_console.log
WEB_CONSOLE_FILE=~/Projects/pcc/logs/web_console.log

# Web server configuration
WEB_PORT=8000
WEB_DIRECTORY=~/Projects/pcc/web

# Log script execution
printf "[%s] Executed ~/scripts/startup_commands.sh\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# Start Cedar Detect
printf "[%s] Starting Cedar Detect:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd ~/Projects/pcc/lib/cedar_detect/python
# Cedar Detect Server Startup (Development)
# cargo run --release --bin cedar-detect-server > "$CEDAR_CONSOLE_FILE" 2>&1 &

# Cedar Detect Server Startup (Production)
# Start the Cedar Detect server in the background with proper logging.
# - 'unbuffer' allocates a pseudo-TTY so the server can run correctly,
#   because it expects a TTY on startup (otherwise it may exit or stop logging).
# - 'stdbuf -oL -eL' forces line-buffered stdout and stderr so log lines
#   are flushed immediately, even when redirected to a file.
# Note: requires: unbuffer: sudo apt install expect
unbuffer stdbuf -oL -eL ~/Projects/pcc/lib/cedar_detect/target/release/cedar-detect-server > "$CEDAR_CONSOLE_FILE" 2>&1 &

# script is more heavy on creating tty:
# WORKS: script -q -c "stdbuf -oL -eL ~/Projects/pcc/lib/cedar_detect/target/release/cedar-detect-server" /dev/null > "$CEDAR_CONSOLE_FILE" 2>&1 &

CEDAR_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$CEDAR_PID" >> "$LOG_FILE"

# Start Pueo Star Tracker Server
printf "[%s] Starting Pueo Star Tracker Server:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd ~/Projects/pcc

# The -oL option tells stdbuf to force the standard output stream of the command to be line-buffered.
# This overrides the default behavior where the C library switches to full buffering when output is
# redirected to a file. Line buffering ensures each line is written to $PUEO_CONSOLE_FILE immediately,
# allowing 'tail -f' to display the output in near real-time.
stdbuf -oL -eL .venv/bin/python pueo_star_camera_operation_code.py > "$PUEO_CONSOLE_FILE" 2>&1 &
# ./.venv/bin/python pueo_star_camera_operation_code.py > "~/Projects/pcc/logs/pueo-console.log" 2>&1 &

PUEO_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$PUEO_PID" >> "$LOG_FILE"

# Start Python HTTP Server
printf "[%s] Starting Python HTTP Server:\n" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
cd ~/Projects/pcc
.venv/bin/python -m http.server "$WEB_PORT" --directory "$WEB_DIRECTORY" > "$WEB_CONSOLE_FILE" 2>&1 &
WEB_PID=$!
printf "[%s]  Process id: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$WEB_PID" >> "$LOG_FILE"
printf "[%s]  Serving directory: %s on port: %d\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$WEB_DIRECTORY" "$WEB_PORT" >> "$LOG_FILE"

# Store PIDs in files for later reference
echo "$CEDAR_PID" > ~/Projects/pcc/logs/cedar.pid
echo "$PUEO_PID" > ~/Projects/pcc/logs/pueo.pid
echo "$WEB_PID" > ~/Projects/pcc/logs/web.pid