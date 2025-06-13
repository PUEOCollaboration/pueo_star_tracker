#!/bin/bash
#
# Script used in chamber mode TESTING to periodically stop/check_lens/start procedure.

# Configuration
WAIT_AFTER_STOP=5           # seconds
WAIT_AFTER_START=300        # 5 minutes (configurable)
SCRIPT_NAME="./pc.sh"       # Path to your pc.sh script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print with color
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run a command with status reporting
run_command() {
    local command=$1
    local description=$2

    print_color "${BLUE}" "▶ ${description}..."
    if $command; then
        print_color "${GREEN}" "✓ ${description} completed successfully"
        return 0
    else
        print_color "${RED}" "✗ ${description} failed!"
        return 1
    fi
}

# Main loop
while true; do
    # Step 1: Stop
    if ! run_command "${SCRIPT_NAME} stop" "Stopping autonomous"; then
        print_color "${YELLOW}" "Warning: Stop command failed, continuing anyway"
    fi

    # Step 2: Wait after stop
    print_color "${BLUE}" "⏳ Waiting ${WAIT_AFTER_STOP} seconds after stop..."
    sleep ${WAIT_AFTER_STOP}

    # Step 3: Check lens
    if ! run_command "${SCRIPT_NAME} check_lens" "Checking lens"; then
        print_color "${RED}" "Error: Lens check failed! Continuing cycle..."
    fi

    # Step 4: Start
    if ! run_command "${SCRIPT_NAME} start" "Starting autonomous"; then
        print_color "${RED}" "Error: Start command failed! Trying again..."
        continue
    fi

    # Step 5: Wait after start
    print_color "${BLUE}" "⏳ Waiting ${WAIT_AFTER_START} seconds (${WAIT_AFTER_START}/60 mins) after start..."
    sleep ${WAIT_AFTER_START}

    # Repeat the cycle
    print_color "${YELLOW}" "\n🌀 Starting new cycle...\n"
done