#!/bin/bash

# Color codes for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR" || exit 1

# Log file
LOG_FILE="$SCRIPT_DIR/debug-test.log"
SERVER_LOG="$SCRIPT_DIR/debug-server.log"
CYCLE_DURATION=60 # Default 60 seconds
SOLVERS=("solver1" "solver2" "solver3")
declare -A cycle_results
declare -A image_stats

# Debug directories - clear on startup
DEBUG_DIR="$SCRIPT_DIR/test_logs"
mkdir -p "$DEBUG_DIR"
# Delete all existing files in debug directory
rm -rf "$DEBUG_DIR"/*
mkdir -p "$DEBUG_DIR"

# Function to log and print with colors
log_and_print() {
    local color=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Log to file (plain text)
    echo "[$timestamp] $message" >> "$LOG_FILE"

    # Print to console with color
    if [[ "$color" == "NOCOLOR" ]]; then
        echo "$message"
    else
        echo -e "${color}[$timestamp] $message${NC}"
    fi
}

# Function for detailed debug logging
debug_log() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] DEBUG: $message" >> "$LOG_FILE"
    echo "[$timestamp] DEBUG: $message" >> "$DEBUG_DIR/detailed-debug.log"
}

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS] [CYCLE_DURATION]"
    echo "Perform autonomous performance testing for PUEO Startracker"
    echo ""
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --solvers LIST  Comma-separated list of solvers to test (default: solver1,solver2,solver3)"
    echo ""
    echo "Arguments:"
    echo "  CYCLE_DURATION  Cycle duration in seconds (default: 60)"
    echo ""
    echo "Note: This script must be run from its directory and will change to the parent directory"
    echo "      where pc.sh is located (../ relative to script location)"
    echo ""
    echo "Example:"
    echo "  $0 120"
    echo "  $0 --solvers solver1,solver3 180"
    exit 0
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                ;;
            --solvers)
                IFS=',' read -ra SOLVERS <<< "$2"
                shift 2
                ;;
            *)
                if [[ $1 =~ ^[0-9]+$ ]]; then
                    CYCLE_DURATION=$1
                    shift
                else
                    log_and_print "$RED" "Error: Invalid argument '$1'"
                    exit 1
                fi
                ;;
        esac
    done
}

# Function to wait with progress indicator
wait_with_progress() {
    local duration=$1
    local message=$2
    local increment=1

    log_and_print "$BLUE" "$message"
    echo -n "["
    for ((i=0; i<duration; i+=increment)); do
        sleep $increment
        echo -n "#"
    done
    echo "]"
}

# Function to capture current server log state for debugging
capture_log_state() {
    local phase=$1
    local debug_file="$DEBUG_DIR/server_log_state_${phase}_$(date +%Y%m%d_%H%M%S).log"

    debug_log "Capturing server log state for phase: $phase"

    if [[ -f "$SERVER_LOG" ]]; then
        # Capture basic info
        echo "=== SERVER LOG STATE: $phase ===" > "$debug_file"
        echo "File: $SERVER_LOG" >> "$debug_file"
        echo "Size: $(wc -c < "$SERVER_LOG") bytes" >> "$debug_file"
        echo "Lines: $(wc -l < "$SERVER_LOG")" >> "$debug_file"
        echo "Last modified: $(ls -la "$SERVER_LOG")" >> "$debug_file"
        echo "" >> "$debug_file"

        # Capture recent content
        echo "=== LAST 20 LINES ===" >> "$debug_file"
        tail -20 "$SERVER_LOG" >> "$debug_file" 2>/dev/null
        echo "" >> "$debug_file"

        # Capture camera events if any
        echo "=== CAMERA EVENTS FOUND ===" >> "$debug_file"
        grep -n "camera_take_image" "$SERVER_LOG" | tail -10 >> "$debug_file" 2>/dev/null
    else
        echo "SERVER LOG DOES NOT EXIST" >> "$debug_file"
    fi

    debug_log "Server log state saved to: $debug_file"
}

# Function to monitor camera take image events - FIXED TIME REMAINING VERSION
monitor_camera_events() {
    local solver=$1
    local start_time=$(date +%s)
    local end_time=$((start_time + CYCLE_DURATION))
    local temp_file="$DEBUG_DIR/camera_events_${solver}_$(date +%Y%m%d_%H%M%S).log"
    local count=0
    local last_time_report=0

    debug_log "Starting monitor_camera_events for solver: $solver"
    debug_log "Start time: $start_time, End time: $end_time"
    debug_log "Temp file: $temp_file"
    debug_log "Server log: $SERVER_LOG, exists: $(if [[ -f "$SERVER_LOG" ]]; then echo "YES"; else echo "NO"; fi)"

    # Capture initial log state
    capture_log_state "before_monitor_$solver"

    # Get current file size to read from this point forward
    local start_size=0
    if [[ -f "$SERVER_LOG" ]]; then
        start_size=$(wc -c < "$SERVER_LOG" 2>/dev/null || echo 0)
    fi
    debug_log "Initial log size: $start_size bytes"

    log_and_print "$CYAN" "Monitoring camera events for solver: $solver (duration: ${CYCLE_DURATION}s)"

    # Report initial time
    local time_remaining=$((end_time - start_time))
    log_and_print "$BLUE" "Time remaining: $time_remaining seconds"
    last_time_report=$time_remaining

    # Main monitoring loop
    while [[ $(date +%s) -lt $end_time ]]; do
        local current_time=$(date +%s)
        local time_remaining=$((end_time - current_time))

        # Show time remaining every 10 seconds (proper implementation)
        if [[ $time_remaining -le $((last_time_report - 10)) ]] || [[ $time_remaining -eq 0 ]]; then
            log_and_print "$BLUE" "Time remaining: $time_remaining seconds"
            debug_log "Time remaining: $time_remaining seconds, Events captured so far: $count"
            last_time_report=$time_remaining
        fi

        # Check if server log exists and has grown
        if [[ -f "$SERVER_LOG" ]]; then
            local current_size=$(wc -c < "$SERVER_LOG" 2>/dev/null || echo 0)
            debug_log "Current log size: $current_size bytes, Start size: $start_size bytes"

            if [[ $current_size -gt $start_size ]]; then
                debug_log "New content detected! Reading from byte $((start_size + 1)) to $current_size"

                # Read new content and filter for camera events
                tail -c +$((start_size + 1)) "$SERVER_LOG" 2>/dev/null | grep "camera_take_image completed" > "$temp_file.tmp"

                if [[ -s "$temp_file.tmp" ]]; then
                    # Append new events to our temp file
                    cat "$temp_file.tmp" >> "$temp_file"
                    local new_count=$(wc -l < "$temp_file.tmp")
                    count=$((count + new_count))

                    debug_log "Found $new_count new camera events in this iteration"
                    debug_log "New events content: $(cat "$temp_file.tmp")"

                    log_and_print "$GREEN" "Captured $new_count new camera events (total: $count)"
                else
                    debug_log "No camera events found in new content, but general content exists"
                    # Log what was actually found for debugging
                    tail -c +$((start_size + 1)) "$SERVER_LOG" 2>/dev/null | head -5 > "$DEBUG_DIR/non_camera_content_$(date +%H%M%S).log"
                fi

                start_size=$current_size
            else
                debug_log "No new content in server log"
            fi
        else
            debug_log "WARNING: Server log does not exist at path: $SERVER_LOG"
            debug_log "Current directory: $(pwd), Files: $(ls -la 2>/dev/null)"
        fi

        sleep 5  # Changed from 2 to 5 seconds as requested
    done

    # Capture final log state
    capture_log_state "after_monitor_$solver"

    # Process captured events
    if [[ -f "$temp_file" ]]; then
        debug_log "Processing captured events from: $temp_file"
        debug_log "Temp file content: $(cat "$temp_file" 2>/dev/null | head -10)"

        local events=()
        while IFS= read -r line; do
            events+=("$line")
        done < "$temp_file"

        cycle_results["$solver,count"]=${#events[@]}
        debug_log "Total events parsed: ${#events[@]}"

        # Calculate statistics
        local total=0
        local min=999999
        local max=0

        for event in "${events[@]}"; do
            debug_log "Processing event: $event"
            # Extract duration using regex
            if [[ $event =~ completed\ in\ ([0-9]+\.[0-9]+)\ seconds ]]; then
                local duration=${BASH_REMATCH[1]}
                total=$(echo "$total + $duration" | bc -l)
                debug_log "Extracted duration: $duration, Total so far: $total"

                # Update min/max
                if (( $(echo "$duration < $min" | bc -l) )); then
                    min=$duration
                    debug_log "New min: $min"
                fi
                if (( $(echo "$duration > $max" | bc -l) )); then
                    max=$duration
                    debug_log "New max: $max"
                fi
            else
                debug_log "NO DURATION FOUND in event: $event"
                debug_log "Trying alternative pattern matching..."
                # Try alternative patterns
                if [[ $event =~ ([0-9]+\.[0-9]+)\ seconds ]]; then
                    local duration=${BASH_REMATCH[1]}
                    debug_log "Alternative pattern found duration: $duration"
                    total=$(echo "$total + $duration" | bc -l)
                fi
            fi
        done

        local avg=0
        if [[ ${#events[@]} -gt 0 ]]; then
            avg=$(echo "scale=3; $total / ${#events[@]}" | bc -l)
        fi

        image_stats["$solver,min"]=$min
        image_stats["$solver,max"]=$max
        image_stats["$solver,avg"]=$avg

        debug_log "Final stats - Count: ${#events[@]}, Min: $min, Max: $max, Avg: $avg"

        # Keep the temp file for debugging (don't remove it)
        debug_log "Keeping temp file for analysis: $temp_file"
    else
        debug_log "WARNING: No temp file found at $temp_file"
        cycle_results["$solver,count"]=0
    fi

    log_and_print "$GREEN" "Completed monitoring for $solver. Captured ${cycle_results["$solver,count"]:-0} events."
}

# Function to run a test cycle
run_cycle() {
    local solver=$1

    log_and_print "$MAGENTA" "=========================================================="
    log_and_print "$MAGENTA" "Starting test cycle for solver: $solver"
    log_and_print "$MAGENTA" "=========================================================="

    # Start the solver
    log_and_print "$CYAN" "Starting solver: $solver"
    debug_log "Executing: ./pc.sh start $solver"

    # Capture command output for debugging
    ./pc.sh start "$solver" > "$DEBUG_DIR/pc_start_${solver}_$(date +%H%M%S).log" 2>&1
    local exit_code=$?
    debug_log "pc.sh start exit code: $exit_code"
    debug_log "Command output: $(cat "$DEBUG_DIR/pc_start_${solver}_*.log" 2>/dev/null | tail -5)"

    if [[ $exit_code -ne 0 ]]; then
        log_and_print "$RED" "Failed to start solver: $solver (exit code: $exit_code)"
        return 1
    fi

    # Wait a moment for the solver to initialize
    sleep 3
    debug_log "Waited 3 seconds after starting solver"

    # Monitor camera events
    monitor_camera_events "$solver"

    # Stop the solver
    log_and_print "$CYAN" "Stopping solver: $solver"
    debug_log "Executing: ./pc.sh stop"

    ./pc.sh stop > "$DEBUG_DIR/pc_stop_${solver}_$(date +%H%M%S).log" 2>&1
    local stop_code=$?
    debug_log "pc.sh stop exit code: $stop_code"

    if [[ $stop_code -ne 0 ]]; then
        log_and_print "$RED" "Failed to stop solver: $solver (exit code: $stop_code)"
    fi

    # Wait before next cycle
    wait_with_progress 30 "Waiting 30 seconds before next cycle..."

    log_and_print "$GREEN" "Cycle for $solver completed"
    log_and_print "$GREEN" "Events captured: ${cycle_results["$solver,count"]:-0}"
}

# Function to display summary
display_summary() {
    log_and_print "$YELLOW" "=================================================================="
    log_and_print "$YELLOW" "=== TEST SUMMARY ==="
    log_and_print "$YELLOW" "=================================================================="
    log_and_print "NOCOLOR" "Cycle duration: $CYCLE_DURATION seconds"
    log_and_print "NOCOLOR" "Solvers tested: ${SOLVERS[*]}"
    log_and_print "NOCOLOR" "Working directory: $(pwd)"
    log_and_print "NOCOLOR" "Debug logs saved in: $DEBUG_DIR"
    log_and_print "NOCOLOR" ""

    printf "%-15s %-10s %-10s %-10s %-10s\n" "Solver" "Events" "Min(sec)" "Max(sec)" "Avg(sec)" | tee -a "$LOG_FILE"
    printf "%-15s %-10s %-10s %-10s %-10s\n" "------" "------" "--------" "--------" "--------" | tee -a "$LOG_FILE"

    for solver in "${SOLVERS[@]}"; do
        local count=${cycle_results["$solver,count"]:-0}
        local min=${image_stats["$solver,min"]:-0}
        local max=${image_stats["$solver,max"]:-0}
        local avg=${image_stats["$solver,avg"]:-0}

        printf "%-15s %-10d %-10.3f %-10.3f %-10.3f\n" "$solver" "$count" "$min" "$max" "$avg" | tee -a "$LOG_FILE"
    done
}

# Main execution
main() {
    # Parse arguments
    parse_arguments "$@"

    # Create log file
    > "$LOG_FILE"
    > "$DEBUG_DIR/detailed-debug.log"

    log_and_print "$GREEN" "=================================================================="
    log_and_print "$GREEN" "PUEO Startracker Autonomous Mode Performance Test"
    log_and_print "$GREEN" "=================================================================="
    log_and_print "$YELLOW" "Cycle duration: $CYCLE_DURATION seconds"
    log_and_print "$YELLOW" "Testing solvers: ${SOLVERS[*]}"
    log_and_print "$YELLOW" "Working directory: $(pwd)"
    log_and_print "$YELLOW" "Script directory: $SCRIPT_DIR"
    log_and_print "$YELLOW" "Debug directory: $DEBUG_DIR"

    # Log system information for debugging
    debug_log "Script started with arguments: $*"
    debug_log "System: $(uname -a)"
    debug_log "Bash version: $BASH_VERSION"
    debug_log "Current user: $(whoami)"
    debug_log "Current directory: $(pwd)"
    debug_log "Directory contents: $(ls -la 2>/dev/null | head -10)"

    # Initial setup - CORRECTED sequence
    log_and_print "$BLUE" "Stopping autonomous mode..."
    debug_log "Executing: ./pc.sh stop"
    ./pc.sh stop > "$DEBUG_DIR/pc_stop_initial.log" 2>&1
    debug_log "Exit code: $?"

    debug_log "Executing: ./pc.sh set_flight_mode preflight"
    ./pc.sh set_flight_mode preflight > "$DEBUG_DIR/pc_set_flight.log" 2>&1
    debug_log "Exit code: $?"

    debug_log "Executing: ./pc.sh set_chamber_mode True"
    ./pc.sh set_chamber_mode True > "$DEBUG_DIR/pc_set_chamber.log" 2>&1
    debug_log "Exit code: $?"

    # Warning for user about RemotePC App
    log_and_print "$RED" "=================================================================="
    log_and_print "$RED" " PLEASE DISCONNECT FROM REMOTEPC APP NOW"
    log_and_print "$RED" " Test will begin in 30 seconds"
    log_and_print "$RED" "=================================================================="

    wait_with_progress 30 "Waiting for user to disconnect from RemotePC App..."

    # Run test cycles
    for solver in "${SOLVERS[@]}"; do
        run_cycle "$solver"
    done

    # Display summary
    display_summary

    log_and_print "$GREEN" "All cycles completed. Exiting."
    log_and_print "$GREEN" "Debug logs saved in: $DEBUG_DIR"
}

# Check if bc is available for floating point calculations
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' command is required for floating point calculations"
    echo "Install it with: sudo apt install bc"
    exit 1
fi

# Check if server log exists
if [[ ! -f "$SERVER_LOG" ]]; then
    log_and_print "$YELLOW" "Warning: $SERVER_LOG not found. Will monitor for its creation."
    debug_log "Server log not found at: $SERVER_LOG"
fi

# Check if pc.sh exists
if [[ ! -f "./pc.sh" ]]; then
    log_and_print "$RED" "Error: pc.sh not found in current directory: $(pwd)"
    debug_log "pc.sh not found in: $(pwd)"
    debug_log "Directory contents: $(ls -la 2>/dev/null)"
    exit 1
fi

# Run main function with all arguments
main "$@"