#!/bin/bash

# Configuration
TARGET_DIR="$HOME/Projects/pcc"
MONITOR_DIRS=("output" "ssd_path" "logs" "log" "sd_card_path")  # Can include symlinks
LOG_FILE="$TARGET_DIR/logs/pcc_folder_stats.log"
INTERVAL_SECONDS=60

# Initialize stats storage
declare -A PREV_COUNT PREV_SIZE
declare -A SNAPSHOT_10M_COUNT SNAPSHOT_10M_SIZE
declare -A SNAPSHOT_30M_COUNT SNAPSHOT_30M_SIZE
declare -A SNAPSHOT_60M_COUNT SNAPSHOT_60M_SIZE

# Create log directory and file
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Function to resolve symlinks to their target
resolve_symlink() {
    local dir="$1"
    if [[ -L "$dir" ]]; then
        readlink -f "$dir"
    else
        echo "$dir"
    fi
}

# Function to convert bytes to MB
bytes_to_mb() {
    echo "$1" | awk '{printf "%.2f", $1/1024/1024}'
}

# Main monitoring loop
while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    LOG_LINE="$TIMESTAMP"

    # Initialize totals
    TOTAL_FILES=0
    TOTAL_SIZE_BYTES=0

    # Loop through directories (including symlinks)
    for DIR in "${MONITOR_DIRS[@]}"; do
        FULL_PATH="$TARGET_DIR/$DIR"
        RESOLVED_PATH=$(resolve_symlink "$FULL_PATH")

        # Skip if resolved path doesn't exist
        if [[ ! -e "$RESOLVED_PATH" ]]; then
            LOG_LINE+=",$DIR:NOT_FOUND"
            continue
        fi

        # Get current stats (follow symlinks with -L in find)
        CURRENT_COUNT=$(find -L "$FULL_PATH" -type f 2>/dev/null | wc -l)
        CURRENT_SIZE_BYTES=$(find -L "$FULL_PATH" -type f -exec stat -c "%s" {} + 2>/dev/null | awk '{sum+=$1} END {print sum}')
        CURRENT_SIZE_BYTES=${CURRENT_SIZE_BYTES:-0}
        CURRENT_SIZE_MB=$(bytes_to_mb "$CURRENT_SIZE_BYTES")

        # Calculate deltas (since last check)
        DELTA_COUNT=$((CURRENT_COUNT - PREV_COUNT["$DIR"]))
        DELTA_SIZE_BYTES=$((CURRENT_SIZE_BYTES - PREV_SIZE["$DIR"]))
        DELTA_SIZE_MB=$(bytes_to_mb "$DELTA_SIZE_BYTES")

        # Update previous values
        PREV_COUNT["$DIR"]=$CURRENT_COUNT
        PREV_SIZE["$DIR"]=$CURRENT_SIZE_BYTES

        # Calculate longer-term deltas (10m/30m/60m)
        DELTA_10M_COUNT=0
        DELTA_10M_SIZE_MB=0
        DELTA_30M_COUNT=0
        DELTA_30M_SIZE_MB=0
        DELTA_60M_COUNT=0
        DELTA_60M_SIZE_MB=0

        if [[ -v SNAPSHOT_10M_COUNT["$DIR"] ]]; then
            DELTA_10M_COUNT=$((CURRENT_COUNT - SNAPSHOT_10M_COUNT["$DIR"]))
            DELTA_10M_SIZE_MB=$(bytes_to_mb $((CURRENT_SIZE_BYTES - SNAPSHOT_10M_SIZE["$DIR"])))
        fi

        if [[ -v SNAPSHOT_30M_COUNT["$DIR"] ]]; then
            DELTA_30M_COUNT=$((CURRENT_COUNT - SNAPSHOT_30M_COUNT["$DIR"]))
            DELTA_30M_SIZE_MB=$(bytes_to_mb $((CURRENT_SIZE_BYTES - SNAPSHOT_30M_SIZE["$DIR"])))
        fi

        if [[ -v SNAPSHOT_60M_COUNT["$DIR"] ]]; then
            DELTA_60M_COUNT=$((CURRENT_COUNT - SNAPSHOT_60M_COUNT["$DIR"]))
            DELTA_60M_SIZE_MB=$(bytes_to_mb $((CURRENT_SIZE_BYTES - SNAPSHOT_60M_SIZE["$DIR"])))
        fi

        # Update totals
        TOTAL_FILES=$((TOTAL_FILES + CURRENT_COUNT))
        TOTAL_SIZE_BYTES=$((TOTAL_SIZE_BYTES + CURRENT_SIZE_BYTES))

        # Append to log line
        LOG_LINE+=",$DIR:Files=$CURRENT_COUNT(Δ1m=$DELTA_COUNT|Δ10m=$DELTA_10M_COUNT|Δ30m=$DELTA_30M_COUNT|Δ60m=$DELTA_60M_COUNT)"
        LOG_LINE+=",SizeMB=$CURRENT_SIZE_MB(Δ1m=$DELTA_SIZE_MB|Δ10m=$DELTA_10M_SIZE_MB|Δ30m=$DELTA_30M_SIZE_MB|Δ60m=$DELTA_60M_SIZE_MB)"
    done

    # Update snapshots periodically
    CURRENT_MINUTE=$(date +%M)
    if (( CURRENT_MINUTE % 10 == 0 )); then
        for DIR in "${MONITOR_DIRS[@]}"; do
            SNAPSHOT_10M_COUNT["$DIR"]=${PREV_COUNT["$DIR"]}
            SNAPSHOT_10M_SIZE["$DIR"]=${PREV_SIZE["$DIR"]}
        done
    fi
    if (( CURRENT_MINUTE % 30 == 0 )); then
        for DIR in "${MONITOR_DIRS[@]}"; do
            SNAPSHOT_30M_COUNT["$DIR"]=${PREV_COUNT["$DIR"]}
            SNAPSHOT_30M_SIZE["$DIR"]=${PREV_SIZE["$DIR"]}
        done
    fi
    if (( CURRENT_MINUTE == 0 )); then
        for DIR in "${MONITOR_DIRS[@]}"; do
            SNAPSHOT_60M_COUNT["$DIR"]=${PREV_COUNT["$DIR"]}
            SNAPSHOT_60M_SIZE["$DIR"]=${PREV_SIZE["$DIR"]}
        done
    fi

    # Log totals
    TOTAL_SIZE_MB=$(bytes_to_mb "$TOTAL_SIZE_BYTES")
    echo "$LOG_LINE,TOTAL:Files=$TOTAL_FILES,SizeMB=$TOTAL_SIZE_MB" >> "$LOG_FILE"

    # Optional console output
    echo -e "[$TIMESTAMP]\n  Total files: $TOTAL_FILES\n  Total size: $TOTAL_SIZE_MB MB"

    sleep $INTERVAL_SECONDS
done

