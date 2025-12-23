#!/usr/bin/env bash
# pcd.sh – "P"UEO "C"ollect logs, execute "D"iagnostic commands, and bundle outputs.
# Produces a single archive containing logs, command results, and inspection artifacts for PUEO support.

set -euo pipefail

# ---------- Colors ----------
CYAN='\033[0;36m'    # Script / header
BLUE='\033[0;34m'    # Phase / section headers
WHITE='\033[0m'      # Default / normal
YELLOW='\033[1;33m'  # Warnings
RED='\033[0;31m'     # Errors
GREEN='\033[0;32m'   # Success / OK messages
NC='\033[0m'         # Reset / no color

# ---------- Helper for colored messages ----------
msg() { local color="$1"; shift; echo -e "${color}$*${NC}"; }

# ---------- Script info ----------
SCRIPT_NAME="pcd.sh – PUEO Collect logs, execute diagnostic commands, and bundle outputs."
msg "$CYAN" "$SCRIPT_NAME"

# ---------- Possible PCC roots ----------
PCC_ROOTS=("/home/pst/Projects/pcc" "$HOME/Projects/pcc")
PCC_ROOT=""
for root in "${PCC_ROOTS[@]}"; do
    if [[ -d "$root" ]]; then
        PCC_ROOT="$root"
        break
    else
        msg "$WHITE" "  Checked but missing: $root"
    fi
done
[[ -n "$PCC_ROOT" ]] || { msg "$RED" "Error: None of the PCC_ROOT directories exist!"; exit 1; }
msg "$WHITE" "Using PCC_ROOT: $PCC_ROOT"

# ---------- Paths / Defaults ----------
OUT_DIR="${PCC_ROOT}/logs"
LOG_DIR="${PCC_ROOT}/logs"
CONF_DIR="${PCC_ROOT}/conf"
OUTPUT_ROOT="${PCC_ROOT}/output"
SDCARD_ROOT="${PCC_ROOT}/sd_card_path"
INSPECTION_DIR="${PCC_ROOT}/inspection_images"
AUTOGAIN_DIR="${PCC_ROOT}/autogain"
PC_DIR="${PCC_ROOT}"
PC_CMD="./pc.sh"

# ---------- Logs / Configs / Commands ----------
LOG_TAIL_LINES=100
LOGS=("${LOG_DIR}/pueo_console.log" "${LOG_DIR}/debug-server.log" "${LOG_DIR}/telemetry.log")
CONFIG_FILES=("${CONF_DIR}/config.ini" "${CONF_DIR}/dynamic.ini")
PC_COMMANDS=("get_settings" "get_flight_telemetry")   # add more if needed

# ---------- Timestamp / Staging ----------
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
staging="$(mktemp -d)"
cleanup() { rm -rf "$staging"; }
trap cleanup EXIT  # ensures staging is removed on any exit
mkdir -p "$staging/logs" "$staging/latest_files" "$staging/commands" "$staging/conf"

zip_path="${OUT_DIR}/pcd_bundle_${timestamp}.zip"

# ---------- Helper Functions ----------

# collect_log_tails <lines> <file1> <file2> ...
# Collect last N lines of each log file
collect_log_tails() {
    local lines="$1"; shift
    local logs=("$@")
    msg "$BLUE" "Collecting last ${lines} lines of logs..."
    for f in "${logs[@]}"; do
        if [[ -f "$f" ]]; then
            local base="$(basename "$f")"
            tail -n "$lines" -- "$f" > "$staging/logs/${base}.tail${lines}"
            msg "$GREEN" "  + ${base} (tail ${lines})"
        else
            msg "$YELLOW" "  WARN: Missing log: $f"
        fi
    done
}

# copy_config_files <file1> <file2> ...
# Copy config files into staging/conf
copy_config_files() {
    local files=("$@")
    msg "$BLUE" "Copying configuration files..."
    for f in "${files[@]}"; do
        if [[ -f "$f" ]]; then
            cp -a -- "$f" "$staging/conf/"
            msg "$GREEN" "  + $(basename "$f")"
        else
            msg "$YELLOW" "  WARN: Missing config file: $f"
            echo "Missing config file: $f" > "$staging/conf/$(basename "$f").MISSING.txt"
        fi
    done
}

# run_pc_command <cmd> [args...]
# Run a PC command and capture output
run_pc_command() {
    local args=("$@")
    local cmd_name="${args[0]}"
    local out_file="${staging}/commands/pc_${cmd_name}_${timestamp}.txt"
    msg "$BLUE" "Running command from ${PC_DIR}: ${PC_CMD} ${args[*]} ..."
    (
        cd "$PC_DIR"
        {
            echo "UTC Timestamp: $timestamp"
            echo "PWD: $(pwd)"
            echo "Command: ${PC_CMD} ${args[*]}"
            echo "---- STDOUT/STDERR ----"
            "${PC_CMD}" "${args[@]}"
            rc=$?
            echo
            echo "Exit code: $rc"
            exit $rc
        } > "$out_file" 2>&1
    ) || msg "$YELLOW" "  WARN: command '${cmd_name}' failed; output captured in bundle"
}

# newest_file <dir> <find predicates...>
newest_file() {
    local dir="$1"; shift
    [[ -d "$dir" ]] || return 1
    find "$dir" -maxdepth 1 -type f "$@" -printf '%T@ %p\0' 2>/dev/null \
        | sort -z -n | tail -z -n 1 | cut -z -d' ' -f2- | tr -d '\0'
}

# newest_dir <dir>
newest_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    find "$dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\0' 2>/dev/null \
        | sort -z -n | tail -z -n 1 | cut -z -d' ' -f2- | tr -d '\0'
}

# copy_newest_files <dir> <label> <count> <find predicates...> [--tag TAG]
copy_newest_files() {
    local dir="$1" label="$2" count="$3"; shift 3
    local tag=""
    [[ "$1" == "--tag" ]] && { tag="$2"; shift 2; }
    [[ -d "$dir" ]] || { msg "$YELLOW" "  WARN: Missing dir: $dir"; return 0; }
    mapfile -d '' -t files < <(
        find "$dir" -maxdepth 1 -type f "$@" -printf '%T@ %p\0' 2>/dev/null \
        | sort -z -nr | head -z -n "$count" | cut -z -d' ' -f2-
    )
    (( ${#files[@]} > 0 )) || { msg "$YELLOW" "  WARN: No files found in: $dir"; return 0; }
    local i=1
    for f in "${files[@]}"; do
        local base="$(basename "$f")"
        if [[ -n "$tag" ]]; then
            cp -a -- "$f" "$staging/latest_files/${label}__${tag}__newest${i}__${base}"
        else
            cp -a -- "$f" "$staging/latest_files/${label}__newest${i}__${base}"
        fi
        msg "$GREEN" "  + $f"
        ((i++))
    done
}

# collect_latest_pair <dir> <label> [tag]
collect_latest_pair() {
    local dir="$1" label="$2" tag="${3:-root}"
    [[ -d "$dir" ]] || { msg "$YELLOW" "  WARN: Missing dir: $dir"; return 0; }
    local txt img
    txt="$(newest_file "$dir" \( -iname '*.txt' -o -iname '*.log' -o -iname '*.csv' -o -iname '*.json' -o -iname '*.ini' \) || true)"
    img="$(newest_file "$dir" \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.fits' -o -iname '*.fit' \) || true)"
    [[ -n "$txt" && -f "$txt" ]] && { cp -a -- "$txt" "$staging/latest_files/${label}__${tag}__TEXT__$(basename "$txt")"; msg "$GREEN" "  + latest text  -> $txt"; } || msg "$YELLOW" "  WARN: No text files found in: $dir"
    [[ -n "$img" && -f "$img" ]] && { cp -a -- "$img" "$staging/latest_files/${label}__${tag}__IMAGE__$(basename "$img")"; msg "$GREEN" "  + latest image -> $img"; } || msg "$YELLOW" "  WARN: No image files found in: $dir"
}

# with_newest_subdir_or_root <base> <callback> [args...]
with_newest_subdir_or_root() {
    local base="$1" cb="$2"; shift 2
    [[ -d "$base" ]] || { msg "$YELLOW" "  WARN: Missing dir: $base"; return; }
    local newest
    newest="$(newest_dir "$base" || true)"
    if [[ -n "$newest" ]]; then
        msg "$YELLOW" "  Using newest subfolder: $newest"
        "$cb" "$newest" "$(basename "$newest")" "$@"
    else
        msg "$YELLOW" "  Using root folder: $base"
        "$cb" "$base" "root" "$@"
    fi
}

# collect_output_files <dir> <tag>
collect_output_files() { copy_newest_files "$1" "output" 2 --tag "$2"; }

# collect_sdcard_pair <dir> <tag>
collect_sdcard_pair() { collect_latest_pair "$1" "sd_card_path" "$2"; }

# collect_inspection_images <dir>
collect_inspection_images() {
    local dir="$1"
    msg "$BLUE" "Collecting 3 newest inspection images..."
    copy_newest_files "$dir" "inspection_images" 3 \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.fits' -o -iname '*.fit' \)
}

# collect_course_focus_images
collect_course_focus_images() {
    local base_dir="$AUTOGAIN_DIR"
    msg "$BLUE" "Collecting course focus images..."

    [[ -d "$base_dir" ]] || { msg "$YELLOW" "  WARN: Missing autogain dir: $base_dir"; return 0; }

    local newest
    newest="$(find "$base_dir" -maxdepth 1 -type d -name '*coarse_focus_images' -printf '%T@ %p\n' \
        | sort -nr | head -n1 | cut -d' ' -f2-)"

    if [[ -n "$newest" ]]; then
        msg "$YELLOW" "  Using newest coarse_focus_images folder: $newest"
    else
        msg "$YELLOW" "  No coarse_focus_images folder found; skipping course focus images"
        return 0
    fi

    local target_dir="$staging/latest_files/course_focus_images"
    mkdir -p "$target_dir"

    local files=("focus_score.png" "diameters_score.png")
    for f in "${files[@]}"; do
        local path="$newest/$f"
        [[ -f $path ]] && { cp -a -- "$path" "$target_dir/"; msg "$GREEN" "  + $path"; } || msg "$YELLOW" "  WARN: Missing $f"
    done

    local summary
    summary="$(find "$newest" -maxdepth 1 -type f -iname 'autofocus_summary*.txt' | head -n1)"
    [[ -f "$summary" ]] && { cp -a -- "$summary" "$target_dir/"; msg "$GREEN" "  + $summary"; } || msg "$YELLOW" "  WARN: Missing autofocus_summary*.txt"
}

# ---------- Main ----------

collect_log_tails "$LOG_TAIL_LINES" "${LOGS[@]}"
copy_config_files "${CONFIG_FILES[@]}"

for cmd in "${PC_COMMANDS[@]}"; do
    read -r -a args <<< "$cmd"
    run_pc_command "${args[@]}"
done

msg "$BLUE" "Collecting output artifacts..."
with_newest_subdir_or_root "$OUTPUT_ROOT" collect_output_files

collect_inspection_images "$INSPECTION_DIR"
collect_course_focus_images

msg "$BLUE" "Collecting latest TEXT + IMAGE from sd_card_path..."
with_newest_subdir_or_root "$SDCARD_ROOT" collect_sdcard_pair

msg "$BLUE" "Creating zip: $zip_path"
(
    cd "$staging"
    zip -r -q "$zip_path" .
)

# Final messages after trap cleanup occurs automatically
msg "$GREEN" "Wrote: $zip_path"
msg "$GREEN" "Done."
