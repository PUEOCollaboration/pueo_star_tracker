#!/usr/bin/env bash
# pgz.sh – PUEO Update - Package git changes since a given date into a single zip for PUEO support.
# Usage: ./pgz.sh 2025-12-01
#  On Windows(PS): & "C:\Program Files\Git\bin\bash.exe" ./pgz.sh 2025-12-01 | tee logs/pgz.log
# Produces logs/pcc_update_<timestamp>.zip containing changed files.

set -euo pipefail

# ---------- Colors ----------
CYAN='\033[0;36m'    # Script / header
BLUE='\033[0;34m'    # Phase / section headers
WHITE='\033[0m'      # Default / normal
YELLOW='\033[1;33m'  # Warnings
RED='\033[0;31m'     # Errors
GREEN='\033[0;32m'   # Success / OK messages
NC='\033[0m'         # Reset / no color

msg() { local color="$1"; shift; echo -e "${color}$*${NC}"; }

# ---------- Arguments ----------
if [[ $# -ne 1 ]]; then
    msg "$RED" "Usage: $0 <YYYY-MM-DD>"
    exit 1
fi
since_date="$1"

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUT_DIR"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

ZIP_PATH="$OUT_DIR/pcc_update_${timestamp}.zip"

msg "$CYAN" "pgz.sh – PUEO Update - Packaging git changes since $since_date"
msg "$BLUE" "Gathering files changed since $since_date ..."

# ---------- Gather changed files ----------
cd "$SCRIPT_DIR"
# List files changed since given date
mapfile -t files < <(git log --since="$since_date" --name-only --pretty=format: | sort -u)

if [[ ${#files[@]} -eq 0 ]]; then
    msg "$YELLOW" "No files changed since $since_date"
    exit 0
fi

msg "$BLUE" "Copying files to staging ..."
for f in "${files[@]}"; do
    if [[ -f "$f" ]]; then
        mkdir -p "$staging/$(dirname "$f")"
        cp -a -- "$f" "$staging/$f"
        msg "$GREEN" "  + $f"
    else
        msg "$YELLOW" "  Skipping missing or deleted file: $f"
    fi
done

# ---------- Create zip ----------
msg "$BLUE" "Creating zip: $ZIP_PATH"

if command -v zip >/dev/null 2>&1; then
    (cd "$staging" && zip -r -q "$ZIP_PATH" .)
elif [[ -f "/c/Tools/7z-extra/7za.exe" ]]; then
    (cd "$staging" && "/c/Tools/7z-extra/7za.exe" a -tzip "$ZIP_PATH" ./*)
else
    msg "$RED" "ERROR: No zip utility found! Install zip or 7za.exe"
    exit 1
fi

msg "$GREEN" "Created zip: $ZIP_PATH"
msg "$GREEN" "Done."
