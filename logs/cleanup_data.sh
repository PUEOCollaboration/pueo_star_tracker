#!/bin/bash
# Data Cleanup Script by Milan (info@stubljar.com)
#   File: ~/Projects/pcc/logs/cleanup_data.sh

# Define the root directory of the app
APP_ROOT=~/Projects/pcc/

# List of folders to clean (relative to APP_ROOT)
SYMLINKS=(
    "autogain"
    "ssd_path"
    "sd_card_path"
    "output"
)

# Define colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Display warning message
echo -e "${RED}*****************************************************************"
echo -e "WARNING: THIS WILL DELETE ALL DATA (LOGS/IMAGES) ON SSD/SD CARDS!"
echo -e "*****************************************************************${NC}"
echo -e "${YELLOW}This operation will permanently remove:"
echo -e " - ALL files in symlinked directories:"
echo -e "   • autogain/       → $(readlink -f ~/Projects/pcc/autogain)"
echo -e "   • ssd_path/       → $(readlink -f ~/Projects/pcc/ssd_path)"
echo -e "   • sd_card_path/   → $(readlink -f ~/Projects/pcc/sd_card_path)"
echo -e "   • output/         → $(readlink -f ~/Projects/pcc/output)"
echo -e " - All log files in ~/Projects/pcc/logs/${NC}"

# Ask for confirmation
read -p "ARE YOU SURE YOU WANT TO DELETE ALL THIS DATA? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cleanup cancelled. No files were deleted.${NC}"
    exit 1
fi

# Change to the app root directory
echo "Change to app root directory: $APP_ROOT"
cd "$APP_ROOT" || exit

# Loop through each folder and clean its contents
for symlink in "${SYMLINKS[@]}"; do
    if [ -L "$symlink" ]; then
        TARGET=$(readlink -f "$symlink")
        echo -e "${RED}CLEANING: ${YELLOW}$symlink${RED} → ${YELLOW}$TARGET${NC}"

        # DELETE FROM TARGET LOCATION (not the symlink path)
        if [ -d "$TARGET" ]; then
            find "$TARGET" -mindepth 1 -delete
            echo "  Deleted $(find "$TARGET" | wc -l) items."
        else
            echo -e "${YELLOW}  Warning: Target $TARGET does not exist${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: $symlink is not a symlink - skipping${NC}"
    fi
done

# Also clean the log files in the root directory (from your original script)
echo -e "${RED}Cleaning log files...${NC}"
echo "Change to app logs directory: $APP_ROOT"
cd "$APP_ROOT/logs/" || exit
rm -f *.pid
rm -f telemetry.log
rm -f telemetry.log.*
rm -f debug-server.log
rm -f debug-server.log.*
rm -f pueo_console.log
rm -f pueo_console.log.*

echo -e "${YELLOW}Cleanup complete. All specified data has been deleted.${NC}"
echo -e "${YELLOW}Remaining space ssd: $(df -h /mnt/raid1/ | tail -1 | awk '{print $4}') free${NC}"
echo -e "${YELLOW}Remaining space sdcard: $(df -h /mnt/sdcard/ | tail -1 | awk '{print $4}') free${NC}"