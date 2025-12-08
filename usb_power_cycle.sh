#!/bin/bash
#make executable using: chmod +x usb_power_cycle_owl.sh
#run by: ./usb_power_cycle.sh
set -euo pipefail

# Owl SBC ONLY
# Power down all USB
if ! command -v outb >/dev/null 2>&1; then
  echo "ERROR: outb not found. Install with: sudo apt install ioport"
  exit 1
fi

sudo outb 0x1c91 0x07

sleep 5

# Power on all USB
sudo outb 0x1c91 0x00
