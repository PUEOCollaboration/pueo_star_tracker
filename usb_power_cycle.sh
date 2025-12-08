#!/bin/bash
#make executable using: chmod +x usb_power_cycle_owl.sh
#run by: ./usb_power_cycle.sh
set -euo pipefail

# Owl SBC ONLY
if ! command -v outb >/dev/null 2>&1; then
  echo "ERROR: outb not found. Install with: sudo apt install ioport"
  exit 1
fi
#power off ports 2 and 4. 
sudo outb 0x1c91 0x04

sleep 5

# Power on all USB
sudo outb 0x1c91 0x00
