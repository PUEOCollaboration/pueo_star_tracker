#!/bin/bash
# PUEO Server Status Tool v2.0
# Manages VL Driver, Services, and Processes with start/stop/restart

# Configuration
TERMINATION_DELAY=2
STARTUP_DELAY=5
STARTUP_SCRIPT="$HOME/scripts/startup_commands.sh"
DRIVER_NAME="vldrivep"
MODULES_CONF="/etc/modules-load.d/modules.conf"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Driver Check Functions ---

check_driver_installed() {
    # Check if driver module exists
    if [[ -f "/lib/modules/$(uname -r)/$DRIVER_NAME.ko" ]]; then
        # Check if loaded in kernel
        if lsmod | grep -q "$DRIVER_NAME"; then
            echo -e "${GREEN}Loaded${NC} (kernel module active)"
        # Check if configured for auto-load
        elif grep -q "^$DRIVER_NAME" "$MODULES_CONF" 2>/dev/null; then
            echo -e "${GREEN}Installed${NC} (will load on boot)"
        else
            echo -e "${YELLOW}Installed but not configured${NC}"
        fi
    else
        echo -e "${RED}Not Installed${NC}"
    fi
}

# --- Core Functions ---

show_help() {
    echo -e "${GREEN}PUEO Server Status Tool${NC}"
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  status       Show current status (default)"
    echo "  start        Stop (if running) and start services"
    echo "  stop         Kill all CEDAR/PUEO processes"
    echo "  restart      Stop + Start services"
    echo "  shutdown     Stop services and power off system"
    echo "  -h, --help   Show this help"
}

show_status() {
    echo "=== System Status ==="

    # 1. VL Driver Status
    echo -n "VL Driver: "
    check_driver_installed

    # 2. VL Install Service
    if systemctl is-enabled vl_install.service &>/dev/null; then
        if [[ $(systemctl show -p ExecMainStatus vl_install.service | cut -d= -f2) -eq 0 ]]; then
            echo -e "VL Install: ${GREEN}Completed Successfully${NC}"
        else
            echo -e "VL Install: ${YELLOW}Service Failed${NC}"
        fi
    else
        echo -e "VL Install: ${RED}Not Configured${NC}"
    fi

    # 3. Process Status
    cedar_pids=$(pgrep -f "cedar-detect-server" | tr '\n' ',' | sed 's/,$//')
    echo -e "CEDAR Detect: $([[ -n "$cedar_pids" ]] && echo "${GREEN}Running (PIDs: ${cedar_pids})${NC}" || echo "${RED}Not Running${NC}")"

    pueo_pids=$(pgrep -f "pueo_star_camera_operation" | tr '\n' ',' | sed 's/,$//')
    echo -e "PUEO Server: $([[ -n "$pueo_pids" ]] && echo "${GREEN}Running (PIDs: ${pueo_pids})${NC}" || echo "${RED}Not Running${NC}")"
}

stop_services() {
    echo -e "${YELLOW}Stopping services (delay: ${TERMINATION_DELAY}s)...${NC}"

    # Kill CEDAR
    cedar_pids=$(pgrep -f "cedar-detect-server")
    if [[ -n "$cedar_pids" ]]; then
        kill $cedar_pids 2>/dev/null
        echo -e "  CEDAR Detect: ${RED}Terminated PIDs: ${cedar_pids}${NC}"
    fi

    # Kill PUEO
    pueo_pids=$(pgrep -f "pueo_star_camera_operation")
    if [[ -n "$pueo_pids" ]]; then
        kill $pueo_pids 2>/dev/null
        echo -e "  PUEO Server: ${RED}Terminated PIDs: ${pueo_pids}${NC}"
    fi

    sleep $TERMINATION_DELAY
}

start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    if [[ -f "$STARTUP_SCRIPT" ]]; then
        bash "$STARTUP_SCRIPT"
        echo -e "  ${GREEN}Executed: $STARTUP_SCRIPT${NC}"

        # Added startup delay
        echo -e "${BLUE}Waiting ${STARTUP_DELAY}s for servers to initialize...${NC}"
        sleep $STARTUP_DELAY
    else
        echo -e "  ${RED}Error: Missing startup script ($STARTUP_SCRIPT)${NC}"
        exit 1
    fi
}

shutdown_server() {
    echo -e "${YELLOW}Initiating shutdown sequence...${NC}"
    stop_services
    echo -e "${RED}System will now power off${NC}"
    sudo shutdown now
}

# --- Command Handling ---
case "$1" in
    start)
        stop_services
        start_services
        echo -e "\n${GREEN}Final Status:${NC}"
        show_status
        ;;
    stop)
        stop_services
        echo -e "\n${GREEN}Final Status:${NC}"
        show_status
        ;;
    restart)
        stop_services
        start_services
        echo -e "\n${GREEN}Final Status:${NC}"
        show_status
        ;;
    shutdown)
        shutdown_server
        ;;
    status|"")
        show_status
        ;;
    -h|--help)
        show_help
        ;;
    *)
        echo -e "${RED}Error: Invalid command${NC}"
        show_help
        exit 1
        ;;
esac