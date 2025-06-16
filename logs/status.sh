#!/bin/bash
# PUEO Server Status Tool
# Manages VL Install, CEDAR Detect, and PUEO Server with start/stop/restart capabilities

# Configuration
TERMINATION_DELAY=2           # Configurable kill delay (seconds)
STARTUP_DELAY=5               # 5-second post-start delay for Python initialization
STARTUP_SCRIPT="$HOME/scripts/startup_commands.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Core Functions ---

show_help() {
    echo -e "${GREEN}PUEO Server Status Tool${NC}"
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  status       Show current status (default)"
    echo "  start        Stop (if running) and start services"
    echo "  stop         Kill all CEDAR/PUEO processes"
    echo "  restart      Stop + Start services"
    echo "  -h, --help   Show this help"
    echo -e "\nReports VL installation state and CEDAR/PUEO process status with PIDs"
}

show_status() {
    # VL Install Status
    if systemctl is-enabled vl_install.service &>/dev/null; then
        if [[ $(systemctl show -p ExecMainStatus vl_install.service | cut -d= -f2) -eq 0 ]]; then
            vl_status="${GREEN}Installed & Enabled (Success)${NC}"
        else
            vl_status="${YELLOW}Installed & Enabled (But Failed)${NC}"
        fi
    else
        vl_status="${RED}Not Installed${NC}"
    fi
    echo -e "VL Install: ${vl_status}"

    # CEDAR Status
    cedar_pids=$(pgrep -f "cedar-detect-server" | tr '\n' ',' | sed 's/,$//')
    if [[ -n "$cedar_pids" ]]; then
        cedar_status="${GREEN}Running (PIDs: ${cedar_pids})${NC}"
    else
        cedar_status="${RED}Not Running${NC}"
    fi
    echo -e "CEDAR Detect: ${cedar_status}"

    # PUEO Status
    pueo_pids=$(pgrep -f "pueo_star_camera_operation" | tr '\n' ',' | sed 's/,$//')
    if [[ -n "$pueo_pids" ]]; then
        pueo_status="${GREEN}Running (PIDs: ${pueo_pids})${NC}"
    else
        pueo_status="${RED}Not Running${NC}"
    fi
    echo -e "PUEO Server: ${pueo_status}"
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