#!/bin/bash
# pci.sh - Interactive wrapper for PUEO CLI

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate virtual environment if found
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found at $SCRIPT_DIR/.venv"
fi

# Available commands help
show_help() {
    echo -e "\nAvailable PUEO Commands:"
    echo "  start               - Start autonomous mode"
    echo "  stop                - Stop autonomous mode"
    echo "  home_lens           - Home the lens"
    echo "  auto_focus <s> <e> <st> - Run autofocus (start end step)"
    echo "  get_[setting]       - Get current value (focus, exposure, etc)"
    echo "  set_[setting] <val> - Set value (focus, exposure, etc)"
    echo "  exit                - Quit this shell"
    echo -e "\nExamples:"
    echo "  set_focus 150"
    echo "  auto_focus 100 200 10"
    echo "  get_exposure"
}

# Main interactive loop
echo -e "\nPUEO Interactive Shell"
echo "Type 'help' for commands, 'exit' to quit"

while true; do
    # Prompt with current time
    printf "\n[$(date +%H:%M:%S)] pueo> "

    # Read command
    read -r -a cmd_args

    # Exit condition
    if [[ "${cmd_args[0]}" == "exit" ]]; then
        break
    fi

    # Help command
    if [[ "${cmd_args[0]}" == "help" ]]; then
        show_help
        continue
    fi

    # Skip empty input
    if [ -z "${cmd_args[0]}" ]; then
        continue
    fi

    # Execute command
    echo "➔ Executing: ${cmd_args[*]}"
    python3 "$SCRIPT_DIR/pueo_cli.py" "${cmd_args[@]}"
done

# Cleanup
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "✓ Virtual environment deactivated"
fi

echo -e "\nPUEO session ended. Goodbye!"