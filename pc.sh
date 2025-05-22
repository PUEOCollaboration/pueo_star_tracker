#!/bin/bash

# pc.sh - Bash wrapper for PUEO CLI tool

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "Warning: Virtual environment not found at $SCRIPT_DIR/.venv/bin/activate"
    echo "Proceeding without virtual environment activation..."
fi

# Set any required environment variables
# export EXAMPLE_VAR="value"

# Execute the Python CLI with all passed arguments
python3 "$SCRIPT_DIR/pueo-cli.py" "$@"
EXIT_CODE=$?

# Deactivate the virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# Exit with the same code as the Python script
exit $EXIT_CODE