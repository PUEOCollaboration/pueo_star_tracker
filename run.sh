#!/bin/bash

# Activate the virtual environment (replace with your virtual environment name)
source .venv/bin/activate

# Set environment variables (if necessary for your script)
# For example:
# export CAMERA_IP="192.168.1.100"

# Execute your Python script
python3 pueo_star_camera_operation_code.py

# Deactivate the virtual environment (optional)
# To keep the environment active for interactive use, comment out this line
deactivate