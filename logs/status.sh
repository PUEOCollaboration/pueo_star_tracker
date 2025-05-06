#!/bin/bash
# Pueo Startup Script by Milan (info@stubljar.com)
#   File: ~/Projects/pcc/logs/status.sh

systemctl status vl_install.service
ps aux | grep -E "cedar-detect-server|pueo_star_camera_operation"
