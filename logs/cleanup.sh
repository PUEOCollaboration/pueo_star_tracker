#!/bin/bash
# Pueo Startup Script by Milan (info@stubljar.com)
#   File: ~/Projects/pcc/logs/cleanup.sh

rm *.pid

rm -f *console.log
rm -f *console.log.*

rm -f telemetry.log
rm -f telemetry.log.*
rm -f debug-client.log
rm -f debug-client.log.*
rm -f debug-server.log
rm -f debug-server.log.*
rm -f debug-test.log
rm -f pcc_folder_stats.log

rm -rf test_logs