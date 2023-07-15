#!/bin/bash

# Set default value for RUN_MODE
RUN_MODE="debug"

# Check if a command line argument was provided
if [ $# -eq 1 ]; then
    RUN_MODE=$1
fi

echo "Frontend running in ${RUN_MODE} mode"


if [ $RUN_MODE = "prod" ]; then
    # Run commands for prod mode
    export BACKEND_SERVICE="detection_vis_backend"    
    docker compose up --build detection_vis_ui 
else
    # Run commands for debug mode
    export BACKEND_SERVICE="localhost"
    streamlit run detection_vis_ui/app.py
fi