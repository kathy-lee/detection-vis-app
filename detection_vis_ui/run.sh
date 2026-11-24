#!/bin/bash

# Set default value for RUN_MODE
RUN_MODE="dev"

# Check if a command line argument was provided
if [ $# -eq 1 ]; then
    RUN_MODE=$1
fi

echo "Frontend running in ${RUN_MODE} mode"

# MODEL_ROOTDIR is this machine's path, kept only in the repo's .env (see .env.example).
ENV_FILE="$(dirname "$0")/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Missing $ENV_FILE: copy .env.example to .env and set the data paths." >&2
    exit 1
fi
set -a
. "$ENV_FILE"
set +a

if [ $RUN_MODE = "prod" ]; then
    # Run commands for prod mode
    export BACKEND_SERVICE="detection_vis_backend"
    docker compose up --build detection_vis_ui 
else
    # Run commands for debug mode
    export BACKEND_SERVICE="localhost"
    streamlit run detection_vis_ui/app.py
fi