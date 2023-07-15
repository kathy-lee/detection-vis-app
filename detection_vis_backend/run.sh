#!/bin/bash

# Set default value for RUN_MODE
RUN_MODE="dev"

# Check if a command line argument was provided
if [ $# -eq 1 ]; then
    RUN_MODE=$1
fi

echo "Backend running in ${RUN_MODE} mode"


if [ $RUN_MODE = "prod" ]; then
    # Run commands for prod mode
    export DATABASE_SERVICE="db"
    export DATABASE_URL="postgresql://postgres:postgres@db/postgres"
    docker compose up --build db detection_vis_backend
else
    # Run commands for debug mode
    export DATABASE_SERVICE="localhost"
    export DATABASE_URL="postgresql://postgres:postgres@localhost/postgres"
    docker compose up -d --build db
    alembic upgrade head
    uvicorn detection_vis_backend.app:app --host 0.0.0.0 --port 8001
fi
