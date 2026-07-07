#!/bin/bash

# Set default value for RUN_MODE
RUN_MODE="dev"

# Check if a command line argument was provided
if [ $# -eq 1 ]; then
    RUN_MODE=$1
fi

echo "Backend running in ${RUN_MODE} mode"

export MODEL_ROOTDIR="/home/kangle/dataset/trained_models"
export DATASET_ROOTDIR="/home/kangle/dataset"
export TMP_ROOTDIR="/home/kangle/dataset/tmp"
export LOGURU_LEVEL="INFO"

if [ $RUN_MODE = "prod" ]; then
    # Run commands for prod mode
    export DATABASE_SERVICE="db"
    export DATABASE_URL="postgresql://postgres:postgres@db/postgres"
    docker compose up --build db redis detection_vis_backend detection_vis_worker
else
    # Run commands for debug mode
    export DATABASE_SERVICE="localhost"
    export DATABASE_URL="postgresql://postgres:postgres@localhost/postgres"
    export REDIS_URL="redis://localhost:6379/0"
    docker compose up -d --build db redis
    while ! docker exec detection-vis-app-db-1 pg_isready -U postgres
    do
        echo "Waiting for postgres database..."
        sleep 1
    done
    alembic upgrade head
    # Start the RQ worker on the host so it has direct GPU access, then serve the API.
    CUDA_VISIBLE_DEVICES=0 python -m detection_vis_backend.jobs.worker &
    WORKER_PID=$!
    trap "kill $WORKER_PID" EXIT
    uvicorn detection_vis_backend.app:app --host 0.0.0.0 --port 8001
fi
