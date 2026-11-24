#!/bin/bash

# Set default value for RUN_MODE
RUN_MODE="dev"

# Check if a command line argument was provided
if [ $# -eq 1 ]; then
    RUN_MODE=$1
fi

echo "Backend running in ${RUN_MODE} mode"

# DATASET_ROOTDIR, MODEL_ROOTDIR and TMP_ROOTDIR (temporary results) are this machine's
# paths, kept only in the repo's .env (see .env.example).
ENV_FILE="$(dirname "$0")/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Missing $ENV_FILE: copy .env.example to .env and set the data paths." >&2
    exit 1
fi
set -a
. "$ENV_FILE"
set +a
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
