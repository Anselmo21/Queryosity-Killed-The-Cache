#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-imdb}"
DATA_DIR="${DATA_DIR:-/home/adolores/imdb-data}"

echo "Starting JOB/IMDB setup..."

echo "Step 1: Creating database"
CONTAINER_NAME="$CONTAINER_NAME" \
POSTGRES_USER="$POSTGRES_USER" \
DB_NAME="$DB_NAME" \
./create_job_db.sh

echo "Step 2: Loading schema"
cat job_schema.sql | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Step 3: Loading data"
CONTAINER_NAME="$CONTAINER_NAME" \
POSTGRES_USER="$POSTGRES_USER" \
DB_NAME="$DB_NAME" \
DATA_DIR="$DATA_DIR" \
./load_job_data.sh

echo "Step 4: Creating FK indexes"
cat job_indexes.sql | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Step 5: Analyzing tables"
docker exec "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" -c "ANALYZE;"

echo "JOB/IMDB setup complete."
