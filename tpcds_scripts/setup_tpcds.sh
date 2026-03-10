#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpcds}"

DATA_DIR="${DATA_DIR:-../../tpcds-data-sf10}"
SCHEMA_SQL="${SCHEMA_SQL:-tpcds_schema.sql}"
KEYS_SQL="${KEYS_SQL:-tpcds_keys.sql}"

echo "Starting TPC-DS setup..."

echo "Step 1: Creating database"
CONTAINER_NAME="$CONTAINER_NAME" \
POSTGRES_USER="$POSTGRES_USER" \
DB_NAME="$DB_NAME" \
./create_tpcds_db.sh

echo "Step 2: Loading schema"
CONTAINER_NAME="$CONTAINER_NAME" \
POSTGRES_USER="$POSTGRES_USER" \
DB_NAME="$DB_NAME" \
SCHEMA_SQL="$SCHEMA_SQL" \
./load_tpcds_schema.sh

echo "Step 3: Loading data"
CONTAINER_NAME="$CONTAINER_NAME" \
POSTGRES_USER="$POSTGRES_USER" \
DB_NAME="$DB_NAME" \
DATA_DIR="$DATA_DIR" \
KEYS_SQL="$KEYS_SQL" \
./load_tpcds_data.sh

echo "TPC-DS setup complete."