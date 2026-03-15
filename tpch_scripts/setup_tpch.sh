#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpch}"
DATA_DIR="${DATA_DIR:-../../tpch-data-sf10}"

./create_tpch_db.sh

cat tpch_schema.sql | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

./load_tpch_data.sh

cat tpch_pkeys.sql | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

cat tpch_fkeys.sql | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

cat tpch_indexes.sql | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Analyzing tables…"
docker exec "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" -c "ANALYZE;"

echo "TPC-H setup complete."