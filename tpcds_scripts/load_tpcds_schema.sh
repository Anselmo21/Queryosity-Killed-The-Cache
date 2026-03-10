#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpcds}"

SCHEMA_SQL="${SCHEMA_SQL:-tpcds_schema.sql}"

echo "Loading schema '${SCHEMA_SQL}' into '${DB_NAME}'..."

cat "$SCHEMA_SQL" | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Done."