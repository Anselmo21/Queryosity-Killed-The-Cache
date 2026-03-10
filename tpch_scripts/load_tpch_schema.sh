#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpch}"

SCHEMA_SQL="${SCHEMA_SQL:-schema.sql}"
PKEYS_SQL="${PKEYS_SQL:-pkeys.sql}"
FKEYS_SQL="${FKEYS_SQL:-fkeys.sql}"

echo "Loading schema into '${DB_NAME}'..."
cat "$SCHEMA_SQL" | docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Loading primary keys into '${DB_NAME}'..."
cat "$PKEYS_SQL" | docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Loading foreign keys into '${DB_NAME}'..."
cat "$FKEYS_SQL" | docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "Done."