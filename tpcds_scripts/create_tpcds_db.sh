#!/usr/bin/env bash
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpcds}"
ADMIN_DB="${ADMIN_DB:-postgres}"

echo "Ensuring database '${DB_NAME}' exists in container '${CONTAINER_NAME}'..."

if ${DOCKER} exec -i "$CONTAINER_NAME" \
  psql -U "$POSTGRES_USER" -d "$ADMIN_DB" -tAc \
  "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" \
  | grep -q '^1$'; then
  echo "Database '${DB_NAME}' already exists."
else
  echo "Creating database '${DB_NAME}'..."
  ${DOCKER} exec -i "$CONTAINER_NAME" \
    psql -U "$POSTGRES_USER" -d "$ADMIN_DB" -v ON_ERROR_STOP=1 \
    -c "CREATE DATABASE \"$DB_NAME\";"
fi

echo "Done."