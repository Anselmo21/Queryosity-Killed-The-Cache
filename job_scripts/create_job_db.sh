#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-imdb}"
ADMIN_DB="${ADMIN_DB:-postgres}"

echo "Checking if database '$DB_NAME' exists..."

EXISTS=$(docker exec "$CONTAINER_NAME" \
  psql -U "$POSTGRES_USER" -d "$ADMIN_DB" -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';")

if [ "$EXISTS" = "1" ]; then
  echo "Database '$DB_NAME' already exists — skipping creation."
else
  echo "Creating database '$DB_NAME'..."
  docker exec "$CONTAINER_NAME" \
    psql -U "$POSTGRES_USER" -d "$ADMIN_DB" -c "CREATE DATABASE $DB_NAME;"
  echo "Database '$DB_NAME' created."
fi
