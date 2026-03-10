#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpcds}"
DATA_DIR="${DATA_DIR:-../../tpcds-data-sf10}"
KEYS_SQL="${KEYS_SQL:-tpcds_keys.sql}"

TABLES=(
  call_center
  catalog_page
  catalog_returns
  catalog_sales
  customer
  customer_address
  customer_demographics
  date_dim
  household_demographics
  income_band
  inventory
  item
  promotion
  reason
  ship_mode
  store
  store_returns
  store_sales
  time_dim
  warehouse
  web_page
  web_returns
  web_sales
  web_site
)

echo "Using data directory: ${DATA_DIR}"

for table in "${TABLES[@]}"; do
  files=( "${DATA_DIR}/${table}"*.dat )
  found=0

  for f in "${files[@]}"; do
    if [ -f "$f" ]; then
      found=1
      break
    fi
  done

  if [ "$found" -eq 0 ]; then
    echo "Missing data files for table: ${table}"
    exit 1
  fi
done

echo "Loading TPC-DS tables..."
for table in "${TABLES[@]}"; do
  echo "Loading ${table}..."

  files=( "${DATA_DIR}/${table}"*.dat )
  valid_files=()

  for f in "${files[@]}"; do
    if [ -f "$f" ]; then
      valid_files+=( "$f" )
    fi
  done

  if [ "${#valid_files[@]}" -eq 0 ]; then
    echo "No data files for ${table}"
    continue
  fi

  cat "${valid_files[@]}" | sed 's/|$//' | docker exec -i "$CONTAINER_NAME" \
    psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
    -c "\copy ${table} FROM STDIN WITH (FORMAT csv, DELIMITER '|');"
done

echo "Applying keys from '${KEYS_SQL}'..."
cat "$KEYS_SQL" | docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME"

echo "TPC-DS data load complete."