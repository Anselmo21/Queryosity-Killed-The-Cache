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

get_table_files() {
  local table="$1"
  local matches=()
  local f

  f="${DATA_DIR}/${table}.dat"
  if [ -f "$f" ]; then
    matches+=( "$f" )
  fi

  for f in "${DATA_DIR}/${table}"_[0-9]*_[0-9]*.dat; do
    if [ -f "$f" ]; then
      matches+=( "$f" )
    fi
  done

  printf '%s\n' "${matches[@]}"
}

echo "Using data directory: ${DATA_DIR}"

for table in "${TABLES[@]}"; do
  mapfile -t table_files < <(get_table_files "$table")

  if [ "${#table_files[@]}" -eq 0 ]; then
    echo "Missing data files for table: ${table}"
    exit 1
  fi
done

echo "Loading TPC-DS tables..."
for table in "${TABLES[@]}"; do
  echo "Loading ${table}..."

  mapfile -t valid_files < <(get_table_files "$table")

  if [ "${#valid_files[@]}" -eq 0 ]; then
    echo "No data files for ${table}"
    continue
  fi

  for f in "${valid_files[@]}"; do
    echo "  -> Loading file: $(basename "$f")"

    sed 's/|$//' "$f" | docker exec -i "$CONTAINER_NAME" \
      psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
      -c "\copy ${table} FROM STDIN WITH (FORMAT text, DELIMITER '|', NULL '');"
  done
done

echo "Applying keys from '${KEYS_SQL}'..."
docker exec -i "$CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" < "$KEYS_SQL"

echo "TPC-DS data load complete."