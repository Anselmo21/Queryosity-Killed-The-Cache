#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-imdb}"
DATA_DIR="${DATA_DIR:-/home/adolores/imdb-data}"

TABLES=(
  aka_name
  aka_title
  cast_info
  char_name
  comp_cast_type
  company_name
  company_type
  complete_cast
  info_type
  keyword
  kind_type
  link_type
  movie_companies
  movie_info
  movie_info_idx
  movie_keyword
  movie_link
  name
  person_info
  role_type
  title
)

echo "Loading JOB/IMDB data from $DATA_DIR..."

for TABLE in "${TABLES[@]}"; do
  CSV_FILE="$DATA_DIR/${TABLE}.csv"
  if [ -f "$CSV_FILE" ]; then
    echo "  Loading $TABLE..."
    cat "$CSV_FILE" | docker exec -i "$CONTAINER_NAME" \
      psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
      -c "\\COPY $TABLE FROM STDIN WITH (FORMAT csv, HEADER false, DELIMITER ',', QUOTE '\"', ESCAPE '\\');"
  else
    echo "  WARNING: $CSV_FILE not found — skipping $TABLE"
  fi
done

echo "Data loading complete."
