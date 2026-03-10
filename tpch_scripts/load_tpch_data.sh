#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-query_scheduler_pg}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${DB_NAME:-tpch}"
DATA_DIR="${DATA_DIR:-../../tpch-data-sf10}"

FILES=(
  region.tbl
  nation.tbl
  part.tbl
  supplier.tbl
  partsupp.tbl
  customer.tbl
  orders.tbl
  lineitem.tbl
)

echo "Using data directory: ${DATA_DIR}"

for f in "${FILES[@]}"; do
  if [ ! -f "${DATA_DIR}/${f}" ]; then
    echo "Missing file: ${DATA_DIR}/${f}"
    exit 1
  fi
done

echo "Copying TPC-H data files into container..."
for f in "${FILES[@]}"; do
  docker cp "${DATA_DIR}/${f}" "${CONTAINER_NAME}:/tmp/${f}"
done

echo "Loading region..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy region from '/tmp/region.tbl' with (format csv, delimiter '|');"

echo "Loading nation..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy nation from '/tmp/nation.tbl' with (format csv, delimiter '|');"

echo "Loading part..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy part from '/tmp/part.tbl' with (format csv, delimiter '|');"

echo "Loading supplier..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy supplier from '/tmp/supplier.tbl' with (format csv, delimiter '|');"

echo "Loading partsupp..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy partsupp from '/tmp/partsupp.tbl' with (format csv, delimiter '|');"

echo "Loading customer..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy customer from '/tmp/customer.tbl' with (format csv, delimiter '|');"

echo "Loading orders..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy orders from '/tmp/orders.tbl' with (format csv, delimiter '|');"

echo "Loading lineitem..."
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$DB_NAME" \
  -c "\copy lineitem from '/tmp/lineitem.tbl' with (format csv, delimiter '|');"

echo "TPC-H data load complete."