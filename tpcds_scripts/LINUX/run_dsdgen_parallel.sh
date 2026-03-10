#!/bin/bash
# TPC-DS data generation script for 100 GB scale, 12 parallel children
# Adjust SCALE, PARALLEL, and OUTPUT_DIR as needed.

OUTPUT_DIR=~/benchmarks/tpcds-data-sf10
SCALE=10
PARALLEL=12

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")" || exit 1

echo "Starting TPC-DS data generation"
echo "  SCALE: $SCALE"
echo "  PARALLEL CHILDREN: $PARALLEL"
echo "  OUTPUT DIR: $OUTPUT_DIR"
echo "-------------------------------------------"

for i in $(seq 1 "$PARALLEL"); do
  echo "Starting child $i..."
  ./dsdgen -DIR "$OUTPUT_DIR" \
           -SCALE "$SCALE" \
           -PARALLEL "$PARALLEL" \
           -CHILD "$i" \
           -VERBOSE Y -FORCE Y &
done

echo "Waiting for all $PARALLEL children to finish..."
wait
echo "All $PARALLEL children finished successfully."
