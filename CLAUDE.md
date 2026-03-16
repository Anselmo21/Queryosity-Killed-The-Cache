# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PostgreSQL query scheduler research project for EECS 6414 (W2026). It benchmarks query execution using TPC-H and TPC-DS workloads against a PostgreSQL 16 instance running in Docker.

## Coding Style
The code generated should have the same python doc style as all of the functions. Use types when necessary 

## Environment Setup

```bash
# Start PostgreSQL container (named query_scheduler_pg)
docker compose up -d

# Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

PostgreSQL defaults: host=`localhost`, port=`5432`, user=`postgres`, password=`postgres`. The Docker default database is `tpcds`; TPC-H uses a separate `tpch` database.

## Managing Dependencies

```bash
pip install <package>
pip freeze > requirements.txt
```

## Code Architecture

All source code lives under `src/`:

- **`src/postgres/connection.py`** — `create_connection()` opens a psycopg connection with `autocommit=True` and sets `search_path` and `statement_timeout`. `close_connection()` closes it safely.
- **`src/postgres/execute.py`** — `execute_query()` runs a SQL string and optionally fetches rows. `get_execution_plan()` runs `EXPLAIN (FORMAT JSON)` or `EXPLAIN (ANALYZE, FORMAT JSON)` and returns the plan as a dict.
- **`src/utilities/`** — Utility helpers (currently a placeholder).

## Workloads

- `workloads/tpch/` — 22 standard TPC-H queries (q1.sql–q22.sql)
- `workloads/tpcds/` — TPC-DS queries; `workloads/tpcds/modified/` contains adapted variants

## Benchmark Database Setup

### TPC-H

```bash
cd tpch_scripts
chmod +x *.sh
export CONTAINER_NAME=query_scheduler_pg
export POSTGRES_USER=postgres
export DB_NAME=tpch
export DATA_DIR=../../tpch-data-sf10
./setup_tpch.sh
```

### TPC-DS

```bash
cd tpcds_scripts
./setup_tpcds.sh
```

Linux binaries for TPC-DS data generation (`dsdgen`, `dsqgen`) are pre-compiled in `tpcds_scripts/LINUX/`. Use `run_dsdgen_parallel.sh` for parallel data generation. For other platforms, build from [tpcds-kit](https://github.com/gregrahn/tpcds-kit).

Data directories are expected **outside** the repository (e.g., `../tpch-data-sf10/`, `../tpcds-data-sf10/`).
