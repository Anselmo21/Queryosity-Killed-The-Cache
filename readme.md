# "Queryosity Killed The Cache" Query Scheduler

This repository contains the code and workloads used for the EECS 6414 (W2026)
course project. It implements a cache-aware query scheduler on top of
PostgreSQL 16, using a genetic algorithm (GA) over a clock-sweep buffer-pool
simulator to find query execution orders that maximize shared-buffer hit
ratio. A DQN-based "SmartQueue" baseline is provided for comparison.

This README is a complete reproduction guide. Following it end-to-end will
get you from a blank machine to trained models, optimized schedules,
real-database execution numbers, and publication-ready plots.

---

## Pipeline Overview

```
           ┌──────────────────────┐
           │  Docker + Postgres   │  (Section 1)
           └─────────┬────────────┘
                     │
           ┌─────────▼────────────┐
           │  Benchmark loaders   │  TPC-H · TPC-DS · JOB/IMDB
           │  (tpch/tpcds/job)    │  (Section 2)
           └─────────┬────────────┘
                     │
           ┌─────────▼────────────┐
           │  Page profiler       │  pg_buffercache → CSV
           │  (src.profiler)      │  (Section 3)
           └─────────┬────────────┘
                     │
      ┌──────────────┴───────────────┐
      │                              │
┌─────▼──────────┐          ┌────────▼────────────┐
│  GA scheduler  │          │  DQN (SmartQueue)   │
│ (src.scheduler)│          │  ml/dqn.ipynb       │
└─────┬──────────┘          └────────┬────────────┘
      │                              │
      └──────────────┬───────────────┘
                     │
           ┌─────────▼────────────┐
           │  Executor (real PG)  │  EXPLAIN (ANALYZE, BUFFERS)
           │  (src.executor)      │  (Section 6)
           └─────────┬────────────┘
                     │
           ┌─────────▼────────────┐
           │  Visualizations      │  (Section 7)
           │  (src.visualization) │
           └──────────────────────┘
```

---

## 1. System Prerequisites

### 1.1 Docker

Docker is required for the PostgreSQL instance.

```bash
docker --version
docker compose version
```

### 1.2 Launch PostgreSQL

From the repository root:

```bash
docker compose up -d
docker ps   # should list a container named: query_scheduler_pg
```

Defaults: `host=localhost`, `port=5432`, `user=postgres`, `password=postgres`.

### 1.3 Python Environment

This project targets Python 3.9+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For DQN training you will also need a working PyTorch install. CPU-only
PyTorch is already in `requirements.txt`; for CUDA, follow the
[official instructions](https://pytorch.org/get-started/locally/).

### 1.4 Adding Dependencies

```bash
source venv/bin/activate
pip install <package>
pip freeze > requirements.txt
```

Commit the updated `requirements.txt`.

---

## 2. Installing Benchmarks

Three workloads are supported: **TPC-H**, **TPC-DS**, and **JOB** (the Join
Order Benchmark on the IMDB dataset). Only the workloads you intend to use
need to be loaded.

> **NOTE:** Setup scripts were tested on RHEL. Other Linux distros should work;
> macOS/Windows users may need to rebuild the TPC-DS binaries. Data
> directories are expected **outside** the repo (e.g. `../tpch-data-sf10/`).

### 2.1 TPC-H

#### Generate data

```bash
git clone https://github.com/gregrahn/tpch-kit
cd tpch-kit/dbgen
make
./dbgen -s 10          # Scale Factor 10
```

Move the generated `.tbl` files outside the repo:

```bash
mkdir ../../tpch-data-sf10
mv *.tbl ../../tpch-data-sf10
```

Expected project layout:

```
project-root/
├── Queryosity-Killed-The-Cache/
└── tpch-data-sf10/
    region.tbl  nation.tbl  supplier.tbl  customer.tbl
    part.tbl    partsupp.tbl  orders.tbl  lineitem.tbl
```

#### Load into Postgres

```bash
cd tpch_scripts
chmod +x *.sh
export CONTAINER_NAME=query_scheduler_pg
export POSTGRES_USER=postgres
export DB_NAME=tpch
export DATA_DIR=../../tpch-data-sf10
./setup_tpch.sh
```

#### Verify

```bash
docker exec -it query_scheduler_pg psql -U postgres -d tpch -c "SELECT COUNT(*) FROM lineitem;"
```

At SF10 this should return ~60M rows.

The 22 benchmark queries live in `workloads/tpch/`. If you need to regenerate
them with different parameters:

```bash
cd tpch-kit/dbgen
for i in {1..22}; do ./qgen $i > query$i.sql; done
```

### 2.2 TPC-DS

Linux `dsdgen`/`dsqgen` binaries are pre-compiled in `tpcds_scripts/LINUX/`.
For other platforms, build from [tpcds-kit](https://github.com/gregrahn/tpcds-kit).

#### Generate data

From `tpcds_scripts/`, use the parallel generator:

```bash
cd tpcds_scripts
./run_dsdgen_parallel.sh   # edit scale factor / output dir inside the script
```

Expected layout:

```
project-root/
├── Queryosity-Killed-The-Cache/
└── tpcds-data-sf10/
```

#### Load into Postgres

```bash
cd tpcds_scripts
./setup_tpcds.sh
```

This runs the same five-step pipeline as TPC-H (create DB → schema → load →
indexes → ANALYZE).

### 2.3 JOB / IMDB

The Join Order Benchmark runs against the IMDB dataset.

#### Get the data

Download the IMDB CSVs (the `imdb.tgz` archive published by the JOB authors)
and extract to a directory outside the repo, e.g. `~/imdb-data/`.

#### Load into Postgres

```bash
cd job_scripts
chmod +x *.sh
export CONTAINER_NAME=query_scheduler_pg
export POSTGRES_USER=postgres
export DB_NAME=imdb
export DATA_DIR=/absolute/path/to/imdb-data
./setup_job.sh
```

JOB queries live in `workloads/job/` (113 queries) and the workload is
registered as `job` in `src/utilities/constants.py`, pointing at the `imdb`
database. You can pass `--workload job` to the profiler and scheduler.

---

## 3. Page-Level Profiling (prerequisite for page-level sim & DQN training)

The scheduler can run in two modes:

- **Table-level** — uses estimated per-table page counts from `EXPLAIN`. No
  profiling needed; the scheduler falls back to this if no page data is
  present.
- **Page-level** — uses the exact set of 8 KB pages each query touches,
  captured via `pg_buffercache`. Produces much more accurate hit-ratio
  estimates and is required for DQN training.

### Run the profiler

```bash
source venv/bin/activate
python -m src.profiler.run_profiler --workload tpch
python -m src.profiler.run_profiler --workload tpcds
```

What happens:

1. `pg_buffercache` extension is created if missing.
2. For each query the Docker container is restarted (cold cache).
3. The query is executed and `pg_buffercache` is dumped.
4. A CSV of `(table, block)` pairs is written to `page_access/<workload>/`.

This is **slow** — it re-runs every query from cold cache. For TPC-DS this
can take hours. Do it once; results are reused by every downstream tool.

After profiling, the scheduler automatically picks up `page_access/<workload>/`
and switches to page-level simulation.

---

## 4. GA Scheduler

`src.scheduler.run_scheduler` runs a genetic algorithm over a clock-sweep
buffer-pool simulator to find a query order that maximizes cache hit ratio.

### Quick start

```bash
source venv/bin/activate
python -m src.scheduler.run_scheduler --workload tpch
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workload` | `tpch` | `tpch`, `tpcds`, or `job` (must be a key in `WORKLOAD_DIRS`) |
| `--cache-pages` | `1000` | Simulated buffer-pool size in 8 KB pages |
| `--algorithm` | `ga` | Scheduling algorithm (`ga` is the only option today) |
| `--generations` | `200` | GA generations |
| `--pop` | `100` | GA population size |
| `--seed` | `None` | Random seed for reproducibility |
| `--fitness` | `lru` | `lru` (cache simulation) or `dqn` (ONNX surrogate) |
| `--onnx-path` | `./dqn.onnx` | Path to exported DQN model (for `--fitness dqn`) |
| `--approximate` | off | Use fast overlap-matrix fitness during GA; exact sim for final result |

Postgres connection overrides: `--host --port --user --password --schema --timeout-ms`.

### Examples

```bash
# Smaller cache → more eviction pressure → more room for the scheduler to help
python -m src.scheduler.run_scheduler --workload tpch --cache-pages 500

# Larger GA run with a fixed seed
python -m src.scheduler.run_scheduler --workload tpcds --generations 300 --pop 150 --seed 42

# Fast approximate fitness during evolution (final score still uses exact sim)
python -m src.scheduler.run_scheduler --workload tpch --approximate

# DQN surrogate fitness (requires an ONNX model — see §8)
python -m src.scheduler.run_scheduler --workload tpch --fitness dqn --onnx-path ./dqn.onnx
```

### Output

The script prints:

1. **Baseline** — hit ratio for a random query order.
2. **GA progress** — best fitness every 50 generations.
3. **Best schedule** — the GA-optimized order and its simulated hit ratio.
4. **Improvement** — delta in percentage points over the baseline.

It also writes JSON artefacts to `viz_data/` that the visualization module
consumes (fitness history, profiles, schedules, metadata).

---

## 5. Executor (real-database measurement)

The scheduler uses a *model* of the cache. The executor runs queries against
the actual Postgres instance with `EXPLAIN (ANALYZE, BUFFERS)` and reports
real shared-buffer hits and reads.

### Quick start

```bash
source venv/bin/activate

# Run in default file order
python -m src.executor.run_executor --workload tpch

# Run a scheduler-optimized order
python -m src.executor.run_executor --workload tpch --order q5,q12,q1,q3,...
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workload` | `tpch` | `tpch`, `tpcds`, or `job` |
| `--order` | `None` | Comma-separated query IDs. Omit for file order. |
| `--compare-baseline` | off | Execute both orders and print a side-by-side comparison |
| `--container` | `query_scheduler_pg` | Docker container (used to flush cache) |

Postgres connection overrides as above.

### Example

```bash
python -m src.executor.run_executor \
  --workload tpch \
  --order q5,q12,q1,q3,q14,q6 \
  --compare-baseline
```

With `--compare-baseline`, two JSON result files land in `viz_data/` for
plotting (`baseline_results_<workload>.json`, `ga_results_<workload>.json`).

---

## 6. Visualizations

After running the scheduler and (optionally) the executor:

```bash
# Scheduler plots: fitness curve, page overlap matrix, cache sensitivity
python -m src.visualization.run_visualizations --workload tpch

# Executor plots: per-query hit ratio, cumulative I/O
python -m src.visualization.run_visualizations --workload tpch --executor

# Both
python -m src.visualization.run_visualizations --workload tpch --executor --scheduler
```

PNGs are written to `plots/`. Inputs come from the JSON files in `viz_data/`
produced by the scheduler and executor runs, so those must be run first.

---

## 7. SmartQueue / DQN Baseline

The `ml/` directory contains the Deep-Q-Network baseline used as a
comparison point against the GA scheduler. The training workflow is a
Jupytext-paired notebook: edit `ml/dqn.notebook.py`, open `ml/dqn.ipynb` in
Jupyter, or run the paired `.py` cells directly.

### 7.1 Prerequisites

- Python environment from §1.3 (PyTorch required — CUDA recommended).
- Page access CSVs for the target benchmark. Run §3 first:
  ```bash
  python -m src.profiler.run_profiler --workload tpch
  ```

### 7.2 Training

Launch Jupyter from the repository root:

```bash
source venv/bin/activate
jupyter lab     # or: jupyter notebook
```

Open `ml/dqn.ipynb` and run all cells. The first four cells prompt for:

| Prompt | Example | Meaning |
|--------|---------|---------|
| `Name` | `tpch-sf10` | Used to name the saved checkpoint `<Name>.pt` |
| `Benchmark` | `tpch` | Subdirectory of `page_access/` to read CSVs from (`tpch`, `tpcds`, `job`) |
| `Cache capacity in 8kb pages` | `1000` | Must match the cache size you will evaluate against |
| `Number of episodes` | `500` | DQN training episodes |

The notebook:

1. Reads `page_access/<benchmark>/q{1..22}.csv`.
2. Builds per-query page sets and a per-table block-count index.
3. Trains the DQN via `DQNTrainer.train(...)` (ε-greedy, target net, replay).
4. Saves the checkpoint to `ml/<Name>.pt`.
5. Plots training loss curves.
6. Runs a greedy scheduling loop: at each step, pick the query with the
   highest Q-value given the current cache state; simulate; repeat.
7. Prints the resulting schedule and average hit rate.

### 7.3 Reproducing the SmartQueue baseline numbers

1. Load and profile TPC-H at SF10 (§2.1, §3).
2. Open `ml/dqn.ipynb`.
3. Inputs: `Name=tpch-sf10-1000`, `Benchmark=tpch`, `Cache=1000`, `Episodes=500`.
4. Run all cells.
5. Record the final schedule and average hit rate printed by the last cell.
6. Compare against the GA result from §4 at the same cache capacity.

Repeat for TPC-DS and JOB by changing the `Benchmark` prompt (and adjusting
the `range(1, 23)` loop in the notebook if your workload has a different
query count).

### 7.4 Limitations / known gaps

- The notebook currently hard-codes the query range `range(1, 23)` (TPC-H's
  22 queries). For TPC-DS or JOB, edit that range to match the number of
  `q{N}.csv` files in `page_access/<benchmark>/`.
- The notebook saves a PyTorch `.pt` checkpoint only. The
  `--fitness dqn` path in `src.scheduler.run_scheduler` expects an **ONNX**
  file and uses a different (table-level) state encoding than the notebook's
  (page-level) encoding. Exporting the notebook's model to ONNX therefore
  will **not** plug directly into the GA scheduler — the DQN hook in the GA
  is experimental scaffolding, not a finished integration. Use the
  notebook's own scheduling loop to obtain SmartQueue baseline numbers.

---

## 8. Running the Full Pipeline (end-to-end example)

For a fresh checkout targeting TPC-H at SF10:

```bash
# 1. Infrastructure
docker compose up -d
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Benchmark data
cd tpch_scripts
export CONTAINER_NAME=query_scheduler_pg POSTGRES_USER=postgres \
       DB_NAME=tpch DATA_DIR=../../tpch-data-sf10
./setup_tpch.sh
cd ..

# 3. Page profiling (slow — do once)
python -m src.profiler.run_profiler --workload tpch

# 4. GA scheduler
python -m src.scheduler.run_scheduler --workload tpch --cache-pages 1000 --seed 42

# 5. Executor — measure on real Postgres
python -m src.executor.run_executor --workload tpch \
       --order <paste-ga-order-here> --compare-baseline

# 6. Plots
python -m src.visualization.run_visualizations --workload tpch --executor --scheduler

# 7. SmartQueue baseline (notebook)
jupyter lab ml/dqn.ipynb
```

---

## 9. Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

---

## 10. Repository Layout

```
src/
├── postgres/        connection & execute helpers
├── scheduler/       GA scheduler (run_scheduler, genetic_algorithm, …)
├── executor/        real-DB executor (run_executor)
├── profiler/        pg_buffercache page-access profiler
├── simulator/       clock-sweep cache simulator + DQN inference wrapper
├── utilities/       constants, configuration, workload loader
└── visualization/   plotting entry point + individual plot modules
ml/
├── dqn.ipynb        SmartQueue training notebook (paired via jupytext)
├── dqn.notebook.py  editable .py view of the notebook
└── dqntrainer.py    DQN architecture and training loop
workloads/
├── tpch/            22 TPC-H queries
└── tpcds/           adapted TPC-DS query subset
tpch_scripts/        TPC-H DB setup (schema, load, PK/FK)
tpcds_scripts/       TPC-DS DB setup (+ pre-built Linux dsdgen/dsqgen)
job_scripts/         JOB/IMDB DB setup
tests/               pytest suite
page_access/         (generated) pg_buffercache profiler output
viz_data/            (generated) scheduler/executor JSON artefacts
plots/               (generated) PNG output from visualizations
```
