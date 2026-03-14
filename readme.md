# "Queryosity Killed The Cache" Query Scheduler

This repository contains the code and workloads used for the EECS 6414 (W2026) course project.

# System Prerequisites 

## 1. Install Docker

Docker is required to run the PostgreSQL database used by this project. 

Verify the installation:

```bash
docker --version
docker compose version
```

## 2. Launching PostgreSQL

Start the PG container from the repository root

```bash 
docker compose up -d 
```

Verify the container is running 

```bash 
docker ps 
```

If working properly, the expected container name will be **query_scheduler_pg** 

## 3. Install Libraries 

The list of packages required by the system is found within `requirements.txt`.

To use this, first create and activate a virtual environment

```bash 
python -m venv venv
source venv/bin/activate
```

Once inside the virtual environment, install the libraries as follow

```bash 
pip install -r requirements.txt
```

### Adding Dependencies

When there is a need to add new libraries, please follow these steps for consistency purposes.

1. Activate your virtual environment 
2. Install the packages you need 

```bash 
pip install psycopg[binary] pandas networkx
```
3. Update the dependency list (or alternatively you can just overwrite the text file directly)

```bash 
pip freeze > requirements.txt
```

4. Commit the updated requirements.txt

# Installing Benchmarks

**NOTE**: The instruction steps below were originally tested and conducted on the RHEL distribution. Some quirks may occur on other operating systems.
There shouldn't be any need to modify the parameters of the scripts involved unless you want a different name for the output directory.

## TPC-H

### 1. Clone the TPC-H Generator

Clone the TPC-H data generator:

```bash
git clone https://github.com/gregrahn/tpch-kit
cd tpch-kit/dbgen
```

Compile the generator:

```bash
make
```

Follow the instructions in the repository. 
The default configuration works for PostgreSQL.


### 2. Generate the Benchmark Data

Generate data with **Scale Factor 10** (or whichever you'd like):

```bash
./dbgen -s 10
```

The scale factor controls the dataset size.

This command generates the following `.tbl` files:

```
region.tbl
nation.tbl
supplier.tbl
customer.tbl
part.tbl
partsupp.tbl
orders.tbl
lineitem.tbl
```

Create a directory **outside this repository** and move the files there:

```bash
mkdir ../tpch-data-sf10
mv *.tbl ../tpch-data-sf10
```

Your directory structure should look like:

```
project-root/
│
├── this-repository/
│
└── tpch-data-sf10/
    region.tbl
    nation.tbl
    supplier.tbl
    customer.tbl
    part.tbl
    partsupp.tbl
    orders.tbl
    lineitem.tbl
```

### 3. Generate TPC-H Benchmark Queries

The TPC-H toolkit also provides a query generator.

Navigate to the generator directory:

```bash
cd tpch-kit/dbgen
```

Generate all 22 benchmark queries:

**Note**: The queries are already stored in this repository under `workloads/tpch`. But, if you are interested to generate, with perhaps different parameters, the following below will work. 

```bash
for i in {1..22}
do
    ./qgen $i > query$i.sql
done
```

### 4. Start PostgreSQL with Docker

See instructions above

### 5. Navigate to Setup Scripts

```bash
cd tpch_scripts
```

Make the scripts executable (if needed):

```bash
chmod +x *.sh
```

### 6. Configure Environment Variables

These variables tell the scripts where the container and dataset are located.

```bash
export CONTAINER_NAME=query_scheduler_pg
export POSTGRES_USER=postgres
export DB_NAME=tpch
export DATA_DIR=../../tpch-data-sf10
```

### 7. Run the Setup Script

Run the full database initialization:

```bash
./setup_tpch.sh
```

This script performs the following pipeline:

```
create_tpch_db.sh
        ↓
tpch_schema.sql
        ↓
load_tpch_data.sh
        ↓
tpch_pkeys.sql
        ↓
tpch_fkeys.sql
```

This will:

1. Create the `tpch` database
2. Load the schema
3. Import the TPC-H data
4. Apply primary keys
5. Apply foreign keys


### 8. Verify the Installation

Connect to PostgreSQL:

```bash
docker exec -it query_scheduler_pg psql -U postgres -d tpch
```

List tables:

```sql
\dt
```

Expected tables:

```
customer
lineitem
nation
orders
part
partsupp
region
supplier
```

Verify data volume:

```sql
SELECT COUNT(*) FROM lineitem;
```

For **SF10**, the result should be approximately:

```
~60,000,000 rows
```

## TPC-DS

### 1. Generate Data

Navigate to the `tpcds_scripts` directory.

If you are using **Linux**, the required executables are already included in the folder. Otherwise, clone the TPC-DS toolkit and build the generator for your system. 

```bash
git clone https://github.com/gregrahn/tpcds-kit
cd tpcds-kit/tools
make
```

Follow the instructions in the repository to compile the executables for your platform.

Data generation can be performed using the run_dsdgen_parallel.sh script. This script can spawn multiple children to speed up generation. Specify the desired scale factor and the output directory for the generated .dat files.

For consistency with the setup scripts, ensure your directory structure looks like this:

```
project-root/
│
├── this-repository/
│
└── tpcds-data-sf10/
```

### 2. Run the Setup Script 

Execute the TPC-DS setup script

```bash
./setup_tpcds.sh 
```

This will perform the same steps as the TPC-H setup.

# Running the Scheduler and Executor

The pipeline has two stages:

1. **Scheduler** (`src.scheduler.run_scheduler`) — simulates query execution over a model LRU cache and uses a genetic algorithm to find the execution order that maximizes cache hit ratio.
2. **Executor** (`src.executor.run_executor`) — executes queries against the real PostgreSQL instance using `EXPLAIN (ANALYZE, BUFFERS)` and reports actual shared buffer hit/read statistics.

The typical workflow is: run the scheduler to produce an optimized order, then pass that order to the executor to measure real-world cache performance.

---

## Scheduler

The genetic algorithm scheduler finds an execution order for a batch of queries that maximizes cache reuse under a simulated LRU buffer pool.

### Quick Start

Make sure the PostgreSQL container is running and the benchmark data is loaded (see above), then:

```bash
source venv/bin/activate
python -m src.scheduler.run_scheduler --workload tpch
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workload` | `tpch` | Query workload: `tpch`, `tpcds` (modified subset), or `tpcds_full` |
| `--cache-pages` | `1000` | Simulated LRU cache capacity in 8 KB pages |
| `--algorithm` | `ga` | Scheduling algorithm (`ga` = genetic algorithm) |
| `--generations` | `200` | Number of GA generations |
| `--pop` | `100` | GA population size |
| `--seed` | None | Random seed for reproducibility |

PostgreSQL connection settings are read from `src/utilities/configurations.py` and can be overridden with `--host`, `--port`, `--user`, `--password`, `--schema`, and `--timeout-ms`.

### Examples

```bash
# TPC-H with a smaller cache (forces more eviction, more room to optimize)
python -m src.scheduler.run_scheduler --workload tpch --cache-pages 500

# TPC-DS modified queries with more generations and a fixed seed
python -m src.scheduler.run_scheduler --workload tpcds --generations 300 --pop 150 --seed 42

# Full TPC-DS workload (99 queries)
python -m src.scheduler.run_scheduler --workload tpcds_full --cache-pages 2000
```

### Output

The script prints:

1. **Baseline** — simulated cache hit ratio for the default (file) order
2. **GA progress** — best fitness every 50 generations
3. **Best schedule** — the optimized query order and its simulated hit ratio
4. **Improvement** — change in cache hit ratio (percentage points) over the baseline

---

## Executor

The executor runs queries against the real PostgreSQL instance and reports actual shared buffer statistics. Use it to validate whether the scheduler's predicted ordering improves real cache performance.

### Quick Start

```bash
source venv/bin/activate

# Run in default order
python -m src.executor.run_executor --workload tpch

# Run in a scheduler-optimized order (copy the order from scheduler output)
python -m src.executor.run_executor --workload tpch --order q5,q12,q1,q3,...
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workload` | `tpch` | Query workload: `tpch`, `tpcds`, or `tpcds_full` |
| `--order` | None | Comma-separated query IDs (e.g. `q5,q3,q1`). Omit to use default file order |
| `--compare-baseline` | off | Also execute in default order and print a side-by-side comparison |
| `--container` | `query_scheduler_pg` | Docker container name used for cache flushing |

PostgreSQL connection settings are read from `src/utilities/configurations.py` and can be overridden with `--host`, `--port`, `--user`, `--password`, `--schema`, and `--timeout-ms`.

### Examples

```bash
# Compare default vs. optimized order
python -m src.executor.run_executor \
  --workload tpch \
  --order q5,q12,q1,q3,q14,q6 \
  --compare-baseline

# Run TPC-DS optimized order
python -m src.executor.run_executor --workload tpcds --order q17,q42,q7,...
```

### Output

The script prints a per-query table showing execution time, shared buffer hits, reads, and hit ratio, followed by totals. When `--compare-baseline` is used, it also prints the improvement in hit ratio and total execution time.

```
  Query           Time (ms)       Hits      Reads    Hit %
  ───────────────────────────────────────────────────────
  q1               1234.5      50,000     10,000   83.33%
  ...
  ───────────────────────────────────────────────────────
  TOTAL            9876.5     400,000     80,000   83.33%
```

### Cache Flushing

The buffer cache is **always flushed** before each schedule run by restarting the Docker container. This clears both PostgreSQL shared buffers and the OS page cache inside the container, ensuring cold-start and reproducible measurements every time.

---

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```
