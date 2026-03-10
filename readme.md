# Installing TPC-H Benchmarks

## 1. Install Docker

Install Docker on your local machine.

Verify the installation:

```bash
docker --version
docker compose version
```

---

## 2. Clone the TPC-H Generator

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

---

## 3. Generate the Benchmark Data

Generate data with **Scale Factor 10**:

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

---

## 4. Generate TPC-H Benchmark Queries

The TPC-H toolkit also provides a query generator.

Navigate to the generator directory:

```bash
cd tpch-kit/dbgen
```

Generate all 22 benchmark queries:

```bash
for i in {1..22}
do
    ./qgen $i > query$i.sql
done
```

## 5. Start PostgreSQL with Docker

From the repository root:

```bash
docker compose up -d
```

Verify the container is running:

```bash
docker ps
```

Expected container name:

```
query_scheduler_pg
```

---

## 6. Navigate to Setup Scripts

```bash
cd tpch_scripts
```

Make the scripts executable:

```bash
chmod +x *.sh
```

---

## 7. Configure Environment Variables

These variables tell the scripts where the container and dataset are located.

```bash
export CONTAINER_NAME=query_scheduler_pg
export POSTGRES_USER=postgres
export DB_NAME=tpch
export DATA_DIR=../../tpch-data-sf10
```

---

## 8. Run the Setup Script

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

---

## 9. Verify the Installation

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