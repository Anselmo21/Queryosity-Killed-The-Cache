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

