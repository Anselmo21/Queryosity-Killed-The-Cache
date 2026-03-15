"""
Default configuration values for the query scheduler system.

Modify the values in this file to match your environment. Scripts read
from here so that connection and timeout settings only need to be
changed in one place.
"""

# PostgreSQL connection defaults
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_SCHEMA = "public"
PG_STATEMENT_TIMEOUT_MS = 600_000  # 0 = no timeout

# Docker container name (used for cache flushing via restart)
PG_CONTAINER_NAME = "query_scheduler_pg"

# Random seed for reproducible baseline ordering
BASELINE_SEED = 55
