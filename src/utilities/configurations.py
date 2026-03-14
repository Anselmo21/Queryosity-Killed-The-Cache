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
PG_STATEMENT_TIMEOUT_MS = 120_000
