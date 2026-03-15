"""
Execute queries against PostgreSQL in a specified order and report actual
buffer cache statistics.

The buffer cache is always flushed (via Docker container restart) before
each schedule run to ensure cold-start, reproducible measurements.

Usage
-----
    # Execute in default (natural) order:
    python -m src.executor.run_executor --workload tpch

    # Execute in a custom order:
    python -m src.executor.run_executor --workload tpch --order q3,q5,q1,q10

    # Compare baseline order vs. a GA-optimized order:
    python -m src.executor.run_executor --workload tpch --order q5,q3,q1 --compare-baseline
"""

from __future__ import annotations

import argparse
import logging
import random
import subprocess
import time

logging.basicConfig(level=logging.WARNING)

from psycopg import Connection

from src.executor.executor import execute_schedule, print_execution_result
from src.postgres.connection import close_connection, create_connection
from src.utilities.configurations import (
    BASELINE_SEED,
    PG_CONTAINER_NAME,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS, WORKLOAD_DIRS
from src.utilities.workload import load_queries


def flush_buffer_cache(container_name: str) -> None:
    """
    Flush the PostgreSQL buffer cache by restarting the Docker container.

    This clears both PostgreSQL shared buffers and the OS page cache
    inside the container, ensuring a cold-start state for the next run.

    Parameters
    ----------
    container_name : str
        Name of the Docker container running PostgreSQL.

    Raises
    ------
    RuntimeError
        If the container fails to restart.
    """
    print(f"  Flushing buffer cache (restarting container '{container_name}')…")
    result = subprocess.run(
        ["docker", "restart", container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to restart container '{container_name}': {result.stderr.strip()}"
        )

    _wait_for_pg(container_name)
    print("  Buffer cache flushed — PostgreSQL is ready.")


def _wait_for_pg(container_name: str, timeout_s: int = 30) -> None:
    """
    Block until PostgreSQL inside the container accepts connections.

    Parameters
    ----------
    container_name : str
        Name of the Docker container running PostgreSQL.
    timeout_s : int
        Maximum seconds to wait before raising.

    Raises
    ------
    RuntimeError
        If PostgreSQL does not become ready within the timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["docker", "exec", container_name, "pg_isready", "-U", "postgres"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError(
        f"PostgreSQL in '{container_name}' did not become ready within {timeout_s}s"
    )


def _connect(args, db_name: str) -> Connection:
    """
    Create a PostgreSQL connection from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing connection parameters.
    db_name : str
        Name of the database to connect to.

    Returns
    -------
    Connection
        Active PostgreSQL connection.
    """
    return create_connection(
        db_name=db_name,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        schema=args.schema,
        statement_timeout_ms=args.timeout_ms,
    )


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, execute queries, and print cache statistics.

    The buffer cache is flushed before each schedule run.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Uses sys.argv when None.
    """
    parser = argparse.ArgumentParser(
        description="Execute queries and report actual buffer cache statistics",
    )
    parser.add_argument(
        "--workload",
        choices=list(WORKLOAD_DIRS),
        default="tpch",
        help="Query workload (default: tpch)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="Comma-separated query IDs in desired execution order "
        "(e.g. q3,q1,q5). If omitted, uses natural file order.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also execute in default order for comparison",
    )
    parser.add_argument(
        "--container",
        default=PG_CONTAINER_NAME,
        help=f"Docker container name for cache flushing (default: {PG_CONTAINER_NAME})",
    )
    parser.add_argument("--host", default=PG_HOST)
    parser.add_argument("--port", type=int, default=PG_PORT)
    parser.add_argument("--user", default=PG_USER)
    parser.add_argument("--password", default=PG_PASSWORD)
    parser.add_argument("--schema", default=PG_SCHEMA)
    parser.add_argument("--timeout-ms", type=int, default=PG_STATEMENT_TIMEOUT_MS)
    args = parser.parse_args(argv)

    db_name = DB_DEFAULTS[args.workload]

    print(f"Loading {args.workload} queries…")
    queries = load_queries(args.workload)
    query_ids = list(queries.keys())
    print(f"  Found {len(queries)} queries")

    if args.order:
        schedule = [q.strip() for q in args.order.split(",")]
        missing = [q for q in schedule if q not in queries]
        if missing:
            parser.error(f"Unknown query IDs: {', '.join(missing)}")
    else:
        schedule = query_ids

    baseline = None

    if args.compare_baseline:
        rng = random.Random(BASELINE_SEED)
        random_order = list(query_ids)
        rng.shuffle(random_order)
        flush_buffer_cache(args.container)
        print(f"\nConnecting to database '{db_name}' at {args.host}:{args.port}…")
        conn = _connect(args, db_name)
        try:
            print("\nExecuting baseline (random order)…")
            print(f"  Order: {' → '.join(random_order)}")
            baseline = execute_schedule(queries, random_order, conn)
            print_execution_result(baseline, "Baseline (random order)")
        finally:
            close_connection(conn)

    flush_buffer_cache(args.container)
    print(f"\nConnecting to database '{db_name}' at {args.host}:{args.port}…")
    conn = _connect(args, db_name)
    try:
        label = "Custom order" if args.order else "Default order"
        print(f"\nExecuting {label.lower()}…")
        result = execute_schedule(queries, schedule, conn)
        print_execution_result(result, label)
    finally:
        close_connection(conn)

    if baseline is not None:
        improvement = result.hit_ratio - baseline.hit_ratio
        print(f"\n{'─' * 48}")
        print(
            f"  Hit ratio improvement : "
            f"{improvement:+.4f}  ({improvement * 100:+.2f}pp)"
        )
        time_diff = result.total_elapsed_ms - baseline.total_elapsed_ms
        print(f"  Time difference       : {time_diff:+.1f} ms")
        print(f"{'─' * 48}\n")


if __name__ == "__main__":
    main()
