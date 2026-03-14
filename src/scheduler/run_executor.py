"""
Execute queries against PostgreSQL in a specified order and report actual
buffer cache statistics.

Usage
-----
    # Execute in default (natural) order:
    python -m src.scheduler.run_executor --workload tpch

    # Execute in a custom order:
    python -m src.scheduler.run_executor --workload tpch --order q3,q5,q1,q10

    # Compare baseline order vs. a GA-optimized order:
    python -m src.scheduler.run_executor --workload tpch --order q5,q3,q1 --compare-baseline

    # Drop OS and PG caches before each run for cold-start measurement:
    python -m src.scheduler.run_executor --workload tpch --drop-caches
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.WARNING)

from src.postgres.connection import close_connection, create_connection
from src.postgres.execute import execute_query
from src.scheduler.executor import execute_schedule, print_schedule_result
from src.scheduler.run_scheduler import load_queries
from src.utilities.configurations import (
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS, WORKLOAD_DIRS


def _drop_pg_caches(conn) -> None:
    """
    Discard PostgreSQL shared buffer contents.

    Parameters
    ----------
    conn : Connection
        Active PostgreSQL connection with sufficient privileges.
    """
    execute_query("DISCARD ALL", conn, fetch_results=False)


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, execute queries, and print cache statistics.

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
        "--drop-caches",
        action="store_true",
        help="Run DISCARD ALL before each execution to clear PG caches",
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

    print(f"\nConnecting to database '{db_name}' at {args.host}:{args.port}…")
    conn = create_connection(
        db_name=db_name,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        schema=args.schema,
        statement_timeout_ms=args.timeout_ms,
    )

    try:
        if args.compare_baseline:
            if args.drop_caches:
                _drop_pg_caches(conn)
            print("\nExecuting baseline (default order)…")
            baseline = execute_schedule(queries, query_ids, conn)
            print_schedule_result(baseline, "Baseline (default order)")

        if args.drop_caches:
            _drop_pg_caches(conn)

        label = "Custom order" if args.order else "Default order"
        print(f"\nExecuting {label.lower()}…")
        result = execute_schedule(queries, schedule, conn)
        print_schedule_result(result, label)

        if args.compare_baseline:
            improvement = result.hit_ratio - baseline.hit_ratio
            print(f"\n{'─' * 48}")
            print(
                f"  Hit ratio improvement : "
                f"{improvement:+.4f}  ({improvement * 100:+.2f}pp)"
            )
            time_diff = result.total_elapsed_ms - baseline.total_elapsed_ms
            print(f"  Time difference       : {time_diff:+.1f} ms")
            print(f"{'─' * 48}\n")

    finally:
        close_connection(conn)


if __name__ == "__main__":
    main()
