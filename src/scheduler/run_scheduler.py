"""
Run a query scheduler against a TPC-H or TPC-DS workload.

Usage
-----
    python -m src.scheduler.run_scheduler --workload tpch
    python -m src.scheduler.run_scheduler --workload tpcds --cache-pages 2000
    python -m src.scheduler.run_scheduler --workload tpch --generations 300 --pop 150 --seed 42

The script connects to PostgreSQL, collects EXPLAIN plans for every query in
the chosen workload, builds access profiles, runs the selected scheduling
algorithm, and prints the resulting schedule with its cache hit ratio vs. the
default (input) order.
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(level=logging.WARNING)

from src.postgres.connection import close_connection, create_connection
from src.scheduler.genetic_algorithm import GAConfig, GAScheduler
from src.simulator.access_profile import build_access_profiles_from_db
from src.simulator.cache_simulator import simulate_schedule
from src.utilities.configurations import (
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS, WORKLOAD_DIRS
from src.utilities.workload import load_queries


def _print_schedule(
    profiles,
    schedule: list[int],
    cache_pages: int,
    label: str,
) -> float:
    """
    Simulate a schedule and print its fitness summary.

    Parameters
    ----------
    profiles : list[AccessProfile]
        Access profiles for each query.
    schedule : list[int]
        Permutation of query indices.
    cache_pages : int
        LRU cache capacity in pages.
    label : str
        Header label for the printed output.

    Returns
    -------
    float
        Cache hit ratio for the schedule.
    """
    sim = simulate_schedule(profiles, schedule, cache_pages)
    ids = [profiles[i].query_id for i in schedule]
    print(f"\n{label}")
    print(f"  Order : {' → '.join(ids)}")
    print(f"  H_total / R_total : {sim.total_hits:,} / {sim.total_requests:,}")
    print(f"  F_hit : {sim.hit_ratio:.4f}  ({sim.hit_ratio * 100:.2f}%)")
    return sim.hit_ratio


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, build access profiles, and run the scheduler.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Uses sys.argv when None.
    """
    parser = argparse.ArgumentParser(description="Cache-aware query scheduler")
    parser.add_argument(
        "--workload",
        choices=list(WORKLOAD_DIRS),
        default="tpch",
        help="Query workload to schedule (default: tpch)",
    )
    parser.add_argument(
        "--cache-pages",
        type=int,
        default=1000,
        help="LRU cache capacity in 8 KB pages (default: 1000)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["ga"],
        default="ga",
        help="Scheduling algorithm to use (default: ga)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Number of GA generations (default: 200)",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=100,
        help="GA population size (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
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
    print(f"  Found {len(queries)} queries: {', '.join(queries)}")

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
        print("Building access profiles via EXPLAIN…")
        t0 = time.perf_counter()
        profiles = build_access_profiles_from_db(queries, conn, analyze=False)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s  ({len(profiles)} profiles)")

        total_pages = sum(p.total_pages for p in profiles)
        print(f"  Total estimated pages across all queries: {total_pages:,}")
        print(f"  Cache capacity: {args.cache_pages:,} pages")
        print(
            f"  Cache-to-workload ratio: "
            f"{args.cache_pages / max(total_pages, 1):.2%}"
        )

    finally:
        close_connection(conn)

    default_schedule = list(range(len(profiles)))
    baseline_fitness = _print_schedule(
        profiles, default_schedule, args.cache_pages, "Baseline (default order)",
    )

    # Build the scheduler based on --algorithm
    if args.algorithm == "ga":
        ga_config = GAConfig(
            population_size=args.pop,
            num_generations=args.generations,
            cache_capacity_pages=args.cache_pages,
            seed=args.seed,
        )

        def on_gen(gen: int, best: float) -> None:
            if gen % 50 == 0 or gen == ga_config.num_generations - 1:
                print(f"  gen {gen:4d}  best F_hit = {best:.4f}")

        scheduler = GAScheduler(config=ga_config, on_generation=on_gen)

        print(
            f"\nRunning GA  "
            f"(pop={ga_config.population_size}, gens={ga_config.num_generations})…"
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    t0 = time.perf_counter()
    result = scheduler.schedule(profiles)
    elapsed = time.perf_counter() - t0
    print(f"  Finished in {elapsed:.1f}s")

    ga_fitness = _print_schedule(
        profiles, result.best_schedule, args.cache_pages, "Best schedule",
    )

    improvement = ga_fitness - baseline_fitness
    print(f"\n{'─' * 48}")
    print(
        f"  Improvement over baseline : "
        f"{improvement:+.4f}  ({improvement * 100:+.2f}pp)"
    )
    if improvement > 0:
        print("  Scheduler found a better order.")
    elif improvement == 0:
        print("  Matched the baseline (already optimal or cache too large).")
    else:
        print(
            "  Baseline order was not beaten — consider tuning parameters "
            "or a smaller cache."
        )
    print(f"{'─' * 48}\n")


if __name__ == "__main__":
    main()
