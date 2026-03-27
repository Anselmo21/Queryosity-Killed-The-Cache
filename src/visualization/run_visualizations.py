"""
CLI entry point for all visualizations.

Reads JSON files produced by serializers.py and generates all plots
into the plots/ directory.

Usage
-----
    # After running the scheduler:
    python -m src.visualization.run_visualizations --workload tpch

    # After running the executor with --compare-baseline:
    python -m src.visualization.run_visualizations --workload tpch --executor

    # Both:
    python -m src.visualization.run_visualizations --workload tpch --executor --scheduler
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIZ_DATA_DIR = Path("viz_data")


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            f"Run the scheduler/executor with serializers first."
        )


def run_scheduler_plots(workload: str) -> None:
    from src.visualization.fitness_curve import plot_fitness_curve
    from src.visualization.overlap_matrix import plot_page_overlap_matrix

    profiles_path  = VIZ_DATA_DIR / f"profiles_{workload}.json"
    fitness_path   = VIZ_DATA_DIR / f"fitness_history_{workload}.json"
    baseline_path  = VIZ_DATA_DIR / f"baseline_schedule_{workload}.json"
    ga_path        = VIZ_DATA_DIR / f"ga_schedule_{workload}.json"
    meta_path      = VIZ_DATA_DIR / f"scheduler_meta_{workload}.json"

    for p in [profiles_path, fitness_path, baseline_path, ga_path]:
        _require(p)

    profiles         = _load(profiles_path)
    fitness_history  = _load(fitness_path)
    baseline_sched   = _load(baseline_path)
    ga_sched         = _load(ga_path)
    meta             = _load(meta_path) if meta_path.exists() else {}
    early_stop_gen   = meta.get("early_stop_gen")

    query_ids      = [p["query_id"] for p in profiles]
    q_index        = {q: i for i, q in enumerate(query_ids)}
    baseline_ids   = [query_ids[i] for i in baseline_sched]
    ga_ids         = [query_ids[i] for i in ga_sched]

    print("  Fitness curve…")
    p = plot_fitness_curve(fitness_history, workload, early_stop_gen)
    print(f"        → {p}")

    print("  Page overlap matrix…")
    p = plot_page_overlap_matrix(profiles, ga_sched, workload)
    print(f"        → {p}")



def run_executor_plots(workload: str) -> None:
    from src.visualization.per_query_hit_ratio import plot_per_query_hit_ratio

    baseline_path = VIZ_DATA_DIR / f"baseline_results_{workload}.json"
    ga_path       = VIZ_DATA_DIR / f"ga_results_{workload}.json"

    for p in [baseline_path, ga_path]:
        _require(p)

    baseline_results = _load(baseline_path)
    ga_results       = _load(ga_path)

    print("  [1/2] Per-query hit ratio…")
    p = plot_per_query_hit_ratio(baseline_results, ga_results, workload)
    print(f"        → {p}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate all scheduler/executor visualizations"
    )
    parser.add_argument(
        "--workload", default="tpch",
        help="Workload name (default: tpch)"
    )
    parser.add_argument(
        "--scheduler", action="store_true", default=True,
        help="Generate scheduler plots (fitness, heatmap, overlap, sensitivity)"
    )
    parser.add_argument(
        "--executor", action="store_true", default=False,
        help="Generate executor plots (per-query hit ratio, cumulative I/O)"
    )

    args = parser.parse_args(argv)

    run_sched = args.scheduler
    run_exec  = args.executor

    if not run_sched and not run_exec:
        parser.error("Nothing to do — pass --scheduler and/or --executor")

    if run_sched:
        print(f"\nGenerating scheduler plots for '{args.workload}'…")
        run_scheduler_plots(args.workload)

    if run_exec:
        print(f"\nGenerating executor plots for '{args.workload}'…")
        run_executor_plots(args.workload)

    print(f"\nDone. Plots saved to plots/")


if __name__ == "__main__":
    main()