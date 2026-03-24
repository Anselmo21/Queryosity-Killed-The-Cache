#  converts internal python objects into JSON 

"""
Serialization helpers for visualization integration.

Converts internal Python objects (AccessProfile, ScheduleResult,
ExecutionResult) into JSON files that the visualization scripts consume.

Usage in run_scheduler.py
-------------------------
    from src.visualization.serializers import dump_scheduler_data
    dump_scheduler_data(profiles, result, random_schedule, workload=args.workload)

Usage in run_executor.py
------------------------
    from src.visualization.serializers import dump_executor_data
    dump_executor_data(baseline_result, ga_result, workload=args.workload)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.scheduler.base_scheduler import ScheduleResult
from src.simulator.access_profile import AccessProfile
from src.executor.executor import ExecutionResult

VIZ_DATA_DIR = Path("viz_data")
VIZ_DATA_DIR.mkdir(exist_ok=True)


def dump_scheduler_data(
    profiles: list[AccessProfile],
    result: ScheduleResult,
    baseline_schedule: list[int],
    workload: str = "tpch",
    early_stop_gen: int | None = None,
) -> None:
    """
    Serialize scheduler outputs to JSON for the visualization scripts.

    Writes to viz_data/:
        profiles_{workload}.json
        fitness_history_{workload}.json
        baseline_schedule_{workload}.json
        ga_schedule_{workload}.json
        scheduler_meta_{workload}.json

    Parameters
    ----------
    profiles : list[AccessProfile]
        Access profiles built from EXPLAIN.
    result : ScheduleResult
        Output from GAScheduler.schedule().
    baseline_schedule : list[int]
        The random baseline permutation used for comparison.
    workload : str
        Workload name (used in filenames).
    early_stop_gen : int | None
        Generation at which early stopping fired, if applicable.
    """
    _write(f"profiles_{workload}.json",
           [_profile_to_dict(p) for p in profiles])
    _write(f"fitness_history_{workload}.json",
           result.fitness_history)
    _write(f"baseline_schedule_{workload}.json",
           baseline_schedule)
    _write(f"ga_schedule_{workload}.json",
           result.best_schedule)
    _write(f"scheduler_meta_{workload}.json",
           {"workload": workload, "early_stop_gen": early_stop_gen})

    print(f"\n  Visualization data saved to {VIZ_DATA_DIR}/")
    print(f"  Run: python -m src.visualization.run_visualizations --workload {workload}")


def dump_executor_data(
    baseline_result: ExecutionResult,
    ga_result: ExecutionResult,
    workload: str = "tpch",
) -> None:
    """
    Serialize executor outputs to JSON for the visualization scripts.

    Writes to viz_data/:
        baseline_results_{workload}.json
        ga_results_{workload}.json

    Parameters
    ----------
    baseline_result : ExecutionResult
        ExecutionResult from the baseline run.
    ga_result : ExecutionResult
        ExecutionResult from the GA-optimized run.
    workload : str
        Workload name.
    """
    _write(f"baseline_results_{workload}.json",
           [_query_result_to_dict(qr) for qr in baseline_result.query_results])
    _write(f"ga_results_{workload}.json",
           [_query_result_to_dict(qr) for qr in ga_result.query_results])

    print(f"\n  Executor visualization data saved to {VIZ_DATA_DIR}/")
    print(f"  Run: python -m src.visualization.run_visualizations --workload {workload}")


def _profile_to_dict(profile: AccessProfile) -> dict:
    return {
        "query_id": profile.query_id,
        "table_pages": dict(profile.table_pages),
    }


def _query_result_to_dict(qr) -> dict:
    return {
        "query_id": qr.query_id,
        "elapsed_ms": qr.elapsed_ms,
        "shared_hit_blocks": qr.shared_hit_blocks,
        "shared_read_blocks": qr.shared_read_blocks,
    }


def _write(filename: str, data) -> None:
    path = VIZ_DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)