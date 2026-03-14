"""
Workload loading utilities shared across scheduler and executor.
"""

from __future__ import annotations

import re

from src.utilities.constants import WORKLOAD_DIRS


def load_queries(workload: str) -> dict[str, str]:
    """
    Load all SQL queries from the workload directory.

    Parameters
    ----------
    workload : str
        Name of the workload.  Must be a key in WORKLOAD_DIRS.

    Returns
    -------
    dict[str, str]
        Mapping from query identifier (file stem) to SQL text.

    Raises
    ------
    FileNotFoundError
        If the workload directory does not exist or contains no SQL files.
    """
    directory = WORKLOAD_DIRS[workload]
    if not directory.exists():
        raise FileNotFoundError(f"Workload directory not found: {directory}")

    def _natural_sort_key(path):
        """Sort 'query2' before 'query10' by splitting on digit boundaries."""
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)
        ]

    files = sorted(directory.glob("*.sql"), key=_natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .sql files found in {directory}")

    queries: dict[str, str] = {}
    for f in files:
        queries[f.stem] = f.read_text()
    return queries
