"""
Workload loading utilities shared across scheduler and executor.
"""

from __future__ import annotations

import re
from typing import cast, LiteralString

from psycopg import sql

from src.utilities.constants import WORKLOAD_DIRS


def load_queries(workload: str) -> dict[str, sql.SQL]:
    """
    Load all SQL queries from the workload directory.

    Parameters
    ----------
    workload : str
        Name of the workload.  Must be a key in WORKLOAD_DIRS.

    Returns
    -------
    dict[str, SQL]
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

    queries: dict[str, sql.SQL] = {}
    for f in files:
        queries[f.stem] = sql.SQL(f.read_text()) # type: ignore
    return queries
