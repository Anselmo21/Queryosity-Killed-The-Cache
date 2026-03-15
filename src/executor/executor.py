from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    """
    Execution result for a single query.

    Attributes
    ----------
    query_id : str
        Identifier of the query.
    elapsed_ms : float
        Wall-clock execution time in milliseconds.
    shared_hit_blocks : int
        Buffer pool pages served from shared memory (cache hits).
    shared_read_blocks : int
        Buffer pool pages read from disk (cache misses).
    """

    query_id: str
    elapsed_ms: float
    shared_hit_blocks: int
    shared_read_blocks: int

    @property
    def total_blocks(self) -> int:
        """
        Total buffer pool pages accessed.
        """
        return self.shared_hit_blocks + self.shared_read_blocks

    @property
    def hit_ratio(self) -> float:
        """
        Cache hit ratio for this query.

        Returns 0.0 when no blocks were accessed.
        """
        if self.total_blocks == 0:
            return 0.0
        return self.shared_hit_blocks / self.total_blocks


@dataclass(frozen=True)
class ExecutionResult:
    """
    Aggregated result of executing a full schedule.

    Attributes
    ----------
    query_results : list[QueryResult]
        Per-query results in execution order.
    total_elapsed_ms : float
        Total wall-clock time for the entire schedule in milliseconds.
    total_shared_hit_blocks : int
        Total buffer pool hits across all queries.
    total_shared_read_blocks : int
        Total buffer pool reads across all queries.
    """

    query_results: list[QueryResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    total_shared_hit_blocks: int = 0
    total_shared_read_blocks: int = 0

    @property
    def total_blocks(self) -> int:
        """
        Total buffer pool pages accessed across all queries.
        """
        return self.total_shared_hit_blocks + self.total_shared_read_blocks

    @property
    def hit_ratio(self) -> float:
        """
        Overall cache hit ratio: total_shared_hit_blocks / total_blocks.

        Returns 0.0 when no blocks were accessed.
        """
        if self.total_blocks == 0:
            return 0.0
        return self.total_shared_hit_blocks / self.total_blocks

    @property
    def avg_hit_ratio(self) -> float:
        """
        Average cache hit ratio across all queries (unweighted).

        Returns 0.0 when there are no query results.
        """
        if not self.query_results:
            return 0.0
        return sum(qr.hit_ratio for qr in self.query_results) / len(self.query_results)


def _sum_blocks(plan_node: dict[str, Any]) -> tuple[int, int]:
    """
    Sum shared hit and read blocks from the top-level plan node.

    EXPLAIN ANALYZE reports cumulative block counts at the root node
    that include all child nodes, so only the root needs to be read.

    Parameters
    ----------
    plan_node : dict[str, Any]
        Root node of an EXPLAIN ANALYZE plan.

    Returns
    -------
    tuple[int, int]
        (shared_hit_blocks, shared_read_blocks).
    """
    hit = plan_node.get("Shared Hit Blocks", 0)
    read = plan_node.get("Shared Read Blocks", 0)
    return hit, read


def execute_schedule(
    queries: dict[str, str],
    schedule: list[str],
    connection: Connection,
) -> ExecutionResult:
    """
    Execute queries in the given order and collect real cache statistics.

    Each query is run via EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) so that
    actual shared buffer hit/read counts and execution times are captured
    without needing to parse result sets.

    Parameters
    ----------
    queries : dict[str, str]
        Mapping from query_id to SQL text.
    schedule : list[str]
        Ordered list of query_ids specifying the execution order.
    connection : Connection
        Active PostgreSQL connection.

    Returns
    -------
    ExecutionResult
        Per-query and aggregate execution statistics.
    """
    query_results: list[QueryResult] = []
    total_hit = 0
    total_read = 0

    t_start = time.perf_counter()

    for i, query_id in enumerate(schedule, 1):
        print(f"  [{i}/{len(schedule)}] Executing {query_id}…")
        sql = queries[query_id]
        explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"

        try:
            with connection.cursor() as cursor:
                t0 = time.perf_counter()
                cursor.execute(explain_sql)
                row = cursor.fetchone()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            logger.warning("Skipping %s: %s", query_id, exc)
            continue

        if row is None:
            logger.warning("Skipping %s: no plan returned", query_id)
            continue

        plan = row[0][0]
        plan_node = plan.get("Plan", plan)
        hit, read = _sum_blocks(plan_node)

        total_hit += hit
        total_read += read

        query_results.append(
            QueryResult(
                query_id=query_id,
                elapsed_ms=elapsed_ms,
                shared_hit_blocks=hit,
                shared_read_blocks=read,
            )
        )

    total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return ExecutionResult(
        query_results=query_results,
        total_elapsed_ms=total_elapsed_ms,
        total_shared_hit_blocks=total_hit,
        total_shared_read_blocks=total_read,
    )


def print_execution_result(result: ExecutionResult, label: str) -> None:
    """
    Print a formatted summary of a schedule execution.

    Parameters
    ----------
    result : ExecutionResult
        Result from execute_schedule.
    label : str
        Header label for the output.
    """
    print(f"\n{label}")
    print(f"  {'Query':<15} {'Time (ms)':>10} {'Hits':>10} {'Reads':>10} {'Hit %':>8}")
    print(f"  {'─' * 55}")
    for qr in result.query_results:
        print(
            f"  {qr.query_id:<15} {qr.elapsed_ms:>10.1f} "
            f"{qr.shared_hit_blocks:>10,} {qr.shared_read_blocks:>10,} "
            f"{qr.hit_ratio * 100:>7.2f}%"
        )
    print(f"  {'─' * 55}")
    print(
        f"  {'TOTAL':<15} {result.total_elapsed_ms:>10.1f} "
        f"{result.total_shared_hit_blocks:>10,} {result.total_shared_read_blocks:>10,} "
        f"{result.hit_ratio * 100:>7.2f}%"
    )
    print(f"  {'AVG HIT %':<15} {'':>10} {'':>10} {'':>10} {result.avg_hit_ratio * 100:>7.2f}%")
