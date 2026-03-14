from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from psycopg import Connection

from src.postgres.execute import get_execution_plan

POSTGRES_PAGE_SIZE = 8192  # bytes


@dataclass(frozen=True)
class AccessProfile:
    """
    Estimated page-level access profile for a single query.

    Attributes
    ----------
    query_id : str
        Human-readable identifier for the query (e.g. "q1").
    table_pages : dict[str, int]
        Mapping from table name to the estimated number of 8 KB pages
        accessed by this query in that table.
    """

    query_id: str
    table_pages: dict[str, int] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        """
        Total number of pages accessed across all tables.
        """
        return sum(self.table_pages.values())


def _extract_relations(plan_node: dict[str, Any]) -> list[tuple[str, int]]:
    """
    Recursively extract relation names and estimated page counts from a
    PostgreSQL EXPLAIN plan tree.

    For nodes without block statistics, pages are estimated as
    ``ceil(Plan Rows * Plan Width / 8192)``.  For nodes with block
    statistics (EXPLAIN ANALYZE), pages are computed as
    ``Shared Hit Blocks + Shared Read Blocks``.

    Parameters
    ----------
    plan_node : dict[str, Any]
        A single node from a PostgreSQL EXPLAIN JSON plan tree.

    Returns
    -------
    list[tuple[str, int]]
        Pairs of (relation_name, estimated_pages) for every table-scanning
        node found in the subtree rooted at plan_node.
    """
    relations: list[tuple[str, int]] = []

    relation = plan_node.get("Relation Name")
    if relation is not None:
        hit_blocks = plan_node.get("Shared Hit Blocks")
        read_blocks = plan_node.get("Shared Read Blocks")

        if hit_blocks is not None and read_blocks is not None:
            pages = hit_blocks + read_blocks
        else:
            rows = plan_node.get("Plan Rows", 0)
            width = plan_node.get("Plan Width", 0)
            pages = max(1, math.ceil(rows * width / POSTGRES_PAGE_SIZE))

        relations.append((relation, pages))

    for child in plan_node.get("Plans", []):
        relations.extend(_extract_relations(child))

    return relations


def build_access_profile(
    query_id: str,
    plan: dict[str, Any],
) -> AccessProfile:
    """
    Build an AccessProfile from a PostgreSQL EXPLAIN JSON plan.

    If the same table appears in multiple plan nodes (e.g. a self-join),
    the maximum page count across those nodes is used, since the pages
    are expected to overlap in the buffer pool.

    Parameters
    ----------
    query_id : str
        Identifier for the query.
    plan : dict[str, Any]
        The top-level plan dict returned by ``get_execution_plan``.
        Expected to contain a ``"Plan"`` key.

    Returns
    -------
    AccessProfile
        Access profile with estimated per-table page counts.
    """
    plan_node = plan.get("Plan", plan)
    relations = _extract_relations(plan_node)

    table_pages: dict[str, int] = {}
    for table, pages in relations:
        table_pages[table] = max(table_pages.get(table, 0), pages)

    return AccessProfile(query_id=query_id, table_pages=table_pages)


def build_access_profiles_from_db(
    queries: dict[str, str],
    connection: Connection,
    analyze: bool = False,
) -> list[AccessProfile]:
    """
    Build access profiles for a batch of queries via EXPLAIN.

    Parameters
    ----------
    queries : dict[str, str]
        Mapping from query_id to SQL text.
    connection : Connection
        Active PostgreSQL connection.
    analyze : bool
        Whether to use EXPLAIN ANALYZE, which executes each query while
        generating its plan.  If False, only estimated statistics are used.

    Returns
    -------
    list[AccessProfile]
        One profile per successfully explained query, in iteration order
        of the input mapping.  Queries that fail EXPLAIN are skipped
        with a warning.
    """
    profiles: list[AccessProfile] = []
    for query_id, sql in queries.items():
        try:
            plan = get_execution_plan(sql, connection, analyze=analyze)
        except RuntimeError:
            logger.warning("Skipping %s: EXPLAIN failed", query_id)
            continue
        profiles.append(build_access_profile(query_id, plan))
    return profiles
