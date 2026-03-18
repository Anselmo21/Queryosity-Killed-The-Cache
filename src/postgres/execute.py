from __future__ import annotations

from typing import Any

from psycopg import Connection, sql


def execute_query(
    query: sql.SQL,
    connection: Connection,
    fetch_results: bool,
) -> list[Any] | None:
    """
    Execute a SQL statement.

    Parameters
    ----------
    query : str
        SQL statement to execute.
    connection : Connection
        Active PostgreSQL connection.
    fetch_results : bool
        Whether rows should be fetched and returned.

    Returns
    -------
    list[Any] | None
        Query result rows if requested, otherwise None.

    Raises
    ------
    RuntimeError
        If the SQL statement cannot be executed.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)

            if fetch_results:
                if cursor.description is None:
                    return []
                return cursor.fetchall()

            return None

    except Exception as exc:
        raise RuntimeError("Failed to execute SQL statement.") from exc


def get_execution_plan(
    query: sql.SQL,
    connection: Connection,
    analyze: bool,
) -> dict[str, Any]:
    """
    Retrieve the PostgreSQL execution plan for a SQL query.

    Parameters
    ----------
    query : SQL
        SQL query to explain.
    connection : Connection
        Active PostgreSQL connection.
    analyze : bool
        Whether to execute the query while generating the plan.
        If True, EXPLAIN ANALYZE is used. Otherwise EXPLAIN is used.

    Returns
    -------
    dict[str, Any]
        Full PostgreSQL execution plan document.

    Raises
    ------
    RuntimeError
        If the execution plan cannot be retrieved.
    """
    if analyze:
        explain_sql = sql.SQL("EXPLAIN (ANALYZE, FORMAT JSON) {}").format(query)
    else:
        explain_sql = sql.SQL("EXPLAIN (FORMAT JSON) {}").format(query)

    try:
        with connection.cursor() as cursor:
            cursor.execute(explain_sql)
            result = cursor.fetchone()

        if result is None:
            raise RuntimeError("No execution plan returned.")

        return result[0][0]

    except Exception as exc:
        raise RuntimeError("Failed to retrieve execution plan.") from exc
