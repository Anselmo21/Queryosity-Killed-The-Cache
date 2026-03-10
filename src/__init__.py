from __future__ import annotations
from typing import Any
from psycopg import Connection


def execute_query(
    query: str,
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
                    connection.commit()
                    return []

                rows = cursor.fetchall()
                connection.commit()
                return rows

            connection.commit()
            return None

    except Exception as exc:
        connection.rollback()
        raise RuntimeError("Failed to execute SQL statement.") from exc


def get_execution_plan(
    query: str,
    connection: Connection,
    execute_plan: bool,
) -> dict[str, Any]:
    """
    Retrieve the PostgreSQL execution plan for a SQL query.

    Parameters
    ----------
    query : str
        SQL query to explain.
    connection : Connection
        Active PostgreSQL connection.
    execute_plan : bool
        Whether the query should be executed while generating the plan.
        If True, EXPLAIN ANALYZE is used. Otherwise EXPLAIN is used.

    Returns
    -------
    dict[str, Any]
        PostgreSQL execution plan in JSON form.

    Raises
    ------
    RuntimeError
        If the execution plan cannot be retrieved.
    """
    explain_sql = (
        f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"
        if execute_plan
        else f"EXPLAIN (FORMAT JSON) {query}"
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(explain_sql)
            result = cursor.fetchone()

        if result is None:
            raise RuntimeError("No execution plan returned.")

        return result[0][0]

    except Exception as exc:
        raise RuntimeError("Failed to retrieve execution plan.") from exc