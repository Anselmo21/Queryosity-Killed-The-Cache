
from __future__ import annotations
from typing import Optional
from psycopg import Connection

import psycopg

_connection: Optional[Connection] = None


def create_connection(
    db_name: str,
    user: str,
    password: str,
    host: str,
    port: int,
    schema: str,
    statement_timeout_ms: int = 10000,
) -> Connection:
    """
    Create and initialize a PostgreSQL connection.

    This function establishes a new connection to a PostgreSQL database,
    configures the schema search path, and applies a statement timeout.

    Parameters
    ----------
    db_name : str
        Name of the PostgreSQL database.
    user : str
        Database user name used for authentication.
    password : str
        Password associated with the database user.
    host : str
        Hostname or IP address of the PostgreSQL server.
    port : int
        Port on which the PostgreSQL server is listening.
    schema : str
        Schema to set in the PostgreSQL search_path.
    statement_timeout_ms : int
        Maximum allowed execution time for SQL statements in milliseconds.

    Returns
    -------
    Connection
        An active psycopg PostgreSQL connection.

    Raises
    ------
    RuntimeError
        If the connection to the database cannot be established.
    """
    global _connection

    try:
        _connection = psycopg.connect(
            host=host,
            port=port,
            dbname=db_name,
            user=user,
            password=password,
            autocommit=False,
        )

        with _connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema};")
            cursor.execute(f"SET statement_timeout TO {statement_timeout_ms};")

        _connection.commit()
        return _connection

    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to PostgreSQL database '{db_name}'."
        ) from exc


def get_connection(
    db_name: str,
    user: str,
    password: str,
    host: str,
    port: int,
    schema: str,
    statement_timeout_ms: int = 10000,
) -> Connection:
    """
    Retrieve an active PostgreSQL connection.

    If no cached connection exists or the cached connection has been closed,
    a new connection will be created.

    Parameters
    ----------
    db_name : str
        Name of the PostgreSQL database.
    user : str
        Database user name used for authentication.
    password : str
        Password associated with the database user.
    host : str
        Hostname or IP address of the PostgreSQL server.
    port : int
        Port on which the PostgreSQL server is listening.
    schema : str
        Schema to set in the PostgreSQL search_path.
    statement_timeout_ms : int
        Maximum allowed execution time for SQL statements in milliseconds.

    Returns
    -------
    Connection
        A valid psycopg PostgreSQL connection.
    """
    global _connection

    if _connection is None or _connection.closed:
        _connection = create_connection(
            db_name=db_name,
            user=user,
            password=password,
            host=host,
            port=port,
            schema=schema,
            statement_timeout_ms=statement_timeout_ms,
        )

    return _connection


def close_connection() -> None:
    """
    Close the cached PostgreSQL connection.

    If a connection exists and is open, it will be closed and the
    internal cached reference will be cleared.
    """
    global _connection

    if _connection is not None and not _connection.closed:
        _connection.close()

    _connection = None