from __future__ import annotations

import psycopg
from psycopg import Connection, sql


def create_connection(
    db_name: str,
    user: str,
    password: str,
    host: str,
    port: int,
    schema: str,
    statement_timeout_ms: int,
) -> Connection:
    """
    Create and configure a PostgreSQL connection.

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
        Schema to set in the PostgreSQL search path.
    statement_timeout_ms : int
        Maximum allowed execution time per statement in milliseconds.
        0 disables the timeout entirely.

    Returns
    -------
    Connection
        An active psycopg PostgreSQL connection.

    Raises
    ------
    RuntimeError
        If the connection cannot be established or configured.
    """
    try:
        connection = psycopg.connect(
            host=host,
            port=port,
            dbname=db_name,
            user=user,
            password=password,
            autocommit=True,
        )

        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET search_path TO {};")
                    .format(sql.Identifier(schema))
            )
            cursor.execute(
                sql.SQL("SET statement_timeout TO {};")
                    .format(sql.Literal(statement_timeout_ms))
            )

        return connection

    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to PostgreSQL database '{db_name}'."
        ) from exc


def close_connection(connection: Connection) -> None:
    """
    Close a PostgreSQL connection.

    Parameters
    ----------
    connection : Connection
        Active PostgreSQL connection to close.
    """
    if not connection.closed:
        connection.close()
