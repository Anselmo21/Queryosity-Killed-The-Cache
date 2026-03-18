from typing import NamedTuple


class QueryPageCount(NamedTuple):
    """A query string paired with its associated page count."""

    query: str
    """The SQL query text."""

    page_count: int
    """The number of pages associated with the query."""


type PageSet = set[QueryPageCount]
"""
A set of query-page count pairs from a single pg_buffercache profiling
sample.
"""


__all__ = [
    "QueryPageCount",
    "PageSet",
]
