from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from src.scheduler.access_profile import AccessProfile


@dataclass(frozen=True)
class SimulationResult:
    """
    Result of simulating a query schedule through the LRU cache.

    Attributes
    ----------
    total_requests : int
        Total page requests across all queries in the schedule.
    total_hits : int
        Number of page requests served from cache.
    """

    total_requests: int
    total_hits: int

    @property
    def hit_ratio(self) -> float:
        """
        Cache hit ratio: total_hits / total_requests.

        Returns 0.0 when there are no requests.
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests


class LRUCache:
    """
    Fixed-capacity LRU cache operating at table granularity.

    Each entry is a table whose size is its page count.  Capacity is
    measured in pages.  Eviction removes the least-recently-used table(s)
    until the new entry fits.  Tables whose page count exceeds the total
    cache capacity are never inserted.

    Parameters
    ----------
    capacity_pages : int
        Maximum number of pages the cache can hold.

    Raises
    ------
    ValueError
        If capacity_pages is negative.
    """

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        self.capacity = capacity_pages
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._used: int = 0

    @property
    def used(self) -> int:
        """
        Number of pages currently held in the cache.
        """
        return self._used

    def access(self, table: str, pages: int) -> bool:
        """
        Access a table in the cache.

        If the table is already cached, it is touched (moved to
        most-recently-used) and the call returns True (hit).  Otherwise
        the table is inserted with LRU eviction as needed and the call
        returns False (miss).  Tables larger than the cache capacity are
        never inserted.

        Parameters
        ----------
        table : str
            Name of the table being accessed.
        pages : int
            Number of 8 KB pages the table occupies.

        Returns
        -------
        bool
            True if the table was already in the cache (hit), False
            otherwise (miss).
        """
        if table in self._entries:
            self._entries.move_to_end(table)
            return True

        if pages > self.capacity:
            return False

        while self._used + pages > self.capacity and self._entries:
            _, evicted_pages = self._entries.popitem(last=False)
            self._used -= evicted_pages

        self._entries[table] = pages
        self._used += pages
        return False

    def reset(self) -> None:
        """
        Clear all entries from the cache.
        """
        self._entries.clear()
        self._used = 0


def simulate_schedule(
    profiles: list[AccessProfile],
    schedule: list[int],
    cache_capacity_pages: int,
) -> SimulationResult:
    """
    Simulate executing queries in the given order through an LRU cache.

    Each query's table accesses are replayed through a fresh cache.  A
    table access counts as a hit if the table is already cached, and as
    a miss otherwise.  The number of pages requested and served from
    cache are accumulated across all queries.

    Parameters
    ----------
    profiles : list[AccessProfile]
        Access profiles indexed by position.  Profile at index i
        corresponds to query index i.
    schedule : list[int]
        Permutation of ``range(len(profiles))`` specifying the execution
        order.
    cache_capacity_pages : int
        LRU cache capacity in 8 KB pages.

    Returns
    -------
    SimulationResult
        Aggregate page request and cache hit counts for the schedule.
    """
    cache = LRUCache(cache_capacity_pages)
    total_requests = 0
    total_hits = 0

    for idx in schedule:
        profile = profiles[idx]
        for table, pages in profile.table_pages.items():
            total_requests += pages
            if cache.access(table, pages):
                total_hits += pages

    return SimulationResult(total_requests=total_requests, total_hits=total_hits)
