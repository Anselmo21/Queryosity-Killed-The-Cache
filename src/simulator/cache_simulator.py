"""
Cache simulation using PostgreSQL's clock-sweep eviction algorithm.

PostgreSQL does not use pure LRU for its shared buffer pool.  Instead it
employs a clock-sweep (also called second-chance) algorithm:

* Each buffer frame carries a **usage count** (0 to ``MAX_USAGE_COUNT``).
* On a **hit** the usage count is incremented (capped at the maximum).
* On a **miss** a clock hand sweeps through the buffer pool, decrementing
  each frame's usage count.  The first frame whose count reaches 0 is
  evicted and replaced with the new page.

This module provides two granularities of clock-sweep cache:

* ``ClockSweepCache`` — table-level, used when only EXPLAIN-derived
  access profiles are available.
* ``PageClockSweepCache`` — individual-page-level, used when
  ``pg_buffercache`` profiling data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.simulator.access_profile import AccessProfile
from src.simulator.simulator_types import PageSet

# PostgreSQL caps the per-buffer usage count at 5 (see bufmgr.c).
MAX_USAGE_COUNT = 5


def encode_page_sets(
    page_sets: list[set[tuple[str, int]]],
) -> tuple[list[list[int]], dict[tuple[str, int], int]]:
    """
    Map (table, block) tuples to contiguous integers for faster hashing.

    Parameters
    ----------
    page_sets : list[set[tuple[str, int]]]
        Per-query page sets using (table, block) tuples.

    Returns
    -------
    tuple[list[list[int]], dict[tuple[str, int], int]]
        Integer-encoded page lists and the mapping used.
    """
    page_to_id: dict[tuple[str, int], int] = {}
    encoded: list[list[int]] = []
    for ps in page_sets:
        int_pages: list[int] = []
        for page in ps:
            if page not in page_to_id:
                page_to_id[page] = len(page_to_id)
            int_pages.append(page_to_id[page])
        encoded.append(int_pages)
    return encoded, page_to_id


@dataclass(frozen=True)
class SimulationResult:
    """
    Result of simulating a query schedule through the cache.

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


class ClockSweepCache:
    """
    Fixed-capacity clock-sweep cache operating at table granularity.

    Models PostgreSQL's shared-buffer eviction strategy.  Each cached
    table carries a usage count that is incremented on hits (up to
    ``MAX_USAGE_COUNT``) and decremented by the clock hand during
    eviction sweeps.  Tables whose page count exceeds the total cache
    capacity are never inserted.

    Parameters
    ----------
    capacity_pages : int
        Maximum number of pages the cache can hold.

    Raises
    ------
    ValueError
        If *capacity_pages* is negative.
    """

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        self.capacity = capacity_pages
        self._tables: list[str] = []
        self._pages: list[int] = []
        self._usage: list[int] = []
        self._lookup: dict[str, int] = {}
        self._used: int = 0
        self._hand: int = 0

    @property
    def used(self) -> int:
        """Number of pages currently held in the cache."""
        return self._used

    def _advance_hand(self) -> None:
        """Advance the clock hand, wrapping around the buffer pool."""
        if self._tables:
            self._hand = (self._hand + 1) % len(self._tables)

    def _evict_one(self) -> int:
        """
        Sweep until a frame with usage_count == 0 is found and evict it.

        Returns
        -------
        int
            Number of pages freed by the eviction.
        """
        while True:
            if self._usage[self._hand] == 0:
                freed = self._pages[self._hand]
                evicted_table = self._tables[self._hand]

                # Replace with last element to keep arrays compact
                last = len(self._tables) - 1
                if self._hand != last:
                    self._tables[self._hand] = self._tables[last]
                    self._pages[self._hand] = self._pages[last]
                    self._usage[self._hand] = self._usage[last]
                    self._lookup[self._tables[self._hand]] = self._hand

                del self._lookup[evicted_table]
                self._tables.pop()
                self._pages.pop()
                self._usage.pop()

                if self._tables:
                    self._hand %= len(self._tables)
                else:
                    self._hand = 0
                return freed

            self._usage[self._hand] -= 1
            self._advance_hand()

    def access(self, table: str, pages: int) -> bool:
        """
        Access a table in the cache.

        On a hit the table's usage count is incremented (capped at
        ``MAX_USAGE_COUNT``).  On a miss the clock hand sweeps to free
        enough space, then the table is inserted with a usage count of 1.

        Parameters
        ----------
        table : str
            Name of the table being accessed.
        pages : int
            Number of 8 KB pages the table occupies.

        Returns
        -------
        bool
            True if the table was already in the cache (hit).
        """
        if table in self._lookup:
            idx = self._lookup[table]
            if self._usage[idx] < MAX_USAGE_COUNT:
                self._usage[idx] += 1
            return True

        if pages > self.capacity:
            return False

        while self._used + pages > self.capacity and self._tables:
            self._used -= self._evict_one()

        slot = len(self._tables)
        self._tables.append(table)
        self._pages.append(pages)
        self._usage.append(1)
        self._lookup[table] = slot
        self._used += pages
        return False

    def reset(self) -> None:
        """Clear all entries from the cache."""
        self._tables.clear()
        self._pages.clear()
        self._usage.clear()
        self._lookup.clear()
        self._used = 0
        self._hand = 0


class PageClockSweepCache:
    """
    Fixed-capacity clock-sweep cache operating at individual page
    granularity.

    Models PostgreSQL's shared-buffer eviction strategy at the page
    level.  The buffer pool is a fixed-size array of slots.  Each slot
    stores an integer page ID and a usage count.  A clock hand sweeps
    to find eviction victims.

    Parameters
    ----------
    capacity_pages : int
        Maximum number of pages the cache can hold.

    Raises
    ------
    ValueError
        If *capacity_pages* is negative.
    """

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        self.capacity = capacity_pages
        self._page_ids: list[int] = []
        self._usage: list[int] = []
        self._lookup: dict[int, int] = {}
        self._hand: int = 0

    def access(self, page: int) -> bool:
        """
        Access a page in the cache.

        On a hit the page's usage count is incremented (capped at
        ``MAX_USAGE_COUNT``).  On a miss the clock hand sweeps until it
        finds a frame with usage_count == 0, evicts it, and inserts the
        new page with a usage count of 1.

        Parameters
        ----------
        page : int
            Integer page ID (from ``encode_page_sets``).

        Returns
        -------
        bool
            True if the page was already cached (hit).
        """
        if page in self._lookup:
            idx = self._lookup[page]
            if self._usage[idx] < MAX_USAGE_COUNT:
                self._usage[idx] += 1
            return True

        if self.capacity == 0:
            return False

        # Buffer pool not yet full — append directly
        if len(self._page_ids) < self.capacity:
            self._lookup[page] = len(self._page_ids)
            self._page_ids.append(page)
            self._usage.append(1)
            return False

        # Sweep for a victim
        while self._usage[self._hand] > 0:
            self._usage[self._hand] -= 1
            self._hand = (self._hand + 1) % self.capacity

        # Evict the victim at the current hand position
        old_page = self._page_ids[self._hand]
        del self._lookup[old_page]

        self._page_ids[self._hand] = page
        self._usage[self._hand] = 1
        self._lookup[page] = self._hand

        self._hand = (self._hand + 1) % self.capacity
        return False

    def reset(self) -> None:
        """Clear all entries from the cache."""
        self._page_ids.clear()
        self._usage.clear()
        self._lookup.clear()
        self._hand = 0


def simulate_schedule(
    profiles: list[AccessProfile],
    schedule: list[int],
    cache_capacity_pages: int,
) -> SimulationResult:
    """
    Simulate executing queries in the given order through a clock-sweep cache.

    Each query's table accesses are replayed through the cache.  A table
    access counts as a hit if the table is already cached, and as a miss
    otherwise.  The number of pages requested and served from cache are
    accumulated across all queries.

    Parameters
    ----------
    profiles : list[AccessProfile]
        Access profiles indexed by position.  Profile at index *i*
        corresponds to query index *i*.
    schedule : list[int]
        Permutation of ``range(len(profiles))`` specifying the execution
        order.
    cache_capacity_pages : int
        Cache capacity in 8 KB pages.

    Returns
    -------
    SimulationResult
        Aggregate page request and cache hit counts for the schedule.
    """
    cache = ClockSweepCache(cache_capacity_pages)
    total_requests = 0
    total_hits = 0

    for idx in schedule:
        profile = profiles[idx]
        for table, pages in profile.table_pages.items():
            total_requests += pages
            if cache.access(table, pages):
                total_hits += pages

    return SimulationResult(total_requests=total_requests, total_hits=total_hits)


def simulate_schedule_page_level(
    page_lists: list[list[int]],
    schedule: list[int],
    cache_capacity_pages: int,
) -> SimulationResult:
    """
    Simulate executing queries through a page-level clock-sweep cache.

    Uses integer-encoded page lists (from ``encode_page_sets``) for fast
    hashing and lookup.

    Parameters
    ----------
    page_lists : list[list[int]]
        Integer-encoded page lists indexed by position.  Each list
        contains page IDs representing the pages accessed by that query.
    schedule : list[int]
        Permutation of ``range(len(page_lists))`` specifying execution order.
    cache_capacity_pages : int
        Cache capacity in pages.

    Returns
    -------
    SimulationResult
        Aggregate page request and cache hit counts.
    """
    cache = PageClockSweepCache(cache_capacity_pages)
    total_requests = 0
    total_hits = 0

    for idx in schedule:
        for page in page_lists[idx]:
            total_requests += 1
            if cache.access(page):
                total_hits += 1

    return SimulationResult(total_requests=total_requests, total_hits=total_hits)
