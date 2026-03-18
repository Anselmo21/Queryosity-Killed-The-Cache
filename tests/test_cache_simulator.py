from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    ClockSweepCache,
    MAX_USAGE_COUNT,
    SimulationResult,
    simulate_schedule,
)


class TestClockSweepCache:
    def test_hit_on_second_access(self):
        cache = ClockSweepCache(capacity_pages=100)
        assert cache.access("t1", 10) is False  # miss
        assert cache.access("t1", 10) is True  # hit

    def test_eviction_when_full(self):
        cache = ClockSweepCache(capacity_pages=20)
        cache.access("t1", 10)  # miss, used=10
        cache.access("t2", 10)  # miss, used=20
        # t3 needs space; clock sweeps t1 (usage 1->0) then t2 (1->0),
        # then wraps and evicts t1 (usage==0).
        cache.access("t3", 10)  # miss, evicts one table
        # At least one of the earlier tables must have been evicted.
        evicted = (
            cache.access("t1", 10) is False
            or cache.access("t2", 10) is False
        )
        assert evicted

    def test_usage_count_protects_hot_entries(self):
        """Repeatedly accessed tables survive eviction sweeps."""
        cache = ClockSweepCache(capacity_pages=20)
        cache.access("hot", 10)
        # Bump hot's usage count several times
        for _ in range(4):
            cache.access("hot", 10)

        cache.access("cold", 10)  # fills cache

        # Insert a new table — clock must sweep through hot (high usage)
        # and cold (usage=1) before evicting.  cold should be evicted
        # because its usage count drops to 0 first.
        cache.access("new", 10)
        assert cache.access("hot", 10) is True  # survived
        assert cache.access("cold", 10) is False  # was evicted

    def test_usage_count_capped(self):
        """Usage count should never exceed MAX_USAGE_COUNT."""
        cache = ClockSweepCache(capacity_pages=100)
        cache.access("t1", 10)
        for _ in range(MAX_USAGE_COUNT + 5):
            cache.access("t1", 10)
        idx = cache._lookup["t1"]
        assert cache._usage[idx] == MAX_USAGE_COUNT

    def test_oversized_entry(self):
        cache = ClockSweepCache(capacity_pages=5)
        cache.access("t1", 3)
        # t_big exceeds capacity — not inserted, t1 survives
        assert cache.access("t_big", 10) is False
        assert cache.access("t1", 3) is True  # still cached

    def test_reset(self):
        cache = ClockSweepCache(capacity_pages=100)
        cache.access("t1", 10)
        cache.reset()
        assert cache.used == 0
        assert cache.access("t1", 10) is False  # miss after reset

    def test_zero_capacity(self):
        cache = ClockSweepCache(capacity_pages=0)
        assert cache.access("t1", 5) is False
        assert cache.access("t1", 5) is False  # never cached


class TestSimulationResult:
    def test_hit_ratio(self):
        r = SimulationResult(total_requests=100, total_hits=25)
        assert r.hit_ratio == 0.25

    def test_hit_ratio_zero_requests(self):
        r = SimulationResult(total_requests=0, total_hits=0)
        assert r.hit_ratio == 0.0


class TestSimulateSchedule:
    def _make_profiles(self) -> list[AccessProfile]:
        """Three queries: q0 and q2 share table A, q1 uses table B."""
        return [
            AccessProfile(query_id="q0", table_pages={"A": 10}),
            AccessProfile(query_id="q1", table_pages={"B": 10}),
            AccessProfile(query_id="q2", table_pages={"A": 10}),
        ]

    def test_adjacent_same_table_gives_hit(self):
        profiles = self._make_profiles()
        # q0 then q2: q2 hits on table A
        result = simulate_schedule(profiles, [0, 2, 1], cache_capacity_pages=100)
        assert result.total_hits == 10  # q2's access to A
        assert result.total_requests == 30

    def test_separated_same_table_may_miss(self):
        profiles = self._make_profiles()
        # q0, q1, q2 with tiny cache: table A evicted by B
        result = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=10)
        # cache holds only 10 pages: A loaded, then B needs space.
        # Clock sweep decrements A's usage (1->0), then evicts A.
        # q2 re-loads A → 0 hits.
        assert result.total_hits == 0

    def test_large_cache_all_hit(self):
        profiles = self._make_profiles()
        # large cache: both A and B fit, q2 hits A
        result = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=100)
        assert result.total_hits == 10
        assert result.hit_ratio == 10 / 30

    def test_identity_schedule(self):
        profiles = self._make_profiles()
        result = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=100)
        result2 = simulate_schedule(profiles, [0, 1, 2], cache_capacity_pages=100)
        assert result.total_hits == result2.total_hits  # deterministic

    def test_empty_schedule(self):
        result = simulate_schedule([], [], cache_capacity_pages=100)
        assert result.total_requests == 0
        assert result.total_hits == 0
