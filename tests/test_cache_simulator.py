from src.scheduler.access_profile import AccessProfile
from src.scheduler.cache_simulator import LRUCache, SimulationResult, simulate_schedule


class TestLRUCache:
    def test_hit_on_second_access(self):
        cache = LRUCache(capacity_pages=100)
        assert cache.access("t1", 10) is False  # miss
        assert cache.access("t1", 10) is True  # hit

    def test_eviction_when_full(self):
        cache = LRUCache(capacity_pages=20)
        cache.access("t1", 10)  # miss, used=10
        cache.access("t2", 10)  # miss, used=20
        cache.access("t3", 10)  # miss, evicts t1, used=20
        assert cache.access("t1", 10) is False  # evicted
        assert cache.access("t3", 10) is True  # still there

    def test_lru_order_after_touch(self):
        cache = LRUCache(capacity_pages=30)
        cache.access("t1", 10)
        cache.access("t2", 10)
        cache.access("t3", 10)  # cache full: t1, t2, t3
        cache.access("t1", 10)  # touch t1 -> LRU order: t2, t3, t1
        cache.access("t4", 10)  # evicts t2 (LRU) -> t3, t1, t4
        assert cache.access("t2", 10) is False  # evicted by t4
        assert cache.access("t1", 10) is True  # still there (was touched)

    def test_oversized_entry(self):
        cache = LRUCache(capacity_pages=5)
        cache.access("t1", 3)
        # t_big exceeds capacity — not inserted, t1 survives
        assert cache.access("t_big", 10) is False
        assert cache.access("t1", 3) is True  # still cached

    def test_reset(self):
        cache = LRUCache(capacity_pages=100)
        cache.access("t1", 10)
        cache.reset()
        assert cache.used == 0
        assert cache.access("t1", 10) is False  # miss after reset

    def test_zero_capacity(self):
        cache = LRUCache(capacity_pages=0)
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
        # cache holds only 10 pages: A loaded, then B evicts A, then A reloaded
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
