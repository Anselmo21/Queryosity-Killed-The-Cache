from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    ClockSweepCache,
    MAX_USAGE_COUNT,
    PageClockSweepCache,
    SimulationResult,
    approximate_schedule_fitness,
    compute_overlap_matrix,
    simulate_schedule,
    simulate_schedule_page_level,
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


class TestPageClockSweepBatchAccess:
    def test_batch_hits_and_misses(self):
        cache = PageClockSweepCache(capacity_pages=10)
        # First access — all misses
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 0

        # Second access — all hits
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 3

    def test_batch_partial_overlap(self):
        cache = PageClockSweepCache(capacity_pages=10)
        cache.batch_access(frozenset({0, 1, 2}))
        # Partial overlap: 0 and 1 are hits, 3 is a miss
        hits = cache.batch_access(frozenset({0, 1, 3}))
        assert hits == 2

    def test_batch_eviction(self):
        cache = PageClockSweepCache(capacity_pages=3)
        cache.batch_access(frozenset({0, 1, 2}))  # fills cache
        cache.batch_access(frozenset({3, 4, 5}))  # evicts all old pages
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 0  # all evicted

    def test_batch_consistent_with_individual(self):
        """batch_access should produce the same hit count as individual access calls."""
        pages = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({0, 4, 5}),
        ]

        # Batch path
        cache_batch = PageClockSweepCache(capacity_pages=5)
        batch_hits = sum(cache_batch.batch_access(ps) for ps in pages)

        # Individual path
        cache_individual = PageClockSweepCache(capacity_pages=5)
        individual_hits = 0
        for ps in pages:
            for page in ps:
                if cache_individual.access(page):
                    individual_hits += 1

        assert batch_hits == individual_hits

    def test_batch_zero_capacity(self):
        cache = PageClockSweepCache(capacity_pages=0)
        hits = cache.batch_access(frozenset({0, 1, 2}))
        assert hits == 0


class TestOverlapMatrix:
    def test_symmetric(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({0, 4, 5}),
        ]
        matrix = compute_overlap_matrix(page_sets)
        n = len(page_sets)
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == matrix[j][i]

    def test_overlap_values(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
            frozenset({0, 4, 5}),
        ]
        matrix = compute_overlap_matrix(page_sets)
        # {0,1,2} & {2,3,4} = {2} → 1
        assert matrix[0][1] == 1
        # {0,1,2} & {0,4,5} = {0} → 1
        assert matrix[0][2] == 1
        # {2,3,4} & {0,4,5} = {4} → 1
        assert matrix[1][2] == 1
        # Diagonal (self-overlap) is 0 by construction
        assert matrix[0][0] == 0

    def test_empty(self):
        assert compute_overlap_matrix([]) == []


class TestApproximateScheduleFitness:
    def test_basic_ordering(self):
        """Schedule that groups overlapping queries should score higher."""
        page_sets = [
            frozenset({0, 1, 2, 3, 4}),       # q0: 5 pages
            frozenset({0, 1, 2, 3, 4, 5, 6}), # q1: 7 pages, shares 5 with q0
            frozenset({10, 11, 12}),            # q2: 3 pages, no overlap
        ]
        page_counts = [len(ps) for ps in page_sets]
        matrix = compute_overlap_matrix(page_sets)

        # cache=6: only the immediate predecessor fits
        #   good [0,1,2]: k=1 (q1): overlap(q0,q1)=5, no discount → 5 hits
        #                  k=2 (q2): budget=6-7=-1 < 0 → 0 hits. total=5/15
        #   bad  [0,2,1]: k=1 (q2): overlap(q0,q2)=0 → 0 hits
        #                  k=2 (q1): budget=6-3=3 ≥ 0 → overlap(q2,q1)=0
        #                            budget=3-5=-2 < 0 → break → 0 hits. total=0/15
        good = approximate_schedule_fitness(
            matrix, page_counts, [0, 1, 2], cache_capacity_pages=6,
        )
        bad = approximate_schedule_fitness(
            matrix, page_counts, [0, 2, 1], cache_capacity_pages=6,
        )
        assert good > bad

    def test_empty_schedule(self):
        assert approximate_schedule_fitness([], [], [], 100) == 0.0

    def test_single_query(self):
        matrix = [[0]]
        page_counts = [10]
        # Single query — no predecessor, so 0 hits
        f = approximate_schedule_fitness(matrix, page_counts, [0], 100)
        assert f == 0.0


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


class TestSimulateSchedulePageLevel:
    def test_basic_hits(self):
        page_sets = [
            frozenset({0, 1, 2}),
            frozenset({2, 3, 4}),
        ]
        result = simulate_schedule_page_level(page_sets, [0, 1], cache_capacity_pages=10)
        # Page 2 is a hit for q1
        assert result.total_hits == 1
        assert result.total_requests == 6

    def test_empty(self):
        result = simulate_schedule_page_level([], [], cache_capacity_pages=10)
        assert result.total_requests == 0
        assert result.total_hits == 0
