"""
Genetic algorithm-based query scheduler.

Uses tournament selection, order crossover (OX), swap mutation, and
elitism to evolve a population of query permutations that maximise
the simulated clock-sweep cache hit ratio.

Optimisations
-------------
* **Fitness memoization** — identical permutations (from elitism or
  convergent crossover) are not re-evaluated.
* **Approximate fitness via overlap matrix** — when page-level data is
  available, a precomputed pairwise overlap matrix reduces per-eval
  cost from O(total_pages) to O(n_queries · w).  The exact clock-sweep
  simulation is still used for the final best schedule.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from src.scheduler.base_scheduler import ScheduleResult, SchedulerBase
from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    SimulationResult,
    approximate_schedule_fitness,
    compute_overlap_matrix,
    simulate_schedule,
    simulate_schedule_page_level,
)


@dataclass
class GAConfig:
    """
    Configuration for the genetic algorithm scheduler.

    Attributes
    ----------
    population_size : int
        Number of individuals per generation.
    num_generations : int
        Maximum number of generations to evolve.
    crossover_rate : float
        Probability of applying crossover to a pair of parents.
    mutation_rate : float
        Probability of applying swap mutation to an offspring.
    tournament_size : int
        Number of candidates drawn for tournament selection.
    elite_count : int
        Number of top individuals preserved unchanged across generations.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages used for fitness evaluation.
    seed : int | None
        Random seed for reproducibility.  None means non-deterministic.
    use_approximate_fitness : bool
        When True and page-level data is available, use the precomputed
        overlap matrix for fast approximate fitness during evolution.
        The final best schedule is always validated with the exact
        clock-sweep simulation.
    """

    population_size: int = 100
    num_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    tournament_size: int = 3
    elite_count: int = 2
    cache_capacity_pages: int = 1000
    seed: int | None = None
    use_approximate_fitness: bool = False


def _fitness_exact(
    individual: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    page_sets: list[frozenset[int]] | None = None,
) -> float:
    """
    Evaluate the cache hit-ratio fitness of an individual using exact
    clock-sweep simulation.

    Parameters
    ----------
    individual : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        Cache capacity in 8 KB pages.
    page_sets : list[frozenset[int]] | None
        Integer-encoded page sets from pg_buffercache profiling.

    Returns
    -------
    float
        Cache hit ratio in the range [0.0, 1.0].
    """
    if page_sets is not None:
        result = simulate_schedule_page_level(
            page_sets, individual, cache_capacity_pages,
        )
    else:
        result = simulate_schedule(profiles, individual, cache_capacity_pages)
    return result.hit_ratio


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    """
    Select one individual from the population via k-tournament selection.

    Randomly samples k individuals and returns a copy of the one with the
    highest fitness.

    Parameters
    ----------
    population : list[list[int]]
        Current population of individuals.
    fitnesses : list[float]
        Fitness value for each individual in the population.
    k : int
        Number of candidates to draw for the tournament.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    list[int]
        Copy of the selected individual.
    """
    candidates = rng.sample(range(len(population)), k)
    winner = max(candidates, key=lambda i: fitnesses[i])
    return list(population[winner])


def _order_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: random.Random,
) -> list[int]:
    """
    Produce an offspring using Order Crossover (OX).

    A random substring is copied from parent1 into the child at the same
    positions.  The remaining positions are filled with elements from
    parent2, preserving their relative order.

    Parameters
    ----------
    parent1 : list[int]
        First parent permutation.
    parent2 : list[int]
        Second parent permutation.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    list[int]
        Offspring permutation.
    """
    n = len(parent1)
    start, end = sorted(rng.sample(range(n), 2))

    child = [-1] * n
    child[start : end + 1] = parent1[start : end + 1]

    inherited = set(child[start : end + 1])
    fill = [g for g in parent2 if g not in inherited]

    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[pos]
            pos += 1

    return child


def _swap_mutation(individual: list[int], rng: random.Random) -> None:
    """
    Apply swap mutation to an individual in-place.

    Two randomly chosen positions are swapped.

    Parameters
    ----------
    individual : list[int]
        Permutation to mutate.
    rng : random.Random
        Random number generator instance.
    """
    n = len(individual)
    i, j = rng.sample(range(n), 2)
    individual[i], individual[j] = individual[j], individual[i]


class GAScheduler(SchedulerBase):
    """
    Genetic algorithm-based query scheduler.

    Uses tournament selection, order crossover, swap mutation, and
    elitism to evolve a population of query permutations that maximise
    the simulated clock-sweep cache hit ratio.

    Parameters
    ----------
    config : GAConfig | None
        Algorithm parameters.  Defaults are used when None.
    on_generation : callable or None
        Optional callback invoked as ``on_generation(gen, best_fitness)``
        after each generation completes.
    """

    def __init__(
        self,
        config: GAConfig | None = None,
        on_generation: Callable[[int, float], None] | None = None,
    ) -> None:
        self.config = config or GAConfig()
        self.on_generation = on_generation

    def schedule(
        self,
        profiles: list[AccessProfile],
        page_sets: list[frozenset[int]] | None = None,
    ) -> ScheduleResult:
        """
        Run the genetic algorithm to find a cache-friendly query schedule.

        Parameters
        ----------
        profiles : list[AccessProfile]
            One access profile per query.
        page_sets : list[frozenset[int]] | None
            Integer-encoded page sets from pg_buffercache profiling.
            When provided, fitness is evaluated at page-level granularity.

        Returns
        -------
        ScheduleResult
            Best schedule, its fitness, simulation result, and fitness
            history.
        """
        return run_ga(
            profiles,
            self.config,
            on_generation=self.on_generation,
            page_sets=page_sets,
        )


def run_ga(
    profiles: list[AccessProfile],
    config: GAConfig | None = None,
    on_generation: Callable[[int, float], None] | None = None,
    page_sets: list[frozenset[int]] | None = None,
) -> ScheduleResult:
    """
    Run the genetic algorithm to find a cache-friendly query schedule.

    Evolves a population of permutations using tournament selection,
    order crossover, swap mutation, and elitism.  Fitness is the cache
    hit ratio obtained by simulating each schedule through a clock-sweep
    cache.

    When ``config.use_approximate_fitness`` is True and *page_sets* are
    provided, the precomputed pairwise overlap matrix is used for fast
    surrogate fitness during evolution.  The final best schedule is
    always validated with the exact clock-sweep simulation.

    Parameters
    ----------
    profiles : list[AccessProfile]
        One access profile per query.  The index in this list is the
        query's integer ID used in chromosomes.
    config : GAConfig | None
        Algorithm parameters.  Defaults are used when None.
    on_generation : callable or None
        Optional callback invoked as ``on_generation(gen, best_fitness)``
        after each generation completes.
    page_sets : list[frozenset[int]] | None
        Integer-encoded page sets from pg_buffercache profiling.  When
        provided, fitness is evaluated at page-level granularity.

    Returns
    -------
    ScheduleResult
        Best schedule, its fitness, full simulation result, and the
        per-generation fitness history.
    """
    if config is None:
        config = GAConfig()

    rng = random.Random(config.seed)
    n = len(profiles)
    cache_capacity_pages = config.cache_capacity_pages

    if n == 0:
        return ScheduleResult(
            best_schedule=[],
            best_fitness=0.0,
            best_simulation=SimulationResult(total_requests=0, total_hits=0),
        )

    if n == 1:
        if page_sets is not None:
            sim = simulate_schedule_page_level(page_sets, [0], cache_capacity_pages)
        else:
            sim = simulate_schedule(profiles, [0], cache_capacity_pages)
        return ScheduleResult(
            best_schedule=[0],
            best_fitness=sim.hit_ratio,
            best_simulation=sim,
            fitness_history=[sim.hit_ratio],
        )

    # --- Precompute overlap matrix for approximate fitness ----------------
    use_approx = config.use_approximate_fitness and page_sets is not None
    overlap_matrix: list[list[int]] | None = None
    page_counts: list[int] | None = None
    if use_approx:
        overlap_matrix = compute_overlap_matrix(page_sets)
        page_counts = [len(ps) for ps in page_sets]

    # --- Fitness evaluation with memoization ------------------------------
    fitness_cache: dict[tuple[int, ...], float] = {}

    def evaluate(individual: list[int]) -> float:
        """Evaluate fitness with memoization."""
        key = tuple(individual)
        cached = fitness_cache.get(key)
        if cached is not None:
            return cached

        if use_approx:
            f = approximate_schedule_fitness(
                overlap_matrix, page_counts, individual, cache_capacity_pages,
            )
        else:
            f = _fitness_exact(
                individual, profiles, cache_capacity_pages, page_sets,
            )
        fitness_cache[key] = f
        return f

    # --- Initial population -----------------------------------------------
    population: list[list[int]] = []
    for _ in range(config.population_size):
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(perm)

    fitnesses = [evaluate(ind) for ind in population]

    fitness_history: list[float] = []

    # --- Evolution loop ---------------------------------------------------
    for gen in range(config.num_generations):
        ranked = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )
        elites = [
            list(population[ranked[i]])
            for i in range(min(config.elite_count, len(population)))
        ]

        new_population: list[list[int]] = list(elites)
        new_fitnesses: list[float] = [
            fitnesses[ranked[i]] for i in range(len(elites))
        ]

        while len(new_population) < config.population_size:
            p1 = _tournament_select(
                population, fitnesses, config.tournament_size, rng,
            )
            p2 = _tournament_select(
                population, fitnesses, config.tournament_size, rng,
            )

            if rng.random() < config.crossover_rate:
                child = _order_crossover(p1, p2, rng)
            else:
                child = list(p1)

            if rng.random() < config.mutation_rate:
                _swap_mutation(child, rng)

            f = evaluate(child)
            new_population.append(child)
            new_fitnesses.append(f)

        population = new_population
        fitnesses = new_fitnesses

        best_gen_fitness = max(fitnesses)
        fitness_history.append(best_gen_fitness)

        if on_generation is not None:
            on_generation(gen, best_gen_fitness)

    # --- Final result (always exact simulation) ---------------------------
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_schedule = population[best_idx]
    if page_sets is not None:
        best_sim = simulate_schedule_page_level(
            page_sets, best_schedule, cache_capacity_pages,
        )
    else:
        best_sim = simulate_schedule(
            profiles, best_schedule, cache_capacity_pages,
        )

    return ScheduleResult(
        best_schedule=best_schedule,
        best_fitness=best_sim.hit_ratio,
        best_simulation=best_sim,
        fitness_history=fitness_history,
    )
