"""
Genetic algorithm-based query scheduler.

Uses tournament selection, order crossover (OX), swap mutation, and
elitism to evolve a population of query permutations that maximise
the simulated LRU cache hit ratio.
"""

from __future__ import annotations

import random
from typing import Callable

from src.scheduler.base_scheduler import ScheduleResult, SchedulerBase
from src.scheduler.genetic_config import GAConfig
from src.scheduler.genetic_utils import (
    Individual,
    make_individual,
    select_parents,
)
from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    SimulationResult,
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
        LRU cache capacity in 8 KB pages used for fitness evaluation.
    seed : int | None
        Random seed for reproducibility.  None means non-deterministic.
    """

    population_size: int = 100
    num_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    tournament_size: int = 3
    elite_count: int = 2
    cache_capacity_pages: int = 1000
    seed: int | None = None


def _fitness(
    individual: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    page_lists: list[list[int]] | None = None,
) -> float:
    """
    Evaluate the cache hit-ratio fitness of an individual.

    When page_lists are provided, uses page-level LRU simulation.
    Otherwise falls back to table-level simulation.

    Parameters
    ----------
    individual : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        LRU cache capacity in 8 KB pages.
    page_lists : list[list[int]] | None
        Integer-encoded page lists from pg_buffercache profiling.

    Returns
    -------
    float
        Cache hit ratio in the range [0.0, 1.0].
    """
    if page_lists is not None:
        result = simulate_schedule_page_level(
            page_lists, individual, cache_capacity_pages,
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
    the simulated LRU cache hit ratio.

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
        page_lists: list[list[int]] | None = None,
    ) -> ScheduleResult:
        """
        Run the genetic algorithm to find a cache-friendly query schedule.

        Parameters
        ----------
        profiles : list[AccessProfile]
            One access profile per query.
        page_lists : list[list[int]] | None
            Integer-encoded page lists from pg_buffercache profiling.
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
            page_lists=page_lists,
        )


def run_ga(
    profiles: list[AccessProfile],
    config: GAConfig | None = None,
    on_generation: Callable[[int, float], None] | None = None,
    page_lists: list[list[int]] | None = None,
) -> ScheduleResult:
    """
    Run the genetic algorithm to find a cache-friendly query schedule.

    Evolves a population of permutations using tournament selection,
    order crossover, swap mutation, and elitism.  Fitness is the cache
    hit ratio obtained by simulating each schedule through an LRU cache.

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
    page_lists : list[list[int]] | None
        Integer-encoded page lists from pg_buffercache profiling.  When
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
        if page_lists is not None:
            sim = simulate_schedule_page_level(page_lists, [0], cache_capacity_pages)
        else:
            sim = simulate_schedule(profiles, [0], cache_capacity_pages)
        return ScheduleResult(
            best_schedule=[0],
            best_fitness=sim.hit_ratio,
            best_simulation=sim,
            fitness_history=[sim.hit_ratio],
        )

    population: list[Individual] = []
    for _ in range(config.population_size):
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(perm)

    fitnesses = [
        _fitness(ind, profiles, cache_capacity_pages, page_lists) for ind in population
    ]

    fitness_history: list[float] = []

    for gen in range(config.num_generations):
        ranked: list[Individual] = sorted(
            population,
            key=lambda individual: individual.fitness(),
            reverse=True,
        )
        elites: list[Individual] = ranked[:config.elite_count]

        new_population: list[Individual] = list(elites)

        # Generate children until population reaches the configured size
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

            f = _fitness(child, profiles, cache_capacity_pages, page_lists)
            new_population.append(child)

        population = new_population

        best_gen_fitness = max(individual.fitness() for individual in population)
        fitness_history.append(best_gen_fitness)

        if on_generation is not None:
            on_generation(gen, best_gen_fitness)

    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_schedule = population[best_idx]
    if page_lists is not None:
        best_sim = simulate_schedule_page_level(
            page_lists, best_schedule, cache_capacity_pages,
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
