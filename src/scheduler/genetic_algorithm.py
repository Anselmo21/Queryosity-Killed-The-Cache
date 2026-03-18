"""
Genetic algorithm-based query scheduler.

Uses tournament selection, order crossover (OX), swap mutation, and
elitism to evolve a population of query permutations that maximise
the simulated LRU cache hit ratio.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.scheduler.base_scheduler import ScheduleResult, SchedulerBase
from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    SimulationResult,
    simulate_schedule,
    simulate_schedule_page_level,
)
from src.simulator.simulator_types import PageSet


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
    page_sets: list[PageSet] | None = None,
) -> float:
    """
    Evaluate the cache hit-ratio fitness of an individual.

    When page_sets are provided, uses page-level LRU simulation.
    Otherwise falls back to table-level simulation.

    Parameters
    ----------
    individual : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        LRU cache capacity in 8 KB pages.
    page_sets : list[PageSet] | None
        Per-query page sets from pg_buffercache profiling.

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
        result = simulate_schedule(
            profiles, individual, cache_capacity_pages
        )
    return result.hit_ratio


def _tournament_select(
    population: list[Individual],
    k: int,
    rng: random.Random,
) -> Individual:
    """
    Select one individual from the population via k-tournament selection.

    Randomly samples k individuals and returns a copy of the one with the
    highest fitness.

    Parameters
    ----------
    population : list[Individual]
        Current population of individuals.
    k : int
        Number of candidates to draw for the tournament.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    Individual
        Copy of the selected individual.
    """
    candidates = rng.sample(range(len(population)), k)
    winner = max(candidates, key=lambda i: population[i].fitness())
    return population[winner]


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


@dataclass
class Individual:
    schedule: list[int]
    profiles: list[AccessProfile]
    cache_capacity_pages: int
    _rng: random.Random
    _config: GAConfig
    _fitness: Optional[float] = field(init=False, default=None)

    def fitness(self) -> float:
        if self._fitness is None:
            self._fitness = _fitness(
                self.schedule,
                self.profiles,
                self.cache_capacity_pages,
                None,
            )
        return self._fitness

    def clone(self) -> Individual:
        # Shallow copy all attributes
        new = copy.copy(self)
        # Deep copy schedule
        new.schedule = list(self.schedule)
        if new._rng.random() < new._config.mutation_rate:
            _swap_mutation(new.schedule, new._rng)
        return new

    def __matmul__(
        self,
        other: Individual,
    ) -> Individual:
        if self._rng.random() < self._config.crossover_rate:
            new_schedule = _order_crossover(
                self.schedule,
                other.schedule,
                rng=self._rng
            )
            new = self.clone()
            new._fitness = None
            new.schedule = new_schedule
            if new._rng.random() < new._config.mutation_rate:
                _swap_mutation(new.schedule, new._rng)
        else:
            new = self.clone()
        return new


@dataclass
class IndividualWithPageSet(Individual):
    page_sets: list[PageSet]

    def fitness(self) -> float:
        if self._fitness is None:
            self._fitness = _fitness(
                self.schedule,
                self.profiles,
                self.cache_capacity_pages,
                self.page_sets,
            )
        return self._fitness


def make_individual(
    schedule: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    rng: random.Random,
    config: GAConfig,
    page_sets: Optional[list[PageSet]],
) -> Individual:
    if page_sets is not None:
        return IndividualWithPageSet(
            schedule=schedule,
            profiles=profiles,
            cache_capacity_pages=cache_capacity_pages,
            _rng=rng,
            _config=config,
            page_sets=page_sets,
        )
    return Individual(
        schedule=schedule,
        profiles=profiles,
        cache_capacity_pages=cache_capacity_pages,
        _rng=rng,
        _config=config,
    )


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
        page_sets: list[PageSet] | None = None,
    ) -> ScheduleResult:
        """
        Run the genetic algorithm to find a cache-friendly query schedule.

        Parameters
        ----------
        profiles : list[AccessProfile]
            One access profile per query.
        page_sets : list[PageSet] | None
            Per-query page sets from pg_buffercache profiling.  When
            provided, fitness is evaluated at page-level granularity.

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
    page_sets: list[PageSet] | None = None,
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
    page_sets : list[PageSet] | None
        Per-query page sets from pg_buffercache profiling.  When
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

    population: list[Individual] = []
    for _ in range(config.population_size):
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(
            make_individual(
                schedule=perm,
                profiles=profiles,
                cache_capacity_pages=cache_capacity_pages,
                rng=rng,
                config=config,
                page_sets=page_sets,
            )
        )

    fitness_history: list[float] = []

    for gen in range(config.num_generations):
        ranked: list[Individual] = sorted(
            population,
            key=lambda individual: individual.fitness(),
            reverse=True,
        )
        elites: list[Individual] = ranked[:config.elite_count]

        new_population: list[Individual] = list(elites)

        while len(new_population) < config.population_size:
            p1 = _tournament_select(
                population, config.tournament_size, rng,
            )
            p2 = _tournament_select(
                population, config.tournament_size, rng,
            )

            child = p1 @ p2

            new_population.append(child)

        population = new_population

        best_gen_fitness = max(individual.fitness() for individual in population)
        fitness_history.append(best_gen_fitness)

        if on_generation is not None:
            on_generation(gen, best_gen_fitness)

    best_idx = max(range(len(population)), key=lambda i: population[i].fitness())
    best_schedule = population[best_idx].schedule
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
