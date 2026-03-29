"""
Genetic algorithm primitives for query schedule evolution.

Provides the core building blocks used by the GA scheduler: fitness
evaluation, tournament selection, order crossover, swap mutation, and
the Individual dataclass hierarchy that encapsulates a schedule with
lazily cached fitness.

Classes
-------
Individual
    A candidate query execution order with lazy cached fitness evaluated
    via table-level LRU simulation.
IndividualWithPageSet
    Extends Individual to evaluate fitness using page-level simulation
    from pg_buffercache data.

Functions
---------
make_individual
    Factory that constructs the appropriate Individual subtype based on
    whether page sets are available.
select_parents
    Selects multiple parents from the population via tournament selection.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from src.scheduler.genetic_config import GAConfig
from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    simulate_schedule,
    simulate_schedule_page_level,
)
def _fitness(
    individual: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    page_sets: list[frozenset[int]] | None = None,
) -> float:
    """
    Evaluate the cache hit-ratio fitness of an individual.

    When page_sets are provided, uses page-level clock-sweep simulation.
    Otherwise falls back to table-level simulation.

    Parameters
    ----------
    individual : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages.
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
    """
    A single individual in the GA population.

    Encapsulates a query execution order (schedule) along with the context
    needed to evaluate its fitness.  Fitness is computed lazily and cached
    to avoid redundant simulation calls.

    Attributes
    ----------
    schedule : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages used for fitness evaluation.
    _rng : random.Random
        Random number generator instance used for mutation and crossover.
    _config : GAConfig
        Algorithm configuration, including mutation and crossover rates.
    _fitness : float or None
        Cached fitness value.  None until first call to fitness().
    """

    schedule: list[int]
    profiles: list[AccessProfile]
    cache_capacity_pages: int
    _rng: random.Random
    _config: GAConfig
    _fitness: Optional[float] = field(init=False, default=None)

    def fitness(self) -> float:
        """
        Return the cache hit-ratio fitness of this individual.

        Evaluates and caches the fitness on the first call using
        table-level simulation.  Subsequent calls return the cached value.

        Returns
        -------
        float
            Cache hit ratio in the range [0.0, 1.0].
        """
        if self._fitness is None:
            self._fitness = _fitness(
                self.schedule,
                self.profiles,
                self.cache_capacity_pages,
                None,
            )
        return self._fitness

    def clone(self) -> Individual:
        """
        Produce a mutated copy of this individual.

        Shallow-copies all attributes and deep-copies the schedule.
        Swap mutation is applied with probability config.mutation_rate.

        Returns
        -------
        Individual
            A new individual with a possibly mutated schedule.  The
            cached fitness is inherited and remains valid unless the
            schedule was mutated.
        """
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
        """
        Produce an offspring by crossing this individual with another.

        With probability config.crossover_rate, applies order crossover
        between the two parents, then applies swap mutation with
        probability config.mutation_rate. If crossover is not applied,
        returns a clone of self with possible mutation.

        Parameters
        ----------
        other : Individual
            The second parent.

        Returns
        -------
        Individual
            A new offspring individual.
        """
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
    """
    An Individual that evaluates fitness using page-level simulation.

    Extends Individual with a set of per-query page sets captured from
    pg_buffercache, enabling finer-grained cache hit estimation.

    Attributes
    ----------
    page_sets : list[frozenset[int]]
        Integer-encoded page sets used for page-level fitness evaluation.
    """

    page_sets: list[frozenset[int]]

    def fitness(self) -> float:
        """
        Return the page-level cache hit-ratio fitness of this individual.

        Evaluates and caches the fitness on the first call using
        page-level simulation.  Subsequent calls return the cached value.

        Returns
        -------
        float
            Cache hit ratio in the range [0.0, 1.0].
        """
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
    page_sets: Optional[list[frozenset[int]]],
) -> Individual:
    """
    Construct an Individual or IndividualWithPageSet depending on inputs.

    Parameters
    ----------
    schedule : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        LRU cache capacity in 8 KB pages.
    rng : random.Random
        Random number generator instance.
    config : GAConfig
        Algorithm configuration.
    page_sets : list[frozenset[int]] or None
        Integer-encoded page sets for page-level simulation.  If None,
        an Individual using table-level simulation is returned.

    Returns
    -------
    Individual
        IndividualWithPageSet if page_sets is provided, otherwise Individual.
    """
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


def select_parents(
    population: list[Individual],
    config: GAConfig,
    rng: random.Random,
    n_parents: int = 2,
) -> tuple[Individual, ...]:
    """
    Select multiple parents from the population via tournament selection.

    Parameters
    ----------
    population : list[Individual]
        Current population to select from.
    config : GAConfig
        Algorithm configuration, providing tournament_size.
    rng : random.Random
        Random number generator instance.
    n_parents : int
        Number of parents to select.  Defaults to 2.

    Returns
    -------
    tuple[Individual, ...]
        Selected parents, each chosen independently by tournament selection.
    """
    parents = []
    for _ in range(n_parents):
        parent = _tournament_select(population, config.tournament_size, rng)
        parents.append(parent)
    return tuple(parents)


__all__ = [
    "_fitness",
    "_tournament_select",
    "_order_crossover",
    "_swap_mutation",
    "Individual",
    "make_individual",
    "select_parents",
]
