"""
Configuration dataclass for the genetic algorithm scheduler.

This module defines GAConfig, which holds all tunable hyperparameters
for the GA: population size, generation count, crossover and mutation
rates, tournament size, elitism count, cache capacity, and random seed.

Classes
-------
GAConfig
    Hyperparameter configuration for the genetic algorithm.
"""

from dataclasses import dataclass


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
