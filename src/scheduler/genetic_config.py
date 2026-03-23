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

from dataclasses import dataclass, field
from typing import Literal

from src.simulator.dqn_simulator import DQN


type FitnessType = Literal['lru'] | Literal['dqn']
"""
Types of supported fitness functions.

- LRU: Simulation of a LRU cache, metric is hit rate.
- DQN: Deep Q-Network, metric is the output of the neural network.
"""


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
    all_tables : list[str]
        List of all tables used in all the queries.
    max_pages : dict[str, int]
        The maximum number of pages used by each table.
    fitness_type : FitnessType
        The type of fitness function to use in the genetic algorithm.
    dqn : DQN | None
        The Deep Q-Network used if the DQN fitness function is selected.
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
    all_tables: list[str] = field(default_factory=list)
    max_pages: dict[str, int] = field(default_factory=dict)
    fitness_type: FitnessType = "lru"
    dqn: DQN | None = None
    seed: int | None = None
