"""
Abstract base class for query scheduling algorithms.

All scheduling algorithms should inherit from ``SchedulerBase`` and
implement the ``schedule`` method.  This makes it straightforward to
add new algorithms (e.g. simulated annealing, greedy heuristics) while
keeping the rest of the pipeline unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import SimulationResult


@dataclass(frozen=True)
class ScheduleResult:
    """
    Outcome produced by any scheduling algorithm.

    Attributes
    ----------
    best_schedule : list[int]
        Best permutation of query indices found by the algorithm.
    best_fitness : float
        Cache hit ratio of the best schedule.
    best_simulation : SimulationResult
        Full simulation result for the best schedule.
    fitness_history : list[float]
        Per-generation best fitness values tracked during optimization.
    metadata : dict
        Algorithm-specific extra information.
    """

    best_schedule: list[int]
    best_fitness: float
    best_simulation: SimulationResult
    fitness_history: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SchedulerBase(ABC):
    """
    Abstract base for query scheduling algorithms.

    Subclasses must implement ``schedule`` which takes a list of access
    profiles and a cache capacity, and returns a ``ScheduleResult``.
    """

    @abstractmethod
    def schedule(
        self,
        profiles: list[AccessProfile],
    ) -> ScheduleResult:
        """
        Find an optimized query execution order.

        Parameters
        ----------
        profiles : list[AccessProfile]
            One access profile per query.

        Returns
        -------
        ScheduleResult
            The best schedule found by the algorithm.
        """
