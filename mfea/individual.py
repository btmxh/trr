from typing import Optional, Self
import numpy as np

# An individual
# Upon initialization, each task's fitness defaults to +inf
class Individual:
    value : np.ndarray
    task_fitness : np.ndarray
    factorial_rank : np.ndarray
    skill_factor : int
    scalar_fitness : float

    def __init__(self, chromosomes : np.ndarray, task_num : int, skill_factor = -1):
        self.value = chromosomes
        self.factorial_rank = np.zeros(task_num)
        self.task_fitness = np.full(task_num, float("inf"))
        self.skill_factor = skill_factor
        self.scalar_fitness = 0

    # Gaussian mutation
    def mutate(self) -> Self:
        low = np.min(self.value)
        high = np.max(self.value)
        sigma = (high - low) / 6
        new_chromosomes = np.clip(self.value + np.random.normal(0, sigma, self.value.size), 0, 1)
        return self.copy(new_chromosomes)
        
    # Somewhat misleading, this function creates a new individual with the same skill factor as `self`
    # If `new_chromosomes` is not provided, this has the same effect as cloning the individual
    def copy(self, new_chromosomes: Optional[np.ndarray] = None) -> Self:
        if new_chromosomes is None:
            new_chromosomes = self.value.copy()
        return type(self)(new_chromosomes, self.task_fitness.size, self.skill_factor)
