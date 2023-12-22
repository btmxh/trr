import numpy as np
from typing import Optional, Self


class Individual:
    value: np.ndarray
    objective_value: dict[str, float]
    ktc_generation: Optional[int]
    factorial_rank: list[Optional[int]]
    skill_factor: int

    def __init__(self, value: np.ndarray, num_tasks=1):
        self.value = value
        self.objective_value = {}
        self.ktc_generation = None
        self.factorial_rank = [None] * num_tasks
        self.skill_factor = -1

    def __str__(self):
        return self.value.__str__()

    def copy(self, keep_objective_values=False) -> Self:
        individual = Individual(np.copy(self.value), len(self.factorial_rank))
        if keep_objective_values:
            individual.objective_value = {**self.objective_value}
        return individual
