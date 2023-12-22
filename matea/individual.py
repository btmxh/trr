import numpy as np
from typing import Optional, Self


class Individual:
    value: np.ndarray
    objective_value: dict[str, float]
    ktc_generation: Optional[int]

    def __init__(self, value: np.ndarray):
        self.value = value
        self.objective_value = {}
        self.ktc_generation = None

    def __str__(self):
        return self.value.__str__()

    def copy(self, keep_objective_values=False) -> Self:
        individual = Individual(np.copy(self.value))
        if keep_objective_values:
            individual.objective_value = {**self.objective_value}
        return individual
