from typing import Self, Callable
from abc import ABC, abstractmethod
from individual import Individual
from objective import sphere
import numpy as np
import random


class TaskIterationResult:
    best_objective_value: float
    best_individual: Individual
    best_from_ktc: bool

    def __init__(self, best_individual: Individual, task_name: str, generation: int):
        self.best_individual = best_individual
        self.best_objective_value = best_individual.objective_value[task_name]
        self.best_from_ktc = best_individual.ktc_generation == generation


class Task(ABC):
    task_counter = 1

    @staticmethod
    def generate_task_name() -> str:
        name = f'Task {Task.task_counter}'
        Task.task_counter += 1
        return name

    name: str
    population: list[Individual]
    dimensions: int
    objective_fn: Callable[[np.ndarray], float]

    def __init__(self, name=None, dimensions=10, objective_fn=sphere):
        self.name = name or Task.generate_task_name()
        self.dimensions = dimensions
        self.objective_fn = objective_fn
        self.population = []

    def init_population(self, pop_size=100):
        for _ in range(pop_size):
            individual = Individual(np.random.random(self.dimensions) * 2 - 1)
            self.evaluate(individual)
            self.population.append(individual)

    def evaluate(self, individual: Individual) -> float:
        if self.name not in individual.objective_value:
            individual.objective_value[self.name] = self.objective_fn(
                individual.value)
        return individual.objective_value[self.name]

    def do_knowledge_transfer_crossover(self, assist_task: Self, generation: int, ktc_crossover_rate: float) -> TaskIterationResult:
        num_dimensions = min(self.dimensions, assist_task.dimensions)
        original_size = len(self.population)
        children = []
        for parent in self.population:
            assist_parent = random.choice(assist_task.population)

            child = parent.copy(keep_objective_values=False)
            child.ktc_generation = generation
            k = random.choice(range(num_dimensions))
            for i in range(num_dimensions):
                if np.random.rand() < ktc_crossover_rate or i == k:
                    child.value[i] = assist_parent.value[i]
            self.evaluate(child)
            children.append(child)

        self.population.sort(key=self.evaluate)
        self.population = self.population[:original_size]

        return TaskIterationResult(self.population[0], self.name, generation)

    @abstractmethod
    def iterate(self, generation: int) -> TaskIterationResult:
        ...
