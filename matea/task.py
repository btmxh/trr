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


class Task:
    name: str
    objective_fn: Callable[[np.ndarray], float]
    num_dimensions: int
    input_range: [float, float]
    translation: np.ndarray
    rotation: np.ndarray

    def __init__(self, objective_fn: Callable[[np.ndarray], float], num_dimensions: int, input_range: [float, float],
                 name: str, translation: Optional[np.ndarray] = None, rotation: Optional[np.ndarray] = None):
        self.name = name
        self.objective_fn = objective_fn
        self.num_dimensions = num_dimensions
        self.input_range = input_range
        self.translation = translation or np.zeros(num_dimensions)
        self.rotation = rotation or np.identity(num_dimensions)

    def evaluate(self, vector: np.ndarray):
        min, max = self.input_range
        vector = vector * (max - min) + min
        vector = self.rotation * vector + self.translation
        return self.objective_fn(vector)


class TaskSolver(ABC):
    task_counter = 1

    @staticmethod
    def generate_task_name() -> str:
        name = f'Task {TaskSolver.task_counter}'
        TaskSolver.task_counter += 1
        return name

    name: str
    population: list[Individual]
    task: Task

    def __init__(self, task: Task, name=None):
        self.name = name or task.name or TaskSolver.generate_task_name()
        self.task = task
        self.population = []

    def init_population(self, pop_size=100):
        for _ in range(pop_size):
            individual = Individual(np.random.random(self.task.num_dimensions))
            self.evaluate(individual)
            self.population.append(individual)

    def evaluate(self, individual: Individual) -> float:
        if self.name not in individual.objective_value:
            individual.objective_value[self.name] = self.objective_fn(
                individual.value)
        return individual.objective_value[self.name]

    def do_knowledge_transfer_crossover(self, assist_task: Self, generation: int, ktc_crossover_rate: float) -> TaskIterationResult:
        num_dimensions = min(self.task.num_dimensions,
                             assist_task.task.num_dimensions)
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
