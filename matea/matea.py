from task import TaskSolver, TaskIterationResult
import numpy as np
import random
from typing import Iterator


class Archive:
    task: TaskSolver
    # stored as a matrix, to make mean and covariance calculation convenient
    individuals: np.ndarray

    def __init__(self, task: TaskSolver):
        self.task = task
        self.individuals = np.stack([x.value for x in task.population], axis=0)

    def update(self, update_rate: float, archive_size: int):
        for individual in self.task.population:
            if np.random.rand() >= update_rate:
                continue

            if archive_size > len(self.individuals):
                self.individuals = np.vstack(
                    [self.individuals, individual.value])
            else:
                index = random.choice(range(len(self.individuals)))
                self.individuals[index] = individual.value

    # returns mean and covariance
    def get_stats(self, num_dims) -> tuple[np.ndarray, np.ndarray]:
        # in theory, it is probably possible to store these values instead of
        # having to recalculate, but that is not the focus of this algorithm,
        # so we will just go with the naive way of doing things
        pop_matrix = np.resize(
            self.individuals, (len(self.individuals), num_dims))
        mean = np.mean(pop_matrix, axis=0)
        covariance = np.cov(pop_matrix.T)
        mean.resize(num_dims)
        covariance.resize((num_dims, num_dims))
        return mean, covariance


class MaTEA:
    alpha: float
    update_rate: float
    ktc_crossover_rate: float
    archive_size: int
    reinforcement_factor: float  # lambda
    attenuation_update_rate: float  # rho

    tasks: dict[str, TaskSolver]
    archives: dict[str, Archive]

    reinforcement_values: dict[(str, str), float]
    knowledge_transfer_scores: dict[str, dict[str, float]]

    def __init__(self, alpha=0.1, update_rate=0.2, ktc_crossover_rate=0.5,
                 archive_size=300, reinforcement_factor=0.8, attenuation_update_rate=0.8):
        self.alpha = alpha
        self.update_rate = update_rate
        self.ktc_crossover_rate = ktc_crossover_rate
        self.archive_size = archive_size
        self.reinforcement_factor = reinforcement_factor
        self.attenuation_update_rate = attenuation_update_rate
        self.tasks = {}
        self.archives = {}
        self.reinforcement_values = {}
        self.knowledge_transfer_scores = {}

    def add_task(self, task: TaskSolver):
        self.tasks[task.name] = task
        self.archives[task.name] = Archive(task)

    def pick_random_task_that_is_not(self, task: TaskSolver) -> tuple[str, TaskSolver]:
        other_tasks = list(t for t in self.tasks.values() if t != task)
        scores = self.knowledge_transfer_scores.get(task.name, None)
        if scores is None:
            return random.choice(other_tasks)
        return random.choices(other_tasks, [scores.get(t.name, 1.0) for t in other_tasks])[0]

    def update_archive(self, task: TaskSolver):
        archive = self.archives[task.name]
        archive.update(self.update_rate, self.archive_size)

    def update_reinforcement_value(self, task: TaskSolver, assist: TaskSolver, result: TaskIterationResult) -> float:
        reinforcement_value = self.reinforcement_values.get(
            (task.name, assist.name), 1.0)
        if result.best_from_ktc:
            reinforcement_value /= self.reinforcement_factor
        else:
            reinforcement_value *= self.reinforcement_factor
        self.reinforcement_values[(
            task.name, assist.name)] = reinforcement_value
        return reinforcement_value

    def kullback_leibler_divergence(self, p: TaskSolver, q: TaskSolver):
        num_dimensions = min(p.task.num_dimensions, q.task.num_dimensions)
        mean_p, covar_p = self.archives[p.name].get_stats(num_dimensions)
        mean_q, covar_q = self.archives[q.name].get_stats(num_dimensions)
        mean_diff = mean_p - mean_q
        covar_p += 1 * np.identity(num_dimensions)
        covar_q += 1 * np.identity(num_dimensions)
        covar_q_inv = np.linalg.inv(covar_q)

        return 0.5 * (np.log(np.linalg.det(covar_p) / np.linalg.det(covar_q))
                      - num_dimensions
                      + mean_diff.T @ covar_q_inv @ mean_diff
                      + np.trace(covar_q_inv @ covar_p))

    def calc_task_similarity(self, task: TaskSolver, assist: TaskSolver):
        return 0.5 * (self.kullback_leibler_divergence(task, assist)
                      + self.kullback_leibler_divergence(assist, task))

    def update_knowledge_transfer_score(self, task: TaskSolver, assist: TaskSolver, result: TaskIterationResult):
        if task.name not in self.knowledge_transfer_scores:
            self.knowledge_transfer_scores[task.name] = {}
        if assist.name not in self.knowledge_transfer_scores[task.name]:
            self.knowledge_transfer_scores[task.name][assist.name] = 1.0
        score = self.knowledge_transfer_scores[task.name][assist.name]

        reinforcement_value = self.update_reinforcement_value(
            task, assist, result)
        task_similarity = self.calc_task_similarity(task, assist)

        score = max(0.001, self.attenuation_update_rate * score +
                    reinforcement_value / np.log(max(0.001, task_similarity)))
        self.knowledge_transfer_scores[task.name][assist.name] = score

    def iterate(self) -> Iterator[dict[str, TaskIterationResult]]:
        generation = 0
        while True:
            generation += 1
            results = {}
            for task in self.tasks.values():
                if np.random.rand() < self.alpha and len(self.tasks) > 1:
                    assist = self.pick_random_task_that_is_not(task)
                    result = task.do_knowledge_transfer_crossover(
                        assist, generation, self.ktc_crossover_rate)
                    self.update_knowledge_transfer_score(task, assist, result)
                else:
                    result = task.iterate(generation)
                self.update_archive(task)
                results[task.name] = result
            yield results
