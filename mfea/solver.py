import sys
from typing import Self, Tuple
import numpy as np
from problems import Problem, generate_problem

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
    def _mutate(self) -> Self:
        low = np.min(self.value)
        high = np.max(self.value)
        sigma = (high - low) / 6
        new_chromosomes = np.clip(self.value + np.random.normal(0, sigma, self.value.size), 0, 1)
        return type(self)(new_chromosomes, self.task_fitness.size, self.skill_factor)
        

# Algorithm implementation
class Solver:
    problem: Problem
    P: list[Individual]
    pop_size: int
    max_gen: int
    D_max: int
    rmp: float
    excel_threshold: int    # Number of individuals considered excellent
    crossover_count: int    # Number of crossovers at each generation
    k: int                  # Number of tasks
    func_evaluations: int
    max_func_evaluations: int

    def __init__(self, problem_name: str, pop_size : int = 1000, max_gen : int = 100, rmp : float = 0.2, excel_threshold : float = 0.4, crossover_count : int = 1000, max_func_evaluations = 100000):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.rmp = rmp
        self.excel_threshold = int(excel_threshold * pop_size)
        self.crossover_count = crossover_count
        self.problem = generate_problem(problem_name)
        self.P = []
        self.D_max = self.problem.max_dimensions()
        self.k = len(self.problem.tasks)
        self.func_evaluations = 0
        self.max_func_evaluations = max_func_evaluations

    # Calculate factorial cost of an individual regarding a task
    def evaluate(self, x : Individual, i : int):
        x.task_fitness[i] = self.problem.tasks[i].evaluate(x.value)
        self.func_evaluations += 1

    # Calculate factoial rank and update scalar fitness of all individuals
    def calc_scalar(self, P : list):
        for i in range(self.k):
            P.sort(key = lambda x : x.task_fitness[i])
            for pos in range(len(P)):
                x = P[pos]
                x.factorial_rank[i] = pos + 1
        for x in P:
            x.skill_factor = np.argmin(x.factorial_rank)
            x.scalar_fitness = 1.0 / x.factorial_rank[x.skill_factor]
        P.sort(key = lambda x : x.scalar_fitness, reverse = True)

    # Initialization step
    def gen_pop(self):
        self.P.clear()
        for _ in range(self.pop_size):
            self.P.append(Individual(np.random.rand(self.D_max), self.k))
        for x in self.P:
            for i in range(self.k):
                self.evaluate(x, i)


    def crossover(self) -> list[Individual]:
        res = []
        for _ in range(self.crossover_count):
            pa = self.P[np.random.randint(0, self.excel_threshold)]     # pa is an "excellent" individual
            pb = self.P[np.random.randint(0, self.pop_size)]            # pb is a random individual
            offsprings = []
            can_mate = pa.skill_factor == pb.skill_factor or np.random.rand() < self.rmp
            if can_mate:
                for i in range(2):
                    # Randomly paste elements of pb into pa
                    arr = pa.value.copy()
                    L = np.random.choice(range(self.D_max), int(self.D_max / 2), replace = False)
                    for i in L:
                        arr[i] = pb.value[i]
                    c = Individual(arr, self.k)
                    # c imitates pa or pb, so we only calculate its fitness to a task
                    c.skill_factor = pa.skill_factor if np.random.rand() < 0.5 else pb.skill_factor
                    self.evaluate(c, c.skill_factor)
                    offsprings.append(c)
            else:
                ca = pa._mutate()
                cb = pb._mutate()
                self.evaluate(ca, ca.skill_factor)
                self.evaluate(cb, cb.skill_factor)
                offsprings.append(ca)
                offsprings.append(cb)
            res += offsprings
        return res

    # Find the best individual regarding task i
    def search_best(self, i : int):
        for x in self.P:
            if x.skill_factor == i:
                return x
        print(f"Error: best individual of {i + 1}-th task not found")
        return self.P[0]

    def solve(self):
        self.gen_pop()
        self.calc_scalar(self.P)
        for gen in range(self.max_gen):
            if self.func_evaluations > self.max_func_evaluations:
                break
            R = self.P + self.crossover()
            self.calc_scalar(R)
            self.P = R[:self.pop_size]
            print(f"Current generation: {gen + 1}")
            for i in range(self.k):
                x = self.search_best(i)
                print(f"\t* Task {i + 1}: {x.task_fitness[i]}")
        print("Final result:")
        for i in range(self.k):
            solution, value = self.get_result(i)
            print(f"\t* Task {i + 1}: {value}, with solution:\n{solution}")

    def get_result(self, task_index: int) -> Tuple[np.ndarray, float]:
        x = self.search_best(task_index)
        return (self.problem.tasks[task_index].map_domain_01(x.value), x.task_fitness[task_index])
 
