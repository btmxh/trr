from typing import Optional, Tuple, Iterable
import numpy as np
from gmfea import GMFEA, DecisionVariableShufflingStrategy, DecisionVariableTranslationStrategy
from individual import Individual
from task import Task, TaskIterationResult
import math

# Solver object


class MFEA:
    tasks: list[Task]
    P: list[Individual]     # Population
    pop_size: int           # Population size
    D_max: int              # Maximum number of dimensions
    rmp: float              # Random mating probability
    excel_threshold: int    # Number of individuals considered excellent
    crossover_count: int    # Number of crossovers at each generation
    k: int                  # Number of tasks
    func_evaluations: int   # Number of objective function evaluations
    # In the paper, there is a limit of 100,000 evaluations
    max_func_evaluations: int  # The aforementioned limit
    # Optional GMFEA object, this is None if you want to run normal MFEA
    gmfea: Optional[GMFEA]
    # Or this is a GMFEA object if we are running GMFEA

    def __init__(self, tasks: list[Task], pop_size: int = 1000, max_gen: int = 100, rmp: float = 0.2, excel_threshold: float = 0.4, crossover_count: int = 1000, max_func_evaluations=100000, gmfea: Optional[GMFEA] = None):
        self.pop_size = pop_size
        self.rmp = rmp
        self.excel_threshold = int(excel_threshold * pop_size)
        self.crossover_count = crossover_count
        self.tasks = tasks
        self.P = []
        self.D_max = max(map(lambda t: t.num_dimensions, tasks))
        self.k = len(tasks)
        self.gmfea = gmfea
        self.max_gen = max_gen

    # Calculate factorial cost of an individual regarding a task
    def evaluate(self, x: Individual, i: int):
        task = self.tasks[i]
        x.objective_value[task.name] = self.tasks[i].evaluate(x.value)

    # Calculate factoial rank and update scalar fitness of all individuals
    def calc_scalar(self, P: list):
        for i, task in enumerate(self.tasks):
            P.sort(key=lambda x: x.objective_value.get(
                task.name, math.inf))
            for pos in range(len(P)):
                x = P[pos]
                x.factorial_rank[i] = pos + 1
        for x in P:
            x.skill_factor = np.argmin(x.factorial_rank)
            x.scalar_fitness = 1.0 / x.factorial_rank[x.skill_factor]
        P.sort(key=lambda x: x.scalar_fitness, reverse=True)

    # Initialization step
    def gen_pop(self):
        for _ in range(self.pop_size):
            self.P.append(Individual(np.random.rand(self.D_max), self.k))
        for x in self.P:
            for i in range(self.k):
                self.evaluate(x, i)

    def mutate(self, individual: Individual) -> Individual:
        low = np.min(individual.value)
        high = np.max(individual.value)
        sigma = (high - low) / 6
        new_value = np.clip(
            individual.value + np.random.normal(0, sigma, individual.value.size), 0, 1)
        new_individual = individual.copy(keep_objective_values=False)
        new_individual.value = new_value
        return new_individual

    # Assortive mating
    # We don't evaluate the offsprings here yet
    def assortive_mating(self, pa: Individual, pb: Individual) -> tuple[Individual, Individual]:
        offsprings = []
        can_mate = pa.skill_factor == pb.skill_factor or np.random.rand() < self.rmp
        if can_mate:
            for i in range(2):
                # Randomly paste elements of pb into pa
                arr = pa.value.copy()
                L = np.random.choice(range(self.D_max), int(
                    self.D_max / 2), replace=False)
                for i in L:
                    arr[i] = pb.value[i]
                imitate_parent = pa if np.random.rand() < 0.5 else pb
                c = imitate_parent.copy(keep_objective_values=False)
                c.value = arr
                # c imitates pa or pb, so we only calculate its fitness to a task
                offsprings.append(c)
            return offsprings[0], offsprings[1]
        else:
            ca = self.mutate(pa)
            cb = self.mutate(pb)
            return ca, cb

    def translate_population(self, gen: int, max_gen: int) -> Tuple[list[Individual], Optional[DecisionVariableTranslationStrategy]]:
        # If not running GMFEA...
        if self.gmfea is None:
            # simply return the original population
            return self.P, None
        # Otherwise, update and return the translated population
        return self.gmfea.dvts.update(self.P, gen, max_gen)

    def shuffle_parents(self, P: list[Individual], pa: Individual, pb: Individual) -> Tuple[Individual, Individual, Optional[DecisionVariableShufflingStrategy], Optional[list[int]], Optional[list[int]]]:
        # If not running GMFEA...
        if self.gmfea is None:
            # simply return the original parents
            return pa, pb, None, None, None
        # Otherwise, shuffle and return the results
        return self.gmfea.dvss.shuffle(self.tasks, P, pa, pb)

    def crossover(self, gen: int, max_gen: int) -> list[Individual]:
        res: list[Individual] = []
        for _ in range(self.crossover_count):
            # Step 3 in GMFEA algorithm
            P, dvts = self.translate_population(gen, max_gen)
            # pa is an "excellent" individual
            pa = P[np.random.randint(0, self.excel_threshold)]
            # pb is a random individual
            pb = P[np.random.randint(0, self.pop_size)]
            # Step 6
            pa, pb, dvss, L1, L2 = self.shuffle_parents(P, pa, pb)
            o1, o2 = self.assortive_mating(pa, pb)
            # Step 8
            if dvts is not None:
                o1 = dvts.translate_back(o1)
                o2 = dvts.translate_back(o2)
            # Step 10
            if L1 is not None and L2 is not None and dvss is not None:
                o1 = dvss.shuffle_back(o1, L1)
                o2 = dvss.shuffle_back(o2, L2)
            res += [o1, o2]
        # We evaluate the offsprings here
        for o in res:
            self.evaluate(o, o.skill_factor)
        return res

    # Find the best individual regarding task i
    def search_best(self, i: int):
        for x in self.P:
            if x.skill_factor == i:
                return x
        print(f"Error: best individual of {i + 1}-th task not found")
        return self.P[0]

    def solve(self):
        for result in self.solve_every_gen():
            for i in range(self.k):
                print(f"\t* Task {i + 1}: {result[i]}")

    def solve_every_gen(self) -> Iterable[dict[str, TaskIterationResult]]:
        self.gen_pop()
        self.calc_scalar(self.P)
        gen = 0
        while True:
            # If too many function evaluations, return immediately
            R = self.P + self.crossover(gen, self.max_gen)
            self.calc_scalar(R)
            self.P = R[:self.pop_size]
            results = {}
            for i, task in enumerate(self.tasks):
                x = self.search_best(i)
                results[task.name] = TaskIterationResult(x, task.name, gen)
            yield results
            gen += 1

    def iterate(self) -> Iterable[dict[str, TaskIterationResult]]:
        for r in self.solve_every_gen():
            yield r
