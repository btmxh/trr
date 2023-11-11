from typing import Optional, Tuple
import numpy as np
from gmfea import GMFEA, DecisionVariableShufflingStrategy, DecisionVariableTranslationStrategy
from problems import Problem, generate_problem
from individual import Individual

# Solver object
class Solver:
    problem: Problem        # Current solving problem
    P: list[Individual]     # Population
    pop_size: int           # Population size
    max_gen: int            # Maximum number of generations
    D_max: int              # Maximum number of dimensions
    rmp: float              # Random mating probability
    excel_threshold: int    # Number of individuals considered excellent
    crossover_count: int    # Number of crossovers at each generation
    k: int                  # Number of tasks
    func_evaluations: int   # Number of objective function evaluations
                            # In the paper, there is a limit of 100,000 evaluations 
    max_func_evaluations: int # The aforementioned limit
    gmfea: Optional[GMFEA]  # Optional GMFEA object, this is None if you want to run normal MFEA
                            # Or this is a GMFEA object if we are running GMFEA

    def __init__(self, problem_name: str, pop_size : int = 1000, max_gen : int = 100, rmp : float = 0.2, excel_threshold : float = 0.4, crossover_count : int = 1000, max_func_evaluations = 100000, gmfea: Optional[GMFEA] = None):
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
        self.gmfea = gmfea

    # Calculate factorial cost of an individual regarding a task
    def evaluate(self, x : Individual, i : int):
        x.task_fitness[i] = self.problem.tasks[i].evaluate(x.value)
        # Increase the number of function evaluations
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
        for _ in range(self.pop_size):
            self.P.append(Individual(np.random.rand(self.D_max), self.k))
        for x in self.P:
            for i in range(self.k):
                self.evaluate(x, i)

    # Assortive mating
    # We don't evaluate the offsprings here yet
    def assortive_mating(self, pa: Individual, pb: Individual) -> tuple[Individual, Individual]:
        offsprings = []
        can_mate = pa.skill_factor == pb.skill_factor or np.random.rand() < self.rmp
        if can_mate:
            for i in range(2):
                # Randomly paste elements of pb into pa
                arr = pa.value.copy()
                L = np.random.choice(range(self.D_max), int(self.D_max / 2), replace = False)
                for i in L:
                    arr[i] = pb.value[i]
                imitate_parent = pa if np.random.rand() < 0.5 else pb
                c = imitate_parent.copy(arr)
                # c imitates pa or pb, so we only calculate its fitness to a task
                offsprings.append(c)
            return offsprings[0], offsprings[1]
        else:
            ca = pa.mutate()
            cb = pb.mutate()
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
        return self.gmfea.dvss.shuffle(self.problem, P, pa, pb)

    def crossover(self, gen: int, max_gen: int) -> list[Individual]:
        res: list[Individual] = []
        for _ in range(self.crossover_count):
            # Step 3 in GMFEA algorithm
            P, dvts = self.translate_population(gen, max_gen)
            pa = P[np.random.randint(0, self.excel_threshold)]     # pa is an "excellent" individual
            pb = P[np.random.randint(0, self.pop_size)]            # pb is a random individual
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
    def search_best(self, i : int):
        for x in self.P:
            if x.skill_factor == i:
                return x
        print(f"Error: best individual of {i + 1}-th task not found")
        return self.P[0]

    def solve(self):
        for result in self.solve_every_gen():
            for i in range(self.k):
                print(f"\t* Task {i + 1}: {result[i]}")

    def solve_every_gen(self):
        self.gen_pop()
        self.calc_scalar(self.P)
        for gen in range(self.max_gen):
            # If too many function evaluations, return immediately
            if self.func_evaluations > self.max_func_evaluations:
                break
            R = self.P + self.crossover(gen, self.max_gen)
            self.calc_scalar(R)
            self.P = R[:self.pop_size]
            print(f"Current generation: {gen + 1}")
            current_result = []
            for i in range(self.k):
                x = self.search_best(i)
                current_result.append(x.task_fitness[i])
            yield current_result
        print("Final result:")
        for i in range(self.k):
            solution, value = self.get_result(i)
            print(f"\t* Task {i + 1}: {value}, with solution:\n{solution}")

    # Return the best solution and the corresponding fitness value
    def get_result(self, task_index: int) -> Tuple[np.ndarray, float]:
        x = self.search_best(task_index)
        return (self.problem.tasks[task_index].map_domain_01(x.value), x.task_fitness[task_index])
 
