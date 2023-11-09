import numpy as np
import problems

# Represents an individual
# Upon initialization, each task's fitness defaults to +inf
class Individual:
    value : np.ndarray
    task_fitness : np.ndarray
    factorial_rank : np.ndarray
    skill_factor : int
    scalar_fitness : float

    def __init__(self, chromosomes : np.ndarray, task_num : int):
        self.value = chromosomes
        self.factorial_rank = np.zeros(task_num)
        self.task_fitness = np.full(task_num, float("inf"))
        self.skill_factor = -1
        self.scalar_fitness = 0

# Algorithm implementation
class Problem:
    __pop_size = 0
    __max_gen = 0
    __P = []
    __problems = []
    __D_max = 0
    __rmp = 0
    __excel_threshold = 0   # Number of individuals considered excellent
    __crossover_count = 0   # Number of crossovers at each generation
    __mutation_rate = 0
    __k = 0                 # Number of tasks

    def __init__(self, pop_size : int = 1000, max_gen : int = 100, rmp : float = 0.2, excel_threshold : float = 0.4, crossover_count : int = 1000, mutation_rate = 0.2):
        self.__pop_size = pop_size
        self.__max_gen = max_gen
        self.__rmp = rmp
        self.__excel_threshold = excel_threshold * pop_size
        self.__crossover_count = crossover_count
        self.__mutation_rate = mutation_rate

    # Map [0, 1]^D -> Task's domain
    def __convert__(self, x : np.ndarray, i : int):
        (low, high) = self.__problems['domains'][i]
        return x * (high - low) + low

    # Calculate factorial cost of an individual regarding a task
    def __evaluate__(self, x : Individual, i : int):
        x.task_fitness[i] = self.__problems['functions'][i](self.__convert__(x.value, i), self.__problems['dimensions'][i])

    # Calculate factoial rank and update scalar fitness of all individuals
    def __calc_scalar__(self, P : list):
        for i in range(self.__k):
            P.sort(key = lambda x : x.task_fitness[i])
            for pos in range(len(P)):
                x = P[pos]
                x.factorial_rank[i] = pos + 1
        for x in P:
            x.skill_factor = np.argmin(x.factorial_rank)
            x.scalar_fitness = 1.0 / x.factorial_rank[x.skill_factor]
        P.sort(key = lambda x : x.scalar_fitness, reverse = True)

    # Initialization step
    def __gen_pop__(self):
        self.__P.clear()
        for _ in range(self.__pop_size):
            self.__P.append(Individual(np.random.rand(self.__D_max), self.__k))
        for x in self.__P:
            for i in range(self.__k):
                task = self.__problems['functions'][i]
                dim = self.__problems['dimensions'][i]
                self.__evaluate__(x, i)

    # Gaussian mutation
    def __mutate__(self, x : np.ndarray):
        low = np.min(x)
        high = np.max(x)
        sigma = (high - low) / 6
        for i in range(x.size):
            x[i] += np.random.normal(0, sigma)
        x = np.clip(x, 0, 1)

    def __crossover__(self) -> list:
        res = []
        for _ in range(self.__crossover_count):
            pa = self.__P[np.random.randint(0, self.__excel_threshold)]     # pa is an "excellent" individual
            pb = self.__P[np.random.randint(0, self.__pop_size)]            # pb is a random individual
            offsprings = []
            can_mate = pa.skill_factor == pb.skill_factor or np.random.rand() < self.__rmp
            if can_mate:
                for i in range(2):
                    # Randomly paste elements of pb into pa
                    arr = pa.value.copy()
                    L = np.random.choice(range(self.__D_max), int(self.__D_max / 2), replace = False)
                    for i in L:
                        arr[i] = pb.value[i]
                    c = Individual(arr, self.__k)
                    # c imitates pa or pb, so we only calculate its fitness to a task
                    c.skill_factor = pa.skill_factor if np.random.rand() < 0.5 else pb.skill_factor
                    self.__evaluate__(c, c.skill_factor)
                    offsprings.append(c)
            else:
                (ca, cb) = (Individual(pa.value.copy(), self.__k), Individual(pb.value.copy(), self.__k))
                if np.random.rand() < self.__mutation_rate:
                    self.__mutate__(ca.value)
                if np.random.rand() < self.__mutation_rate:
                    self.__mutate__(cb.value)
                ca.skill_factor = pa.skill_factor
                cb.skill_factor = pb.skill_factor
                self.__evaluate__(ca, ca.skill_factor)
                self.__evaluate__(cb, cb.skill_factor)
                offsprings.append(ca)
                offsprings.append(cb)
            res += offsprings
        return res

    # Find the best individual regarding task i
    def __search_best__(self, i : int):
        for x in self.__P:
            if x.skill_factor == i:
                return x
        print(f"Error: best individual of {i + 1}-th task not found")
        return self.__P[0]

    def solve(self, problem : str):
        self.__problems = problems.problems[problem]
        self.__D_max = max(self.__problems['dimensions'])
        self.__k = len(self.__problems['functions'])

        self.__gen_pop__()
        self.__calc_scalar__(self.__P)
        for gen in range(self.__max_gen):
            R = self.__P + self.__crossover__()
            self.__calc_scalar__(R)
            self.__P = R[:self.__pop_size]
            print(f"Current generation: {gen + 1}")
            for i in range(len(self.__problems['functions'])):
                x = self.__search_best__(i)
                print(f"\t* Task {i + 1}: {x.task_fitness[i]}")
        print("Final result:")
        for i in range(self.__k):
            x = self.__search_best__(i)
            print(f"\t* Task {i + 1}: {x.task_fitness[i]}, with solution:\n{self.__convert__(x.value, i)[:self.__problems['dimensions'][i]]}")

problem = Problem(700, 100, 0.2, 0.4, 1000, 0.2)
problem.solve('NI+MS')
