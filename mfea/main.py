import numpy as np
import problems

# An individual
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
    _pop_size = 0
    _max_gen = 0
    _P = []
    _problems = []
    _D_max = 0
    _rmp = 0
    _excel_threshold = 0    # Number of individuals considered excellent
    _crossover_count = 0    # Number of crossovers at each generation
    _mutation_rate = 0
    _k = 0                  # Number of tasks

    def __init__(self, pop_size : int = 1000, max_gen : int = 100, rmp : float = 0.2, excel_threshold : float = 0.4, crossover_count : int = 1000, mutation_rate = 0.2):
        self._pop_size = pop_size
        self._max_gen = max_gen
        self._rmp = rmp
        self._excel_threshold = excel_threshold * pop_size
        self._crossover_count = crossover_count
        self._mutation_rate = mutation_rate

    # Map [0, 1]^D -> Task's domain
    def _convert(self, x : np.ndarray, i : int):
        (low, high) = self._problems['domains'][i]
        return x * (high - low) + low

    # Calculate factorial cost of an individual regarding a task
    def _evaluate(self, x : Individual, i : int):
        x.task_fitness[i] = self._problems['functions'][i](self._convert(x.value, i), self._problems['dimensions'][i])

    # Calculate factoial rank and update scalar fitness of all individuals
    def _calc_scalar(self, P : list):
        for i in range(self._k):
            P.sort(key = lambda x : x.task_fitness[i])
            for pos in range(len(P)):
                x = P[pos]
                x.factorial_rank[i] = pos + 1
        for x in P:
            x.skill_factor = np.argmin(x.factorial_rank)
            x.scalar_fitness = 1.0 / x.factorial_rank[x.skill_factor]
        P.sort(key = lambda x : x.scalar_fitness, reverse = True)

    # Initialization step
    def _gen_pop(self):
        self._P.clear()
        for _ in range(self._pop_size):
            self._P.append(Individual(np.random.rand(self._D_max), self._k))
        for x in self._P:
            for i in range(self._k):
                task = self._problems['functions'][i]
                dim = self._problems['dimensions'][i]
                self._evaluate(x, i)

    # Gaussian mutation
    def _mutate(self, x : np.ndarray):
        low = np.min(x)
        high = np.max(x)
        sigma = (high - low) / 6
        for i in range(x.size):
            x[i] += np.random.normal(0, sigma)
        x = np.clip(x, 0, 1)

    def _crossover(self) -> list:
        res = []
        for _ in range(self._crossover_count):
            pa = self._P[np.random.randint(0, self._excel_threshold)]       # pa is a "excellent" individual
            pb = self._P[np.random.randint(0, self._pop_size)]              # pb is a random individual
            offsprings = []
            can_mate = pa.skill_factor == pb.skill_factor or np.random.rand() < self._rmp
            if can_mate:
                for i in range(2):
                    # Randomly paste elements of pb into pa
                    arr = pa.value.copy()
                    L = np.random.choice(range(self._D_max), int(self._D_max / 2), replace = False)
                    for i in L:
                        arr[i] = pb.value[i]
                    c = Individual(arr, self._k)
                    # c imitates pa or pb, so we only calculate its fitness to a task
                    c.skill_factor = pa.skill_factor if np.random.rand() < 0.5 else pb.skill_factor
                    self._evaluate(c, c.skill_factor)
                    offsprings.append(c)
            else:
                (ca, cb) = (Individual(pa.value.copy(), self._k), Individual(pb.value.copy(), self._k))
                if np.random.rand() < self._mutation_rate:
                    self._mutate(ca.value)
                if np.random.rand() < self._mutation_rate:
                    self._mutate(cb.value)
                ca.skill_factor = pa.skill_factor
                cb.skill_factor = pb.skill_factor
                self._evaluate(ca, ca.skill_factor)
                self._evaluate(cb, cb.skill_factor)
                offsprings.append(ca)
                offsprings.append(cb)
            res += offsprings
        return res

    # Find the best individual regarding task i
    def _search_best(self, i : int):
        for x in self._P:
            if x.skill_factor == i:
                return x
        print(f"Best individual of {i + 1}-th task not found")

    def solve(self, problem : str):
        self._problems = problems.problems[problem]
        self._D_max = max(self._problems['dimensions'])
        self._k = len(self._problems['functions'])

        self._gen_pop()
        self._calc_scalar(self._P)
        for gen in range(self._max_gen):
            R = self._P + self._crossover()
            self._calc_scalar(R)
            self._P = R[:self._pop_size]
            print(f"Current generation: {gen + 1}")
            for i in range(len(self._problems['functions'])):
                x = self._search_best(i)
                print(f"\t* Task {i + 1}: {x.task_fitness[i]}")
        print("Final result:")
        for i in range(self._k):
            x = self._search_best(i)
            print(f"\t* Task {i + 1}: {x.task_fitness[i]}, with solution:\n{self._convert(x.value, i)[:self._problems['dimensions'][i]]}")

problem = Problem(700, 100, 0.2, 0.4, 1000, 0.2)
problem.solve('NI+MS')