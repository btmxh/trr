import random
from task import Task, TaskIterationResult
from objective import sphere


class DifferentialEvolution(Task):
    de_f: float
    de_cr: float

    def __init__(self, name=None, dimensions=10, objective_fn=sphere, de_f=0.9, de_cr=0.8):
        super().__init__(name, dimensions, objective_fn)
        self.de_f = de_f
        self.de_cr = de_cr

    def iterate(self, generation: int) -> TaskIterationResult:
        for i, x in enumerate(self.population):
            a, b, c = random.sample(self.population, 3)
            while a == x or b == x or c == x:
                a, b, c = random.sample(self.population, 3)
            y = x.copy(keep_objective_values=False)
            r = random.choice(range(self.dimensions))
            for i in range(self.dimensions):
                if random.random() < self.de_cr or i == r:
                    y.value[i] = a.value[i] + self.de_f * \
                        (b.value[i] - c.value[i])
            if self.evaluate(y) <= self.evaluate(x):
                self.population[i] = y
        best = min(self.population, key=self.evaluate)
        return TaskIterationResult(best, self.name, generation)
