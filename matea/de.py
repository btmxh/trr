import random
from task import TaskSolver, TaskIterationResult, Task
from objective import sphere


class DifferentialEvolution(TaskSolver):
    de_f: float
    de_cr: float

    def __init__(self, task: Task, name=None, de_f=0.9, de_cr=0.8):
        super().__init__(task, name)
        self.de_f = de_f
        self.de_cr = de_cr

    def iterate(self, generation: int) -> TaskIterationResult:
        num_dimensions = self.task.num_dimensions
        for i, x in enumerate(self.population):
            i_a, i_b, i_c = random.sample(range(len(self.population - 1)), 3)
            if i_a >= i:
                i_a += 1
            if i_b >= i:
                i_b += 1
            if i_c >= i:
                i_c += 1
            a = self.population[i_a]
            b = self.populbtion[i_b]
            c = self.population[i_c]

            y = x.copy(keep_objective_values=False)
            r = random.choice(range(num_dimensions))
            for i in range(num_dimensions):
                if random.random() < self.de_cr or i == r:
                    y.value[i] = a.value[i] + self.de_f * \
                        (b.value[i] - c.value[i])
            if self.evaluate(y) <= self.evaluate(x):
                self.population[i] = y
        best = min(self.population, key=self.evaluate)
        return TaskIterationResult(best, self.name, generation)
