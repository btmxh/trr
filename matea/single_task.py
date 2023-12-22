from task import TaskSolver, TaskIterationResult
from typing import Iterator

class SingleTaskSolver:
    solvers: list[TaskSolver]

    def __init__(self, solvers: list[TaskSolver]):
        self.solvers = solvers

    def iterate(self) -> Iterator[dict[str, TaskIterationResult]]:
        gen = 0
        while True:
            yield {task.name: task.iterate(gen) for task in self.solvers}
            gen += 1
