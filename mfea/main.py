import sys
import numpy as np
from solver import Solver
from problems import generate_problem

def sample_runs(name: str, num_runs=20):
    problem = generate_problem(name)
    current_sum = [0.0] * len(problem.tasks)
    current_squared_sum = [0.0] * len(problem.tasks)
    for i in range(num_runs):
        print(f"Run #{i + 1}:")
        solver = Solver(name, pop_size=100, max_gen=10000000, rmp=0.3)
        solver.solve()
        for task_index in range(len(problem.tasks)):
            _, value = solver.get_result(task_index)
            current_sum[task_index] += value
            current_squared_sum[task_index] += value ** 2
    for task_index in range(len(problem.tasks)):
        mean = current_sum[task_index] / num_runs
        squared_mean = current_squared_sum[task_index] / num_runs
        variance = squared_mean - mean ** 2
        stddev = np.sqrt(variance)
        print(f"Task {task_index + 1}: mean={mean}, stddev={stddev}")

sample_runs(sys.argv[1])
