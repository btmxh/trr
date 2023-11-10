import sys
import numpy as np
from gmfea import GMFEA, DecisionVariableShufflingStrategy, DecisionVariableTranslationStrategy
from solver import Solver
from problems import generate_problem

# average from 20 runs
def sample_runs(name: str, num_runs=20):
    problem = generate_problem(name)
    # basically we keep track of X1 + X2 + ... and X1^2 + X2^2 + ...
    current_sum = [0.0] * len(problem.tasks)
    current_squared_sum = [0.0] * len(problem.tasks)
    for i in range(num_runs):
        print(f"Run #{i + 1}:")
        solver = Solver(name, pop_size=100, max_gen=1000, rmp=0.3, crossover_count=50,
        # comment the following 4 lines to fallback to normal MFEA
        gmfea=GMFEA(
            dvts = DecisionVariableTranslationStrategy(len(problem.tasks), 0.4, 20, 100),
            dvss = DecisionVariableShufflingStrategy(),
        )
        )
        solver.solve()
        for task_index in range(len(problem.tasks)):
            _, value = solver.get_result(task_index)
            current_sum[task_index] += value
            current_squared_sum[task_index] += value ** 2
    for task_index in range(len(problem.tasks)):
        # calculate E[X] and E[X^2]
        mean = current_sum[task_index] / num_runs
        squared_mean = current_squared_sum[task_index] / num_runs
        # just basic stats here
        variance = squared_mean - mean ** 2
        # stddev = standard deviation
        stddev = np.sqrt(variance)
        print(f"Task {task_index + 1}: mean={mean}, stddev={stddev}")

# usage: python mfea/main.py <name of problem>
# e.g. python mfea/main.py CI+HS
sample_runs(sys.argv[1])
