import matplotlib.pyplot as plt
import numpy as np
from problems import generate_problem
from solver import Solver
from gmfea import GMFEA, DecisionVariableShufflingStrategy, DecisionVariableTranslationStrategy

gen_from = 30
problem = generate_problem('CI+HS')
solver = Solver('CI+HS', pop_size=100, max_gen=100, rmp=0.3, crossover_count=50,
        # comment the following 4 lines to fallback to normal MFEA
        gmfea=GMFEA(
            dvts = DecisionVariableTranslationStrategy(len(problem.tasks), 0.4, 20, 100),
            dvss = DecisionVariableShufflingStrategy(),
        ))
y = [[] for _ in range(solver.k)]
max_gen = 0
for gen, result in enumerate(solver.solve_every_gen()):
    if gen >= gen_from:
        for i in range(solver.k):
            fitness = result[i]
            y[i].append(fitness)
    max_gen = gen
x = list(range(gen_from, max_gen + 1))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel("Generation")
ax1.set_ylabel("Task 1", color=color)
ax1.plot(x, y[0], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel("Task 2", color=color)
ax2.plot(x, y[1], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
