from matea import MaTEA
from mfea import MFEA
from de import DifferentialEvolution
from task import Task, TaskSolver
import numpy as np
from objective import sphere, ackley, weierstrass, rosenbrock, schwefel, griewank, rastrigin
from single_task import SingleTaskSolver

T1 = Task(name="Task 1 (Sphere 50D, no translation)",
          objective_fn=sphere, input_range=[-100, 100], num_dimensions=50)
T2 = Task(name="Task 2 (Sphere 50D, positive translation)", objective_fn=sphere,
          input_range=[-100, 100], num_dimensions=50, translation=np.ones(50) * 80)
T3 = Task(name="Task 3 (Sphere 50D, negative translation)", objective_fn=sphere,
          input_range=[-100, 100], num_dimensions=50, translation=np.ones(50) * -80)
T4 = Task(name="Task 4 (Weierstrass 25D, with translation)", objective_fn=weierstrass,
          input_range=[-0.5, 0.5], num_dimensions=25, translation=np.ones(25) * -0.4)
T5 = Task(name="Task 5 (Rosenbrock 50D, no translation)",
          objective_fn=rosenbrock, input_range=[-50, 50], num_dimensions=50)
T6 = Task(name="Task 6 (Ackley 50D, with translation)", objective_fn=ackley,
          input_range=[-50, 50], num_dimensions=50, translation=np.ones(50) * 40)
T7 = Task(name="Task 7 (Weierstrass 50D, with translation)", objective_fn=weierstrass,
          input_range=[-0.5, 0.5], num_dimensions=50, translation=np.ones(50) * -0.4)
T8 = Task(name="Task 8 (Schwefel 50D, with translation)", objective_fn=schwefel,
          input_range=[-500, 500], num_dimensions=50, translation=np.ones(50) * 420.9687)
T9 = Task(name="Task 9 (Griewank 50D, with translation)", objective_fn=griewank,
          input_range=[-100, 100], num_dimensions=50, translation=np.array([-80] * 25 + [80] * 25))
T10 = Task(name="Task 10 (Rastrigin 50D, with translation)", objective_fn=rastrigin,
           input_range=[-50, 50], num_dimensions=50, translation=np.array([40] * 25 + [-40] * 25))
tasks = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]


def de_solvers():
    solvers = []
    for task in tasks:
        de = DifferentialEvolution(task)
        de.init_population(100)
        solvers.append(de)
    return solvers


def MaTDE(tasks: list[Task]) -> MaTEA:
    ea = MaTEA()
    for solver in de_solvers():
        ea.add_task(solver)
    return ea


# solver = MaTDE(tasks)
solver = MFEA(tasks)
# solver = SingleTaskSolver(de_solvers())
for gen, results in enumerate(solver.iterate()):
    print(f"Generation {gen + 1}")
    for name, result in results.items():
        print(name, result.best_objective_value)
        # print(result.best_objective_value)
    if gen > 100:
        break
