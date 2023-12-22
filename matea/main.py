from matea import MaTEA
from de import DifferentialEvolution
from task import Task
from objective import sphere, ackley

T1 = Task(name="Task 1 (Sphere 50D, no translation)", objective_fn=sphere, input_range=[-100, 100], num_dimensions=50)

solver = MaTEA()
s10 = DifferentialEvolution(
    name="10D Sphere", dimensions=10, objective_fn=sphere)
s10.init_population(100)
solver.add_task(s10)
a20 = DifferentialEvolution(
    name="20D Ackley", dimensions=20, objective_fn=ackley)
a20.init_population(100)
solver.add_task(a20)
a5 = DifferentialEvolution(
    name="5D Ackley", dimensions=5, objective_fn=ackley)
a5.init_population(100)
solver.add_task(a5)
for gen, results in enumerate(solver.iterate()):
    print(f"Generation {gen + 1}")
    for name, result in results.items():
        print(name)
        print(result.best_objective_value)
        print(result.best_individual)
