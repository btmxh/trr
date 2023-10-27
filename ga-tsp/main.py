import tsplib95
import random
import typing
import sys

# Hyperparameters
POPULATION_SIZE = 1000
NUM_GENERATIONS = 1000
NUM_CROSSOVERS = 500
MUTATION_RATE = 0.01

# Load dataset
path = sys.argv[1] if len(sys.argv) > 1 else 'ga-tsp/wi29.tsp'
problem = tsplib95.load(path)
nodes = list(problem.get_nodes())

# type alias
Solution = list

# ALGORITHM IMPLEMENTATION (the only relevant part)
population = []

def fitness(solution: Solution) -> float:
    distance = 0
    for i in range(len(solution)):
        next_node_index = (i + 1) % len(solution)
        distance += problem.get_weight(solution[i], solution[next_node_index])
    return -distance

def crossover(first_parent: Solution, second_parent: Solution):
    # this implementation is OX1
    # see https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Order_crossover_(OX1)

    # for simplicity, we will only select one segment in the first step: [begin, end)
    # NOTE: `end`` is not included in the range
    begin = random.randint(0, len(first_parent))
    end = random.randint(0, len(first_parent))

    if begin >= end:
        # swap the two indices if they are not in order
        temp = begin
        begin = end
        end = temp

    child = list(first_parent)
    range_set = set(child[begin:end])

    # reorder the genes in the range according to their order in `second_parent`
    # the wikipedia example should make this trivial to understand
    index = begin
    for gene in second_parent:
        if gene in range_set:
            child[index] = gene
            index += 1
    return child

def mutate(solution: Solution):
    i = random.randint(0, len(solution) - 1)
    j = random.randint(0, len(solution) - 1)
    temp = solution[i]
    solution[i] = solution[j]
    solution[j] = temp

for _ in range(POPULATION_SIZE):
    solution = list(nodes)
    random.shuffle(solution)
    population.append(solution)

for generation in range(NUM_GENERATIONS + 1):
    print(f"Generation {generation + 1}")
    population.sort(key=fitness, reverse=True)
    print(f"Best fitness: {fitness(population[0])}")

    if generation == NUM_GENERATIONS:
        # if last generation, dump data and exit
        print(f"Optimal solution: ${population[0]}")
        positions = population[0]
        break

    # we move the selection step here because after crossover, the population list would not be properly sorted.
    # selection
    del population[POPULATION_SIZE:]

    # crossover
    for _ in range(NUM_CROSSOVERS):
        # pick a random index, prioritizing small indices (which corresponds to high fitness)
        # NOTE: first `POPULATION_SIZE` chromosomes of the population is from the previous generation,
        # and the remaining chromosomes are added in this crossover process. We would not want to
        # accidentally select a "newly-born" chromosome to do crossover.
        index = min(int(random.triangular(0, POPULATION_SIZE)), len(population) - 1)
        first_parent = population[index]
        
        # do it again to get the second parent
        index = min(int(random.triangular(0, POPULATION_SIZE)), len(population) - 1)
        second_parent = population[index]

        child = crossover(first_parent, second_parent)
        population.append(child)

    # mutation
    for solution in population:
        if random.random() < MUTATION_RATE:
            mutate(solution)
# END OF ALGORITHM IMPLEMENTATION

# visualize the solution (this is just code copied from the internet, which is not relevant to the algorithm)
if not problem.is_depictable():
    # problem not visualizable
    exit(0)

import matplotlib.pyplot as plt
positions = [typing.cast(list, problem.get_display(node)) for node in population[0]]
xs = [pos[0] for pos in positions]
ys = [pos[1] for pos in positions]

fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter(xs, ys)
ax[1].scatter(xs, ys)
for i, node in enumerate(positions):
    next_index = (i + 1) % len(positions)
    next_node = positions[next_index]
    ax[1].annotate("",
            xy=node, xycoords='data',
            xytext=next_node, textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))

textstr = f"N nodes: {len(positions)}\nTotal length: {fitness(population[0]):.3f}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14, # Textbox
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
