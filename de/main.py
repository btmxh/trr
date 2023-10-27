n = 2
NP = 10 * n
CR = 0.9
F = 0.8
NUM_GENERATIONS = 10000

import numpy as np
import random

class solution:
    def __init__(self, value) -> None:
        self.value = value

    @classmethod
    def random(cls):
        return cls(np.random.random(n))
    
    def fitness(self) -> float:
        # Sphere function
        # return -np.inner(self.value, self.value)
        # Ackley function
        a = 20
        b = 0.2
        c = 2 * np.pi
        return -a * np.exp(-b * np.linalg.norm(self.value)) - np.exp(np.sum(np.cos(c * self.value)) / n) + a + np.exp(1)
    
    def mutate(self, a, b, c):
        y = a.value + F * (b.value - c.value)
        R = random.randint(0, NP - 1)
        for i in range(n):
            if random.uniform(0, 1) >= CR or i == R:
                y[i] = self.value[i]
        return solution(y)

population = []
for _ in range(NP):
    population.append(solution.random())

for i in range(NUM_GENERATIONS):
    best = None
    best_fitness = None
    for i_x in range(NP):
        x = population[i_x]
        a, b, c = random.sample(population, 3)
        while a == x or b == x or c == x:
            a, b, c = random.sample(population, 3)
        y = x.mutate(a, b, c)

        if y.fitness() > x.fitness():
            population[i_x] = y
            x = y
        if best_fitness is None or best_fitness < x.fitness():
            best_fitness = x.fitness()
            best = x
    print(f"Generation {i + 1}:", best.value, best_fitness)

