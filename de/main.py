n = 10
NP = 10 * n
NUM_GENERATIONS = 1000
ARGMIN = -5
ARGMAX = 10
EPSILON = 1e-4
# DE params
CR = 0.9
F = 0.8
SBX_ETA = 5
# GA params
MUTATION_RATE = 0.1

import numpy as np
import random
import matplotlib.pyplot as plt

class solution:
    def __init__(self, value) -> None:
        self.value = value
        self.lazy_fitness = None

    @classmethod
    def random(cls):
        return cls(np.random.random(n))
    
    def fitness(self):
        if self.lazy_fitness is not None:
            return self.lazy_fitness
        x = self.value
        self.lazy_fitness = -sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(n - 1))
        return self.lazy_fitness
    
    def de_mutate(self, a, b, c):
        y = a.value + F * (b.value - c.value)
        R = random.randint(0, NP - 1)
        for i in range(n):
            if random.uniform(0, 1) >= CR or i == R:
                y[i] = self.value[i]
        y = np.clip(y, -5, 10)
        return solution(y)

    def copy(self):
        return solution(self.value.copy())
    

    def ga_crossover(self, other):
        def calc_beta_q(alpha):
            rand = random.random()
            if rand < 1 / alpha:
                return pow(rand * alpha, 1 / (SBX_ETA + 1))
            else:
                return pow(1 / (2 - rand * alpha), 1 / (SBX_ETA + 1))

        c = [self.copy(), other.copy()]
        for i in range(n):
            p1 = self.value[i]
            p2 = other.value[i]
            if abs(p1 - p2) < EPSILON:
                continue
            mn = min(self.value[i], other.value[i])
            mx = max(self.value[i], other.value[i])
            for j in range(2):
                numerator = (mn - ARGMIN) if j == 0 else (ARGMAX - mx)
                beta = 1 + 2 * numerator / (mx - mn)
                alpha = pow(beta, -(SBX_ETA + 1))
                beta_q = calc_beta_q(alpha)
                c[j].value[i] = np.clip(0.5 * (mn + mx) - beta_q * (mn + mx), ARGMIN, ARGMAX)
            if random.random() < 0.5:
                c[0].value[i], c[1].value[i] = c[1].value[i], c[0].value[i]
        return c[0], c[1]
            
    def ga_mutate(self):
        if random.random() > MUTATION_RATE:
            return self
        # https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)#Mutation_without_consideration_of_restrictions
        diff = np.max(self.value) - np.min(self.value)
        value = self.value + np.random.normal(0, diff / 6)
        value = np.clip(value, ARGMIN, ARGMAX)
        return solution(value)
        


def solve_de():
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
            y = x.de_mutate(a, b, c)

            if y.fitness() > x.fitness():
                population[i_x] = y
                x = y
            if best_fitness is None or best_fitness < x.fitness():
                best_fitness = x.fitness()
                best = x
        yield best, best_fitness

def solve_ga():
    population = []
    for _ in range(NP):
        population.append(solution.random())
    for i in range(NUM_GENERATIONS):
        for j in range(NP // 2):
            p1, p2 = random.sample(population, 2)
            c1, c2 = p1.ga_crossover(p2)
            c1 = c1.ga_mutate()
            c2 = c2.ga_mutate()
            population.append(c1)
            population.append(c2)
        population.sort(key = lambda sol: sol.fitness(), reverse=True)
        population = population[:NP]
        yield population[0], population[0].fitness()

def test_de():
    for gen, (best, best_fitness) in enumerate(solve_de()):
        print(f"Generation {gen + 1}:", best.value, best_fitness)

def test_ga():
    for gen, (best, best_fitness) in enumerate(solve_ga()):
        print(f"Generation {gen + 1}:", best.value, best_fitness)

def plot_compare():
    gen_from = 30
    x = list(range(gen_from, NUM_GENERATIONS))
    de = []
    ga = []
    for gen, (_, best_fitness) in enumerate(solve_de()):
        if gen >= gen_from:
            de.append(best_fitness)
    for gen, (_, best_fitness) in enumerate(solve_ga()):
        if gen >= gen_from:
            ga.append(best_fitness)
    plt.plot(x, de, 'r', label="DE")
    plt.plot(x, ga, 'b', label="GA")
    plt.tight_layout()
    plt.show()

plot_compare()
