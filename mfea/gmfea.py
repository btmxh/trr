import numpy as np
import random
from mfea.problems import Problem
from mfea.solver import Individual

class DecisionVariableTranslationStrategy:
    K: int
    mu: float
    theta: int
    phi: int
    translations = []

    def __init__(self, K: int, mu: float, theta: int, phi: int, scale_factor: float = 1.25):
        self.K = K
        self.mu = mu
        self.theta = theta
        self.phi = phi
        self.scale_factor = scale_factor

    def update(self, population: list[Individual], gen: int, max_gen: int) -> list[Individual]:
        D_max = population[0].value.size
        new_population = []
        if gen > self.phi:
            if gen % self.theta == 0:
                self.translations.clear()
                for k in range(self.K):
                    alpha = (gen / max_gen) ** 2
                    cp = np.full(D_max, 0.5)
                    sorted_by_task = sorted(population, key=lambda x: x.task_fitness[k])
                    m_k = np.zeros(D_max)
                    for individual in sorted_by_task[:int(self.mu * len(population))]:
                        m_k += individual.value
                    m_k /= int(self.mu * len(population))
                    d_k = self.scale_factor * alpha * (cp - m_k)
                    self.translations.append(d_k)
            for individual in population:
                new_value = individual.value + self.translations[individual.skill_factor]
                new_population.append(individual.copy(new_value))
            return new_population
        return population

class DecisionVariableShufflingStrategy:
    def __init__(self):
        pass

    def shuffle(self, problem: Problem, population: list[Individual], p1: Individual, p2: Individual):
        D1 = problem.tasks[p1.skill_factor].dimensions
        D2 = problem.tasks[p2.skill_factor].dimensions
        D_max = max(D1, D2)
        L1 = list(range(D_max))
        L2 = list(range(D_max))
        if D1 == D2:
            return p1, p2, L1, L2

        if D1 < D2:
            same_skill_factors = [p for p in population if p.skill_factor == p2.skill_factor]
            p = random.sample(same_skill_factors, 1)[0].copy()
            random.shuffle(L1)
            for i in range(D1):
                p.value[L1[i]] = p1.value[i]
            return p, p2, L1, L2
        
        if D1 > D2:
            same_skill_factors = [p for p in population if p.skill_factor == p1.skill_factor]
            p = random.sample(same_skill_factors, 1)[0].copy()
            random.shuffle(L2)
            for i in range(D2):
                p.value[L2[i]] = p2.value[i]
            return p1, p, L1, L2
