from typing import Optional, Self, Tuple
import numpy as np
import random
from problems import Problem
from individual import Individual

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

    def update(self, population: list[Individual], gen: int, max_gen: int) -> Tuple[list[Individual], Optional[Self]]:
        D_max = population[0].value.size
        new_population = []
        if gen > self.phi:
            if gen % self.theta == 0:
                # Clear this list, since this contains translation vectors from past iterations
                self.translations.clear()
                for k in range(self.K):
                    alpha = (gen / max_gen) ** 2
                    cp = np.full(D_max, 0.5)
                    # The following 5 lines are used to calculate m_k
                    sorted_by_task = sorted(population, key=lambda x: x.task_fitness[k])
                    m_k = np.zeros(D_max)
                    for individual in sorted_by_task[:int(self.mu * len(population))]:
                        m_k += individual.value
                    m_k /= int(self.mu * len(population))
                    d_k = self.scale_factor * alpha * (cp - m_k)
                    self.translations.append(d_k)
            if len(self.translations) > 0:
                for individual in population:
                    # Translate forwards: + translation_vector
                    new_value = individual.value + self.translations[individual.skill_factor]
                    new_population.append(individual.copy(new_value))
                return new_population, self
        return population, None

    def translate_back(self, individual: Individual):
        # Translate backwards: - translation_vector
        new_value = individual.value - self.translations[individual.skill_factor]
        return individual.copy(new_value)

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
            # If D1 = D2, we don't have to shuffle back, so return no information
            return p1, p2, None, None, None

        if D1 < D2:
            same_skill_factors = [p for p in population if p.skill_factor == p2.skill_factor]
            p = random.sample(same_skill_factors, 1)[0].copy()
            random.shuffle(L1)
            for i in range(D1):
                p.value[L1[i]] = p1.value[i]
            # Here, we return self, L1 and L2
            # which contains the information to shuffle back
            return p, p2, self, L1, L2
        else:
            same_skill_factors = [p for p in population if p.skill_factor == p1.skill_factor]
            p = random.sample(same_skill_factors, 1)[0].copy()
            random.shuffle(L2)
            for i in range(D2):
                p.value[L2[i]] = p2.value[i]
            return p1, p, self, L1, L2
    
    def shuffle_back(self, individual: Individual, permutation: list[int]):
        # `inverse` is the inverse permutation of `permutation`
        inverse = np.argsort(permutation)
        new_individual = individual.copy()
        # so applying `inverse` to individual will shuffle back individual:
        for i in range(len(inverse)):
            new_individual.value[inverse[i]] = individual.value[i]
        return new_individual

class GMFEA:
    dvts: DecisionVariableTranslationStrategy
    dvss: DecisionVariableShufflingStrategy

    def __init__(self, dvts: DecisionVariableTranslationStrategy, dvss: DecisionVariableShufflingStrategy) -> None:
        self.dvts = dvts
        self.dvss = dvss
