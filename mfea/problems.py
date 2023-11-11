from typing import Callable, Optional, Self, Tuple
import numpy as np
from scipy.stats import special_ortho_group

def sphere(x, d):
    return np.dot(x[:d], x[:d])

def ackley (x,d):
    return -20 * np.exp(-0.2 / np.sqrt(d) * np.linalg.norm(x[:d])) - np.exp(1/d * np.sum(np.cos(2 * np.pi * x[:d]))) + 20 +  np.e

def rosenbrock(x,d):
    sum = 0
    for i in range (d-1):
        sum += 100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i] - 1) ** 2
    return sum

def rastrign (x,d):
    return np.sum(x[:d] ** 2 - 10 * np.cos(2 * np.pi * x[:d]) + 10) 


def griewank(x,d):
    y = np.zeros(d)
    for i in range(d):
        y[i] = x[i] / np.sqrt(i + 1)
    return 1 + 1/4000 * sphere(x,d) - np.prod(np.cos(y))

def schwefel(x,d): 
    return 418.9829 * d - np.sum(x[:d] * np.sin(np.absolute(x[:d]) ** (1/2)))

def weiertrass(x,d):
    k_max = 20
    a = 0.5
    b = 3

    sum = 0
    for i in range(d):
        for k in range(k_max + 1):
            sum += a ** k * np.cos(2 * np.pi * b ** k * (x[i] + 0.5))
    
    for k in range(k_max + 1):
        sum -= d * (a ** k * np.cos(2 * np.pi * b ** k * 0.5))
    
    return sum

ObjectiveFunction = Callable[[np.ndarray, int], float]
class Task:
    objective_fn: ObjectiveFunction
    dimensions: int
    domain: Tuple[float, float]
    rotation_matrix: Optional[np.ndarray]
    translation: Optional[np.ndarray]

    def __init__(self, function: ObjectiveFunction, dimensions: int, domain: Tuple[float, float], has_rotation: bool = True, translation: Optional[np.ndarray] = None) -> None:
        self.objective_fn = function
        self.dimensions = dimensions
        self.domain = domain
        self.rotation_matrix = np.identity(dimensions) if has_rotation else None
        self.translation = translation

    def map_domain_01(self, vec: np.ndarray) -> np.ndarray:
        min, max = self.domain
        return vec[:self.dimensions] * (max - min) + min

    def generate_task(self):
        task = Task(self.objective_fn, self.dimensions, self.domain, translation=self.translation)
        if self.rotation_matrix is not None:
            task.rotation_matrix = special_ortho_group.rvs(self.dimensions)
        return task
    
    def evaluate(self, vec: np.ndarray) -> float:
        vec = self.map_domain_01(vec)
        if self.translation is not None:
            vec += self.translation
        if self.rotation_matrix is not None:
            vec = np.matmul(self.rotation_matrix, vec)
        return self.objective_fn(vec, self.dimensions)

class Problem:
    name: str
    tasks: list[Task]

    def __init__(self, name: str, *tasks: Task) -> None:
        self.name = name
        self.tasks = list(tasks)

    def generate_random(self) -> Self:
        return type(self)(self.name, *[task.generate_task() for task in self.tasks])
    
    def max_dimensions(self) -> int:
        return max(map(lambda task: task.dimensions, self.tasks))

def generate_problems():
    return { problem.name: problem.generate_random() for problem in [
        Problem("CI+HS", Task(griewank, 50, (-100, 100)), Task(rastrign, 50, (-50, 50))),
        Problem("CI+MS", Task(ackley, 50, (-50, 50)), Task(rastrign, 50, (-50, 50))),
        Problem("CI+LS", Task(ackley, 50, (-50, 50)), Task(schwefel, 50, (-500, 500), has_rotation=False)),
        Problem("PI+HS", Task(rastrign, 50, (-50, 50)), Task(sphere, 50, (-100, 100), has_rotation=False, translation=np.array([0] * 25 + [-20] * 25))),
        Problem("PI+MS", Task(ackley, 50, (-50, 50)), Task(rosenbrock, 50, (-50, 50), has_rotation=False)),
        Problem("PI+LS", Task(ackley, 50, (-50, 50)), Task(weiertrass, 50, (-0.5, 0.5))),
        Problem("NI+HS", Task(rosenbrock, 50, (-50, 50), has_rotation=False), Task(rastrign, 50, (-50, 50))),
        Problem("NI+MS", Task(griewank, 50, (-100, 100), translation=np.array([-10] * 50)), Task(weiertrass, 25, (-0.5, 0.5))),
        Problem("NI+LS", Task(rastrign, 50, (-50, 50)), Task(schwefel, 50, (-500, 500), has_rotation=False)),   
    ]}

def generate_problem(name: str):
    return generate_problems()[name]
