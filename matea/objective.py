import numpy as np


def sphere(x: np.ndarray):
    return np.dot(x, x)


def ackley(x: np.ndarray):
    d, = x.shape
    return -20 * np.exp(-0.2 / np.sqrt(d) * np.linalg.norm(x)) - np.exp(1/d * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e


def rosenbrock(x: np.ndarray):
    sum = 0
    d, = x.shape
    for i in range(d-1):
        sum += 100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i] - 1) ** 2
    return sum


def rastrigin(x: np.ndarray):
    d, = x.shape
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def griewank(x):
    d, = x.shape
    y = np.zeros(d)
    for i in range(d):
        y[i] = x[i] / np.sqrt(i + 1)
    return 1 + 1/4000 * sphere(x) - np.prod(np.cos(y))


def schwefel(x):
    d, = x.shape
    return 418.9829 * d - np.sum(x[:d] * np.sin(np.absolute(x[:d]) ** (1/2)))


def weierstrass(x):
    d, = x.shape
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
