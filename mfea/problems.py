import numpy as np

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

problems = {
    'CI+HS': { 'functions': [griewank, rastrign], 'dimensions': [50, 50] },
    'CI+MS': { 'functions': [ackley, rastrign], 'dimensions': [50, 50] },
    'CI+LS': { 'functions': [ackley, schwefel], 'dimensions': [50, 50] },
    'PI+HS': { 'functions': [rastrign, sphere], 'dimensions': [50, 50] },
    'PI+MS': { 'functions': [ackley, rosenbrock], 'dimensions': [50, 50] },
    'PI+LS': { 'functions': [ackley, weiertrass], 'dimensions': [50, 25] },
    'NI+HS': { 'functions': [rosenbrock, rastrign], 'dimensions': [50, 50] },
    'NI+MS': { 'functions': [griewank, weiertrass], 'dimensions': [50, 25] },
    'NI+LS': { 'functions': [rastrign, schwefel], 'dimensions': [50, 50] },
}