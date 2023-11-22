import numpy as np
import sys

# minimize f(x,y) = c_x*x^2 + c_y*y^2
# input via CLI args
# usage example: python main.py 1 2 exact
c_x, c_y, line_search = sys.argv[1:]

c_x = float(c_x)
c_y = float(c_y)

A = np.array([
    [c_x, 0],
    [0, c_y]
]) * 2
b = np.zeros(2).T

# initial solution
x0 = np.array([2, 1])

epsilon = 1e-5

def f(x):
    return np.matmul(np.matmul(x.T, A) + b, x) / 2

def jacobian(x):
    return np.matmul(x.T, A) + b

def gradient(x):
    return jacobian(x).T

def exact_line_search(f, x0, d):
    return -np.matmul(jacobian(x0), d) / np.matmul(np.matmul(d.T, A), d)

def constant_step_length(f, x0, d):
    return 0.1

def backtracking_line_search(f, x0, d):
    alpha = 0.5
    m1 = 0.5
    t = 1
    while f(x0 + t * d) > f(x0) + m1 * t * np.matmul(jacobian(x0), d):
        t *= alpha
    return t

# this uses generator functions to make things far more readable
def descent(f, x0, line_search):
    yield x0
    d = -gradient(x0)
    # terminate if `d` is too small
    if np.linalg.norm(d) < epsilon:
        return
    step_length = line_search(f, x0, d)
    x0 = x0 + step_length * d
    yield from descent(f, x0, line_search)

results = []

for line_search in [exact_line_search, constant_step_length, backtracking_line_search]:
    num_iterations = 10
    solutions = []
    for i, x in enumerate(descent(f, x0, line_search)):
        if i >= num_iterations:
            break
        solutions.append(x)
    results.append(solutions)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
for result in results:
    xs = [p[0] for p in result]
    ys = [p[1] for p in result]
    ax.scatter(xs, ys)
    for i in range(len(result) - 1):
        p1 = result[i]
        p2 = result[i + 1]
        ax.annotate("", xy=p1, xytext=p2, xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.tight_layout()
plt.show()
