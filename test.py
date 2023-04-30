import numpy as np
from scipy.optimize import minimize_scalar

def bezier(t, control_points):
    n = len(control_points) - 1
    return np.sum([binom(n, i) * (1 - t) ** (n - i) * t ** i * control_points[i] for i in range(n + 1)], axis=0)

def binom(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

def closest_x_coordinate(x, control_points, tolerance=1e-4):
    def distance_squared_to_curve(t):
        return (bezier(t, control_points)[0] - x) ** 2
    res = minimize_scalar(distance_squared_to_curve, bounds=(0, 1), method='bounded', options={'xatol': tolerance})
    return res.x

x = 2.5
control_points = [np.array([0, 0]), np.array([1, 1]), np.array([2, -1]), np.array([3, 0])]
t = closest_x_coordinate(x, control_points)
print(t)