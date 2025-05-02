"""
Logistic map and reduced Henon map generators.

"""

import numpy as np


def logistic_map(x0, r, m, l=0):
    """
    Logistic map generator.

    Args:
        r (float): growth constant
        m (float): number of iterations
        l (float): optional number of iterations to
                    truncate from begining

    Returns:
        x (np.ndarray) map of x for m iterations
    """

    x = np.zeros(m); 
    x[0] = x0

    # Iterate logistic map
    for k in range(m-1):
        x[k+1] = x[k] * r * (1 - x[k])

    # Truncate first iterations from solution matrix if l>0
    x = np.array(x[l:m])

    return x


def henon_map(a, b, m, l=0):
    """
    Reduced 1D Henon map generator.

    Args:
        a (float): x term quadratic decay constant
        b (float): y term linear growth constant
        m (float): number of iterations
        l (float): optional number of iterations to
                    truncate from begining

    Returns:
        x (np.ndarray) map of x for m iterations
    """

    # Preallocate solution vectors + asssing initial conditions
    x = np.zeros(m); 

    # Deconstruct initial conditions
    x0 = 0.5 # x_0 initial condition in 2D equivalent map
    y_k = 0 # y_0 initial condition in 2D equivalent map
    #x[0] = x0               # 1D initial condition for x_0
    #x[1] = y0 + 1 - a*x0**2 # 1D initial condition for x_1

    # Iterate Henon map over all iterations and values of a
    for k in range(m-1):
        x[k+1] = y_k + 1 - a*x[k]**2
        y_k = b*x[k]

    # Truncate first iterations from solution matrix if l>0
    x = np.array(x[l:m])

    return x