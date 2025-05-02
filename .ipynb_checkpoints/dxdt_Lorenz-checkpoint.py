"""
Evolution function for Lorenz attractor.

"""

import numpy as np


def dxdt_Lorenz(x, a, r, b):
    """
    Args:
        x (list or np.ndarray) State vector
        a (float): Prantyl number
        r (float): Rayleigh number
        b (float): Geometric coefficient
        
    Returns:
        np.ndarray: State evolution vector
    """
    
    dxdt = np.zeros(3)

    dxdt[0] = a*(x[1] - x[0])
    dxdt[1] = r*x[0] - x[1] - x[0]*x[2]
    dxdt[2] = x[0]*x[1] - b*x[2]    
    
    return dxdt    


