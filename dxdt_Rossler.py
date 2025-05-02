"""
Evolution function for Rossler attractor.

"""

import numpy as np


def dxdt_Rossler(x, a, b, c):
    """
    Args:
        x (list or np.ndarray): State vector
        a = (float): Constant parameter
        b = (float): Constant parameter
        c = (float): Constant parameter
        
    Returns:
        dxdt (np.ndarray): State evolution vector
    """
    
    dxdt = np.zeros(3)
    
    dxdt[0] = -(x[1] + x[2])
    dxdt[1] = x[0] + a*x[1]
    dxdt[2] = b + x[2]*(x[0] - c)    
    
    return dxdt