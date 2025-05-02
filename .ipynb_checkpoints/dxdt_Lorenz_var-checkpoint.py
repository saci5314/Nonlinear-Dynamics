"""
Evolution function for Lorenz attractor and variational/perturbational terms.

"""

import numpy as np


def dxdt_Lorenz_var(x, a, r, b):
    """
    Args:
        x (list or np.ndarray) State vector
        a (float): Prantyl number
        r (float): Rayleigh number
        b (float): Geometric coefficient
        
    Outputs:
        dxdt (np.ndarray): State evolution vector
    """
    
    ### Preallocating
    dxdt = np.zeros(12)
    
    ### Nominal system derivatives
    dxdt[0] = a*(x[1] - x[0]) # xdot
    dxdt[1] = r*x[0] - x[1] - x[0]*x[2] # ydot
    dxdt[2] = x[0]*x[1] - b*x[2] # zdot
    
    ### Variational derivatives w/ respect to x
    dxdt[3] = -a*x[3] + a*x[4] # ddxx
    dxdt[4] = (r - x[2])*x[3] - x[4] - x[0]*x[5] #ddxy
    dxdt[5] = x[1]*x[3] + x[0]*x[4] - b*x[5] # ddxz
    
    ### Variational derivatives w/ respect to y
    dxdt[6] = -a*x[6] + a*x[7] # ddyx
    dxdt[7] = (r - x[2])*x[6] - x[7] - x[0]*x[8] #ddyy
    dxdt[8] = x[1]*x[6] + x[0]*x[7] - b*x[8] # ddyx
    
    ### Variational derivatives w/ respect to z
    dxdt[9] = -a*x[9] + a*x[10] # ddzx
    dxdt[10] = (r - x[2])*x[9] - x[10] - x[0]*x[11] #ddzy
    dxdt[11] = x[1]*x[9] + x[0]*x[10] - b*x[11] # ddzz
    
    return dxdt
