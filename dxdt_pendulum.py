"""
Reduced 2nd -> 1st order evolution function for driven, damped pendulum.

"""

import numpy as np


def dxdt_pendulum(x, g, m, l, g, beta=0, A=0, alpha=0):
    """
    Args:
        x (np.ndarray): [postion, angular velocity]
        m (float): Pendulum mass or effective mass
        l (float): Pendulum length or effective length
        beta (float): Damping coefficient
        A (float): Driving amplitude
        alpha (float): Driving frequency
        
    Returns:
        np.ndarray: [angular vel, angular accel]
    """
    
    dxdt = np.zeros(2)
    
    ### Translate 1st order change
    dxdt[0] = x[1]
    
    ### Calc 2nd order change
    dxdt[1] = -(beta/m)*x[1] \
              - (g/l)*np.sin(x[0]) \
              + (A/(m*l))*np.cos(alpha*t)
    
    return dxdt