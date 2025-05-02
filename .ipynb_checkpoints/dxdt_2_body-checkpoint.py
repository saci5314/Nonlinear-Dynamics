"""
Reduced 1st-order evolution function for 2-body problem.

"""

import numpy as np


def dxdt_2_body(x, m_1, m_2, G):
    """
    2-body equations of motion.
    
    Args:
        x (np.ndarray): State vector
        m_1 (float): Body 1 mass
        m_2 (float): body 2 mass
        G (float): Gravitational constant
        
    Returns:
        np.ndarray: State evolution vector
    """
    
    xdot = np.zeros(12)
    
    ### Adjust relative displacement to avoid dividing
    ### by 0 if bodies pass too close.
    epsilon = 1e-10
    r12 = np.linalg.norm(x[0:3] - x[3:6])
    if np.linalg.norm(x[0:3] - x[3:6]) < epsilon:
         r12 = epsilon
    
    ### body 1 velocity components
    xdot[0:3] = x[6:9]
    
    ### body 2 velocity components
    xdot[3:6] = x[9:]
            
    ### body 1 acceleration components
    xdot[6:9] = -G * m_2 * (x[0:3] - x[3:6]) / r12**3
    
    ### body 2 acceleration components
    xdot[9:12] = -G * m_1 * (x[3:6] - x[0:3]) / r12**3

    return xdot