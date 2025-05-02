"""
Reduced 1st-order evolution function for 3-body problem.

"""

import numpy as np
from numpy.linalg import norm


def dxdt_3_body(x, m_1, m_2, m_3, G):
    """
    3-body equations of motion.

    Args:
        x (np.ndarray): State vector
        m_1 (float): Body 1 mass
        m_2 (float): body 2 mass
        m_3 (float): body 3 mass
        G (float): Gravitational constant
        
    Returns:
        np.ndarray: State evolution vector
    """
    
    xdot = np.zeros(18)
    
    ### body 1 velocity components
    xdot[0:3] = x[9:12]
    
    ### body 2 velocity components
    xdot[3:6] = x[12:15]
    
    ### body 3 velocity components
    xdot[6:9] = x[15:18]
    
    ### body 1 acceleration components
    xdot[9:12] = -G * (m_2*(x[0:3] - x[3:6])/norm(x[0:3] - x[3:6])**3 + \
                       m_3*(x[0:3] - x[6:9])/norm(x[0:3] - x[6:9])**3)
    
    ### body 2 acceleration components
    xdot[12:15] = -G * (m_1*(x[3:6] - x[0:3])/norm(x[3:6] - x[0:3])**3 + \
                        m_3*(x[3:6] - x[6:9])/norm(x[3:6] - x[6:9])**3)
    
    ### body 3 acceleration components
    xdot[15:18] = -G * (m_1*(x[6:9] - x[0:3])/norm(x[6:9] - x[0:3])**3 + \
                        m_2*(x[6:9] - x[3:6])/norm(x[6:9] - x[3:6])**3)
    
    return xdot
