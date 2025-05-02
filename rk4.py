"""
Nonadaptive, nonautonomous 4th-order Runge-Kutta integrator.

"""

import numpy as np


def rk4(dxdt, x0, t0, dt, n):
    """
    Args:
        dxdt: System evolution function 
        x0 (float or np.ndarray): Initial conditions
        t0 (float): Initial time
        dt (float): Timestep size
        n (float): Timestep qty
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - t: Time data
            - x: Solution data
    """
    
    ### Preprocessing
    t = np.linspace(t0, t0+n*dt, n+1)
    x = np.zeros([len(x0), n+1])
    x[:,0] = x0
    
    ### Timestepping
    for i in range(n):
        # Calc nonautonomous Runge-Kutta stages
        k1 = dxdt(x[:,i],        t[i]       )*dt
        k2 = dxdt(x[:,i] + k1/2, t[i] + dt/2)*dt
        k3 = dxdt(x[:,i] + k2/2, t[i] + dt/2)*dt
        k4 = dxdt(x[:,i] + k3,   t[i] + dt  )*dt
        
        # Calc new solution
        x[:,i+1] = x[:,i] + (k1 + 2*k2 + 2*k3 + k4)/6

    return t, x
    
    
