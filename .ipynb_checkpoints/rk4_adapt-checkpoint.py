"""
Aadaptive, nonautonomous 4th-order Runge-Kutta integrator.

"""

import numpy as np


def rk4_adapt(dxdt, x0, t0, tf, dt_min=.00001, dt_max=None):
    """
    Args:
        dxdt: System evolution function
        x0 (float): Initial conditions
        t0 (float): Initial time
        tf (float): End time
        dt_min (float): Timestep underwriting condition
        dt_max (float): Optional baseline/maximum timestep size
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - t: Time data
            - x: Solution data
    """

    ### Time data
    t = [t0]; # [s] current time
    if dt_max == None: dt_max = tf/10  # [s] max timestep
    h = dt_max # [s] current timestep
    
    ### Integration data
    i = 0; i_max = 1e5 # iteration counter
    k1 = []; k2 = []; k3 = []; k4 = [] # rk stage
    delta = np.zeros(len(x0))
    
    ### Soluton data
    x = x0
    x1 = np.zeros(len(x0))
    x2 = np.zeros(len(x0))

    ### Timestepping
    while(i < i_max):
        i += 1
        h = dt_max

        # Solve for full/half steps and perform adaptation
        delta = np.ones(len(x0))
        while(max(abs(delta)) >= .001):
            # Nonautonomous Runge-Kutta stages for dt=2h
            k1 = dxdt(x0,        t[-1]      )*2*h
            k2 = dxdt(x0 + k1/2, t[-1] + h  )*2*h
            k3 = dxdt(x0 + k2/2, t[-1] + h  )*2*h
            k4 = dxdt(x0 + k3,   t[-1] + 2*h)*2*h

            # Solution for 1 step with dt=2h
            x1 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6

            # 1st set of nonautonomous Runge-Kutta stages for dt=h
            k1 = dxdt(x0,        t[-1]      )*h
            k2 = dxdt(x0 + k1/2, t[-1] + h/2)*h
            k3 = dxdt(x0 + k2/2, t[-1] + h/2)*h
            k4 = dxdt(x0 + k3,   t[-1] + h  )*h

            # Solution for 1 step with dt=h
            x2 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6           

            # 2nd set of nonautonomous Runge-Kutta stages for dt=h
            k1 = dxdt(x2,        t[-1]      )*h
            k2 = dxdt(x2 + k1/2, t[-1] + h/2)*h
            k3 = dxdt(x2 + k2/2, t[-1] + h/2)*h
            k4 = dxdt(x2 + k3,   t[-1] + h  )*h

            # Solution for 2nd step with dt=h
            x2 = x2 + (k1 + 2*k2 + 2*k3 + k4)/6  
            
            # Calculate relative error
            delta = x2 - x1
                
            # Bisect timestep
            if h/2 >= dt_min:
                h = h/2
            elif h/2 < dt_min:
                h = dt_min
                #print("Aborted rk4_adapt: Underwrote timestep trying to meet tolerances.")
                #break
                
            
        # Accept timestep & solution
        t.append(t[-1] + 4*h)
        x0 = x2; x = np.vstack([x, x2])
                
        # Check nominal timeout
        if t[-1] + 4*h > tf: 
            print("Nominal integration timeout.")
            break

        # Check for max iterations
        if i >= i_max:
            print("Aborted rk4_adapt(): Max number of iterations reached.")
            break
        
    return t, np.transpose(x)