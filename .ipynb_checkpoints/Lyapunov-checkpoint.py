"""
Methods to calculate maximum Lyapunov exponent of time series
data via Wolf's algorithm.

"""

import numpy as np


def nearest(x, i, epsilon):
    """
    Stupid (linear) nearest-neighbor search.
    
    Args:
        x (np.ndarray): Data set
        i (float): Search point index
        epsilon (float): Distance tolerance
    """
    
    L = []
    indicies = []
    
    # Find points within a radius epsilon of x
    for i_diff in range(20, len(x)):
        
        # Calc new distance
        L_new = np.linalg.norm(x[i,:] - x[i_diff,:])
        
        if L_new <= epsilon:
            L.append(L_new)
            indicies.append(i_diff)
        
    # Check if no neighbor was found
    if not L: return -1, -1
        
    # Closest point
    i_min = L.index(min(L))
    return L[i_min], indicies[i_min]


def nearest_dir(x, i_fid, i_diff, epsilon, theta):
    """
    Directional neares-neighbor search.
    
    Args:
        x (np.ndarray): Data set
        i_fid (int): Fiduciary trajectory index
        i_diff (int): Difference trajectory index
        epsilon (float): Distance tolerance
        theta (float): Angular tolerance
    """
    
    L = []
    indicies = []
    
    ### Vector to old difference curve
    a = x[i_diff] - x[i_fid]
    
    ### Find points within sperical cone defined by epsilon and theta
    for i in range(len(x)):
        
        # Ignore self and directly adjacent stuff
        if i_fid-5 < i < i_fid+5: continue
        
        # Calc distance from point on fiduciary curve
        r = np.linalg.norm(x[i,:] - x[i_fid,:])
        
        # Calc pitch angle from L' vector
        b = x[i] - x[i_fid] # vector to new difference curve
        cos_angle = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Acceptance criteria
        if r <= epsilon and angle <= theta:
            L.append(r)
            indicies.append(i)
                      
    # Check if no neighbor was found
    if not L: return -1, -1

    # Pull closest point
    i_min = L.index(min(L))
    return L[i_min], indicies[i_min]
        
    

def Wolf(t, x, epsilon, theta, i0=0):
    """
    Employs Wolf's algorithm to calculate largest
    Lyapunov exponent from data set.
    
    Args:
        t (np.ndarray): time data
        x (np.ndarray): time series data
        epsilon (float): nearest neighbor nearness tolerance
        theta (float): nearest neighbor directionality tolerance
        i0 (int): Starting index to truncate transient
        
    Returns:
        float: Maximum Lyapunov exponent
    """

    ### Truncate transient
    x = x[i0:len(t)-1]
    N = len(x)
    
    ### Preallocate
    L = []
    L_prime = []
    indicies = []
    
    ### Get first neighbor
    L_new, i_diff = nearest(x, 0, epsilon)
    
    ### Check that first neighbor is real
    if i_diff == -1:
        print("No suitable initial neighbor found. Update search criteria.")
        return [], [], [], np.nan
    
    L.append(L_new)
    indicies.append(i_diff)
    
    ### Step through trajectory
    i = 0
    while i < N - 1 and i_diff < N - 1:
        
        # Compute next difference trajectory size
        L_prime_new = np.linalg.norm(x[i_diff,:] - x[i,:])
        
        # Detect replacement step
        if L_prime_new > epsilon:
            L_prime.append(L_prime_new)
            
            # find new difference trajectory
            L_new, i_diff = nearest_dir(x, i, i_diff, epsilon, theta)
            
            # abandon if suitable neighbor found
            if i_diff == -1:
                print("No suitable neighbor found. Update search criteria.")
                return np.array(L), np.array(L_prime), indicies, np.nan
            
            L.append(L_new)
            indicies.append(i_diff)
        
        # Incrementing
        i += 1 # fiduciary trajectory
        i_diff += 1 # difference trajectory

    ### Pop trailing norm
    L.pop()
    indicies.pop()
    
    ### Calculate largest Lyapunov exponent
    L = np.array(L)
    L_prime = np.array(L_prime)
    lambda_1 = np.sum(np.log2(L_prime/L))/t[N]
    
    return lambda_1
