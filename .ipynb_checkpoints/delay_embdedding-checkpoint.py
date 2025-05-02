"""
Embedding / delay reconstruction methods.

"""

import numpy as np


def downsample(x, M):
    """
    Returns ownsampled data set x with spacing M.
    
    """
    x_sampled = np.zeros(len(x)//M)
    
    for i in range(len(x_sampled)): 
        x_sampled[i] = x[i*M]
        
    return x[::M]


def embed(t, x, tau, m):
    """
    Generates delay reconstruction of time series data.
    
    Args:
        t (np.ndarray): Time data
        x (np.ndarray): Time series data
        tau (float): Delay time
        
    Returns:
        np.ndarray: Delay reconstruction
    
    """
    
    ### Sizing
    n_dt = int(tau/(t[1] - t[0])) # step size
    N = len(x) - m*n_dt # max length of embedding
    x_emb = np.zeros((N, m))
    
    ### Embedding
    for n in range(N):
        for i in range(m):
            x_emb[n, i] = x[n + i*n_dt]
    
    return x_emb
    

def extract(x_emb, j, k):
    """
    Extracts jth and kth elements from embedding.
    
    """
    
    ### Sizing
    N = np.shape(x_emb)[0]
    j_vec = np.zeros(N)
    k_vec = np.zeros(N)
    
    ### Reconstructing
    for n in range(N):
        j_vec[n] = x_emb[n, j]
        k_vec[n] = x_emb[n, k]
    
    return j_vec, k_vec
