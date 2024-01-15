import numpy as np 
# relevant calculations 

def shannon_entropy(p):
    return -np.sum(np.where(p != 0, p * np.log2(p), 0))

def kl_divergence(p, q): # assuming p and q are numpy arrays (discrete)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def coarse_entropy():
    pass  