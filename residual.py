import numpy as np
from numpy.fft import fft, ifft

def residual(dt, fE, fI, M, nx, Q, u0, u):
    
    """
    Input:
        dt           -    time step
        fE           -    explicit function values
        fI           -    implicit function values
        M            -    number of collocation nodes
        nx           -    degrees of freedom in space
        Q            -    integration matrix 't0 to node'
        u0           -    initial value
        u            -    current solution
        
    Return:
        Residual of current iteration
    """
    
    # Initialisation
    res_tmp = np.zeros(M * nx, dtype='cfloat')
    res_max_nodes = np.zeros(nx, dtype='float')
    res_max = []
    res = np.zeros(1, dtype='float')
    
        
    for m in range(0, M-1):
        for j in range(0, M):
            res_tmp[m*nx:m*nx+nx] += dt * Q[m+1, j] * (fE[j*nx:j*nx+nx] + fI[j*nx:j*nx+nx] )
    
    for m in range(0, M-1):
        res_tmp[m*nx:m*nx+nx] += u0 - u[(m+1)*nx:(m+1)*nx+nx]
        res_max_nodes[m] = abs(max(res_tmp[m*nx:m*nx+nx]))
        res_max.append(res_max_nodes[m])
        
    res = max(res_max)
    
    return res