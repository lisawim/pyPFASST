import numpy as np
import scipy as sp
from scipy.integrate import quad
from int_nodes_test import int_nodes

def spectral_int_matrix(M, dt, dt_int, t_int):
    # Input:
    # M       - size of the spectral integration matrix
    # dt      - time step of one full time step
    # dt_int  - time step between intermediate nodes
    # t_int   - consists of t with intermediate points t_m
    #
    # Output:
    # returns the spectral integration matrix, all of size M x M+1
    # Note that the spectral integration matrices are like the rectangle rule, because this yields lower triangular matrices
    
    def quad_lagrange(a, b, m, M, t):
        # Input:
        # a, b - the interval in which the Lagrange polynomial is computed
        # m    - the index that is skipped in the computation of the Lagrange polynomial
        # M    - the "length" of the Lagrange polynomial
        # t    - the nodes that are used for the computation
        #
        # Output:
        # returns the Lagrange polynomial as function
        def lagrange(s):
            prod = 1
            for k in range(0, M):
                if k == m:
                    continue
                prod *= (s - t[k])/(t[m] - t[k])
            return prod

        int = quad(lagrange, a, b)[0]

        return int
         
    S = np.zeros((M-1, M))
    S2 = np.zeros((M-1, M))
    SE = np.zeros((M-1, M))
    SI = np.zeros((M-1, M))
    Q = np.zeros((M, M)) # M x M
    QE = np.zeros((M, M))
    QI = np.zeros((M, M))
            
    # Matrix S does a 'node to node' integration, Q does a 't0 to node' integration    
    if M == 3:
        l0 = lambda x: (x-t_int[1])/(t_int[0]-t_int[1]) * (x-t_int[2])/(t_int[0]-t_int[2])
        l1 = lambda x: (x-t_int[0])/(t_int[1]-t_int[0]) * (x-t_int[2])/(t_int[1]-t_int[2])
        l2 = lambda x: (x-t_int[0])/(t_int[2]-t_int[0]) * (x-t_int[1])/(t_int[2]-t_int[1])
            
        Q[0, 0] = sp.integrate.quad(l0, t_int[0], t_int[0])[0]
        Q[0, 1] = sp.integrate.quad(l1, t_int[0], t_int[0])[0]
        Q[0, 2] = sp.integrate.quad(l2, t_int[0], t_int[0])[0]
            
        Q[1, 0] = sp.integrate.quad(l0, t_int[0], t_int[1])[0]
        Q[1, 1] = sp.integrate.quad(l1, t_int[0], t_int[1])[0]
        Q[1, 2] = sp.integrate.quad(l2, t_int[0], t_int[1])[0]
            
        Q[2, 0] = sp.integrate.quad(l0, t_int[0], t_int[2])[0]
        Q[2, 1] = sp.integrate.quad(l1, t_int[0], t_int[2])[0]
        Q[2, 2] = sp.integrate.quad(l2, t_int[0], t_int[2])[0]
        
        S[0, 0] = sp.integrate.quad(l0, t_int[0], t_int[1])[0]
        S[0, 1] = sp.integrate.quad(l1, t_int[0], t_int[1])[0]
        S[0, 2] = sp.integrate.quad(l2, t_int[0], t_int[1])[0]
            
        S[1, 0] = sp.integrate.quad(l0, t_int[1], t_int[2])[0]
        S[1, 1] = sp.integrate.quad(l1, t_int[1], t_int[2])[0]
        S[1, 2] = sp.integrate.quad(l2, t_int[1], t_int[2])[0]
        
            
    elif M == 5:
        l0 = lambda x: (x-t_int[1])/(t_int[0]-t_int[1]) * (x-t_int[2])/(t_int[0]-t_int[2]) * (x-t_int[3])/(t_int[0]-t_int[3]) * (x-t_int[4])/(t_int[0]-t_int[4])
        l1 = lambda x: (x-t_int[0])/(t_int[1]-t_int[0]) * (x-t_int[2])/(t_int[1]-t_int[2]) * (x-t_int[3])/(t_int[1]-t_int[3]) * (x-t_int[4])/(t_int[1]-t_int[4])
        l2 = lambda x: (x-t_int[0])/(t_int[2]-t_int[0]) * (x-t_int[1])/(t_int[2]-t_int[1]) * (x-t_int[3])/(t_int[2]-t_int[3]) * (x-t_int[4])/(t_int[2]-t_int[4])
        l3 = lambda x: (x-t_int[0])/(t_int[3]-t_int[0]) * (x-t_int[1])/(t_int[3]-t_int[1]) * (x-t_int[2])/(t_int[3]-t_int[2]) * (x-t_int[4])/(t_int[3]-t_int[4])
        l4 = lambda x: (x-t_int[0])/(t_int[4]-t_int[0]) * (x-t_int[1])/(t_int[4]-t_int[1]) * (x-t_int[2])/(t_int[4]-t_int[2]) * (x-t_int[3])/(t_int[4]-t_int[3])
            
        Q[0, 0] = sp.integrate.quad(l0, t_int[0], t_int[0])[0]
        Q[0, 1] = sp.integrate.quad(l1, t_int[0], t_int[0])[0]
        Q[0, 2] = sp.integrate.quad(l2, t_int[0], t_int[0])[0]
        Q[0, 3] = sp.integrate.quad(l3, t_int[0], t_int[0])[0]
        Q[0, 4] = sp.integrate.quad(l4, t_int[0], t_int[0])[0]
            
        Q[1, 0] = sp.integrate.quad(l0, t_int[0], t_int[1])[0]
        Q[1, 1] = sp.integrate.quad(l1, t_int[0], t_int[1])[0]
        Q[1, 2] = sp.integrate.quad(l2, t_int[0], t_int[1])[0]
        Q[1, 3] = sp.integrate.quad(l3, t_int[0], t_int[1])[0]
        Q[1, 4] = sp.integrate.quad(l4, t_int[0], t_int[1])[0]
            
        Q[2, 0] = sp.integrate.quad(l0, t_int[0], t_int[2])[0]
        Q[2, 1] = sp.integrate.quad(l1, t_int[0], t_int[2])[0]
        Q[2, 2] = sp.integrate.quad(l2, t_int[0], t_int[2])[0]
        Q[2, 3] = sp.integrate.quad(l3, t_int[0], t_int[2])[0]
        Q[2, 4] = sp.integrate.quad(l4, t_int[0], t_int[2])[0]
            
        Q[3, 0] = sp.integrate.quad(l0, t_int[0], t_int[3])[0]
        Q[3, 1] = sp.integrate.quad(l1, t_int[0], t_int[3])[0]
        Q[3, 2] = sp.integrate.quad(l2, t_int[0], t_int[3])[0]
        Q[3, 3] = sp.integrate.quad(l3, t_int[0], t_int[3])[0]
        Q[3, 4] = sp.integrate.quad(l4, t_int[0], t_int[3])[0]
            
        Q[4, 0] = sp.integrate.quad(l0, t_int[0], t_int[4])[0]
        Q[4, 1] = sp.integrate.quad(l1, t_int[0], t_int[4])[0]
        Q[4, 2] = sp.integrate.quad(l2, t_int[0], t_int[4])[0]
        Q[4, 3] = sp.integrate.quad(l3, t_int[0], t_int[4])[0]
        Q[4, 4] = sp.integrate.quad(l4, t_int[0], t_int[4])[0]
            
        S[0, 0] = sp.integrate.quad(l0, t_int[0], t_int[1])[0]
        S[0, 1] = sp.integrate.quad(l1, t_int[0], t_int[1])[0]
        S[0, 2] = sp.integrate.quad(l2, t_int[0], t_int[1])[0]
        S[0, 3] = sp.integrate.quad(l3, t_int[0], t_int[1])[0]
        S[0, 4] = sp.integrate.quad(l4, t_int[0], t_int[1])[0]
            
        S[1, 0] = sp.integrate.quad(l0, t_int[1], t_int[2])[0]
        S[1, 1] = sp.integrate.quad(l1, t_int[1], t_int[2])[0]
        S[1, 2] = sp.integrate.quad(l2, t_int[1], t_int[2])[0]
        S[1, 3] = sp.integrate.quad(l3, t_int[1], t_int[2])[0]
        S[1, 4] = sp.integrate.quad(l4, t_int[1], t_int[2])[0]
            
        S[2, 0] = sp.integrate.quad(l0, t_int[2], t_int[3])[0]
        S[2, 1] = sp.integrate.quad(l1, t_int[2], t_int[3])[0]
        S[2, 2] = sp.integrate.quad(l2, t_int[2], t_int[3])[0]
        S[2, 3] = sp.integrate.quad(l3, t_int[2], t_int[3])[0]
        S[2, 4] = sp.integrate.quad(l4, t_int[2], t_int[3])[0]
            
        S[3, 0] = sp.integrate.quad(l0, t_int[3], t_int[4])[0]
        S[3, 1] = sp.integrate.quad(l1, t_int[3], t_int[4])[0]
        S[3, 2] = sp.integrate.quad(l2, t_int[3], t_int[4])[0]
        S[3, 3] = sp.integrate.quad(l3, t_int[3], t_int[4])[0]
        S[3, 4] = sp.integrate.quad(l4, t_int[3], t_int[4])[0]
    
    
            
    S = (1/dt) * S    
    Q = (1/dt) * Q
    
    # SE, SI integrate 'node to node'
    #for i in range(0, M-1):
    #    for j in range(0, M):
    #        if j <= i:
                #SE[i, j] = dt_int[j]/dt
    #            SE[i, j] = dt_int[j]
                    
    #for i in range(0, M-1):
    #    for j in range(0, M):
    #        if j>0 and j<=(i+1):
                #SI[i, j] = dt_int[j-1]/dt
    #            SI[i, j] = dt_int[j-1]
    
    for i in range(0, M):
        for j in range(0, M):
            if j <= (i-1):
                QE[i, j] = dt_int[j]/dt
                
    for i in range(0, M):
        for j in range(0, M):
            if j>0 and j<=i:
                QI[i, j] = dt_int[j-1]/dt
                
    S2[0, :] = Q[1, :]
    
    for j in range(2, M):
        S2[j-1, :] = Q[j, :] - Q[j-1, :]
    
    
    #return Q, QE, QI, S, SE, SI
    return Q, QE, QI, S2, SE, SI
