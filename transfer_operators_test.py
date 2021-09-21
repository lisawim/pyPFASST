import numpy as np
from numpy.fft import fft, ifft
import scipy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def interpolation(u, u_M, dtf, Mc, nxc, Mf, nxf, tc, tf):
    # Input:
    # u        - the vector u which has the size nx + 1
    # u_M      - initial value at 0-th node
    # dtf      - time steps on fine level
    # Mc       - number of coarse intermediate nodes plus 1
    # nxc      - number of coarse spatial nodes
    # Mf       - number of fine intermediate nodes plus 1
    # nxf      - number of fine spatial nodes
    # tf       - temporal nodes on fine level
    #
    # Output:
    # returns the interpolation of vector u on fine level
    
    # for time interpolation - the Lagrange polynomial is used
    def polynomial_interpolation(tau0, u0, tau2, u2, tau4, u4, z):
        f = lambda x: u0 * ((x - tau2)/(tau0 - tau2)) * ((x - tau4)/(tau0 - tau4)) + u2 * ((x - tau0)/(tau2 - tau0)) * ((x - tau4)/(tau2 - tau4)) + u4 * ((x - tau0)/(tau4 - tau0)) * ((x - tau2)/(tau4 - tau2))
        return f(z)
    
        
    uInt = np.zeros(nxf * Mf, dtype='float')
    tmp_u = np.zeros(nxc * Mc, dtype='cfloat')
    tmp_uf = np.zeros(nxf, dtype=np.complex128)
        
    # First, a spatial interpolation is used with FFT
    uf0 = np.zeros(nxf, dtype='float')
    uf2 = np.zeros(nxf, dtype='float')
    uf4 = np.zeros(nxf, dtype='float')
    
    
    if np.shape(u)[0] == nxc:
        tmp_u = fft(u)
        uInt = np.zeros(nxf, dtype='float')
        
        #tmp_uf[0:nxc//2] = tmp_u[0:nxc//2]
        #tmp_uf[-1] = tmp_u[-1]
        
        #uInt = ifft(2*tmp_uf)
        
        # Only a spatial interpolation is done
        uInt[0:nxc//2] = tmp_u[0:nxc//2]
        uInt[3*nxc//2:nxf] = tmp_u[nxc//2:nxc]
        
        uInt = ifft(2*uInt)
        
    else:
        for m in range(0, Mc):
            tmp_u[m*nxc:m*nxc+nxc] = fft(u[m*nxc:m*nxc+nxc])
            
        uf0[0:nxc//2] = tmp_u[0:nxc//2]
        uf2[0:nxc//2] = tmp_u[nxc:nxc+nxc//2]
        uf4[0:nxc//2] = tmp_u[2*nxc:2*nxc+nxc//2]
        
        uf0[3*nxc//2:nxf] = tmp_u[nxc//2:nxc]
        uf2[3*nxc//2:nxf] = tmp_u[nxc+nxc//2:2*nxc]
        uf4[3*nxc//2:nxf] = tmp_u[2*nxc+nxc//2:3*nxc]
        
        #uf0[-1] = tmp_u[nxc-1]
        #uf2[-1] = tmp_u[2*nxc-1]
        #uf4[-1] = tmp_u[3*nxc-1]
        
        uf0 = ifft(2*uf0)
        uf2 = ifft(2*uf2)
        uf4 = ifft(2*uf4)

        uf1 = np.zeros(nxf, dtype='float')
        uf3 = np.zeros(nxf, dtype='float')
                       
        for l in range(0, nxf):
            uf1[l] = polynomial_interpolation(tf[0], uf0[l], tf[2], uf2[l], tf[4], uf4[l], tf[1])
            uf3[l] = polynomial_interpolation(tf[0], uf0[l], tf[2], uf2[l], tf[4], uf4[l], tf[3])
        
        # The interpolated u's to the different time nodes will be collected
        uf0uf1 = np.concatenate((uf0, uf1), axis=0)
        uf0uf1uf2 = np.concatenate((uf0uf1, uf2), axis=0)
        uf0uf1uf2uf3 = np.concatenate((uf0uf1uf2, uf3), axis=0)
        uInt = np.concatenate((uf0uf1uf2uf3, uf4), axis=0)         
    
    return uInt

def restriction(u, Mc, nxc, Mf, nxf):
    # Input:
    # u        - the vector u which has the size nxf or Mf*nxf
    # Mc       - number of coarse collocation nodes
    # nxc      - number of coarse spatial nodes
    # Mf       - number of fine collocation nodes
    # nxf      - number of fine spatial nodes
    #
    # Output:
    # returns the restriction of vector u on a coarse level with injection
                
    uRestr = np.zeros(nxc * Mc, dtype='float')

    R = np.zeros((nxc, nxf))
    
    # matrix choose only low frequencies because the vector has components in spectral space (pseudospectral discretization)
    R[0:nxc//2, 0:nxc//2] = np.eye(nxc//2)
    R[(nxc - nxc//2):nxc, (nxf - nxc//2):nxf] = np.eye(nxc//2)
    R = 1/2 * R
    
    coarsening_factor = nxf//nxc
    
    
    if Mc == 1 and Mf == 1:
        uRestr = u[::coarsening_factor]
        #uRestr = R.dot(u)
    
    else:
        for l in range(0, Mf):
            if l == 0:
                tmp_u = u[l*nxf:l*nxf+nxf]
                uRestr[l*nxc:l*nxc+nxc] = tmp_u[::coarsening_factor]
                #uRestr[l*nxc:l*nxc+nxc] = R.dot(u[l*nxf:l*nxf+nxf])
                
            elif l%2 == 0 and l>0:
                tmp_u = u[l*nxf:l*nxf+nxf]
                uRestr[(l//2)*nxc:(l//2)*nxc+nxc] = tmp_u[::coarsening_factor]
                #uRestr[(l//2)*nxc:(l//2)*nxc+nxc] = R.dot(u[l*nxf:l*nxf+nxf])
                
            else:
                continue 
                

    return uRestr