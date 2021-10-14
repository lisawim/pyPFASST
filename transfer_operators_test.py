import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
import scipy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def interpolation(u, Mc, nxc, uf, Mf, nxf, tc, tf):
    # Input:
    # u        - the vector u which has the size nx + 1
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
    tmp_u = np.zeros((nxf//2)+1, dtype='cfloat')
        
    # First, a spatial interpolation is used with FFT
    tmp_uf0 = np.zeros((nxf//2)+1, dtype=np.complex128)
    tmp_uf2 = np.zeros((nxf//2)+1, dtype=np.complex128)
    tmp_uf4 = np.zeros((nxf//2)+1, dtype=np.complex128)
    tmp_uf = np.zeros(3*nxf, dtype='float')
    uf0 = np.zeros(nxf, dtype='float')
    uf1 = np.zeros(nxf, dtype='float')
    uf2 = np.zeros(nxf, dtype='float')
    uf3 = np.zeros(nxf, dtype='float')
    uf4 = np.zeros(nxf, dtype='float')
    
    
    if np.shape(u)[0] == nxc:

        uInt = np.zeros(nxf, dtype='float')
        
        tmp = rfft(u)
        
        tmp_u[0:nxc//2] = tmp[0:nxc//2]
        tmp_u[-1] = tmp[-1]
        
        uInt = irfft(tmp_u * 2)
        
    else:
        
        tmp_uc0 = rfft(u[0:nxc])
        tmp_uc2 = rfft(u[nxc:2*nxc])
        tmp_uc4 = rfft(u[2*nxc:3*nxc])
        
        tmp_uf0[0:nxc//2] = tmp_uc0[0:nxc//2]
        tmp_uf2[0:nxc//2] = tmp_uc2[0:nxc//2]
        tmp_uf4[0:nxc//2] = tmp_uc4[0:nxc//2]
        
        tmp_uf0[-1] = tmp_uc0[-1]
        tmp_uf2[-1] = tmp_uc2[-1]
        tmp_uf4[-1] = tmp_uc4[-1]
        
        uf0 = 2 * irfft(tmp_uf0)
        uf2 = 2 * irfft(tmp_uf2)
        uf4 = 2 * irfft(tmp_uf4)
        
                       
        for l in range(0, nxf):
            uf1[l] = polynomial_interpolation(tf[0], uf0[l], tf[2], uf2[l], tf[4], uf4[l], tf[1])
            uf3[l] = polynomial_interpolation(tf[0], uf0[l], tf[2], uf2[l], tf[4], uf4[l], tf[3])
        
        # The interpolated u's to the different time nodes will be collected
        uf0uf1 = np.concatenate((uf0, uf1), axis=0)
        uf0uf1uf2 = np.concatenate((uf0uf1, uf2), axis=0)
        uf0uf1uf2uf3 = np.concatenate((uf0uf1uf2, uf3), axis=0)
        uInt = np.concatenate((uf0uf1uf2uf3, uf4), axis=0)         
    
    return uInt

def restriction(u, Mc, nxc, Mf, nxf, tc, tf):
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
    tmp_uf = np.zeros(nxc * Mf, dtype='float')
    tmp_uc = np.zeros(nxf * Mc, dtype='float')
    
    # coarsening factor to restrict in space
    coarsening_factor = nxf//nxc
    
    
    if Mc == 1 and Mf == 1:
        uRestr = u[::coarsening_factor]
    
    else:
        # Restriction in space
        for l in range(0, Mf):
            if l == 0:
                tmp_u = u[l*nxf:l*nxf+nxf]
                uRestr[l*nxc:l*nxc+nxc] = tmp_u[::coarsening_factor]
                
            elif l%2 == 0 and l>0:
                tmp_u = u[l*nxf:l*nxf+nxf]
                uRestr[(l//2)*nxc:(l//2)*nxc+nxc] = tmp_u[::coarsening_factor]
                
            else:
                continue 
        
                

    return uRestr