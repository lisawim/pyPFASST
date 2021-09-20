import numpy as np
from numpy.fft import fft, ifft
import scipy
from scipy import signal

def interpolation(u, u_M, dtf, Mc, nxc, Mf, nxf, tf):
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
        
    uInt = np.zeros(nxf * Mf, dtype='cfloat')
        
    # First, a spatial interpolation is used with FFT
    uf0hat = np.zeros(nxf, dtype='cfloat')
    uf2hat = np.zeros(nxf, dtype='cfloat')
    uf4hat = np.zeros(nxf, dtype='cfloat')
    
    if np.shape(u)[0] == nxc:
        uInt = np.zeros(nxf, dtype='cfloat')
        
        # Only a spatial interpolation is done
        uInt[0:nxc//2] = 2*u[0:nxc//2]
        uInt[3*nxc//2:nxf] = 2*u[nxc//2:nxc]
        
    else:
        uf0hat[0:nxc//2] = 2*u[0:nxc//2]
        uf2hat[0:nxc//2] = 2*u[nxc:nxc+nxc//2]
        uf4hat[0:nxc//2] = 2*u[2*nxc:2*nxc+nxc//2]
        
        uf0hat[3*nxc//2:nxf] = 2*u[nxc//2:nxc]
        uf2hat[3*nxc//2:nxf] = 2*u[nxc+nxc//2:2*nxc]
        uf4hat[3*nxc//2:nxf] = 2*u[2*nxc+nxc//2:3*nxc]

        uf1hat = np.zeros(nxf, dtype='cfloat')
        uf3hat = np.zeros(nxf, dtype='cfloat')
                       
        for l in range(0, nxf):
            uf1hat[l] = polynomial_interpolation(tf[0], uf0hat[l], tf[2], uf2hat[l], tf[4], uf4hat[l], tf[1])
            uf3hat[l] = polynomial_interpolation(tf[0], uf0hat[l], tf[2], uf2hat[l], tf[4], uf4hat[l], tf[3])
        
        # The interpolated u's to the different time nodes will be collected
        uf0uf1 = np.concatenate((uf0hat, uf1hat), axis=0)
        uf0uf1uf2 = np.concatenate((uf0uf1, uf2hat), axis=0)
        uf0uf1uf2uf3 = np.concatenate((uf0uf1uf2, uf3hat), axis=0)
        uInt = np.concatenate((uf0uf1uf2uf3, uf4hat), axis=0)         
    
    return uInt

def restriction(u, Mc, nxc, Mf, nxf):
    # Input:
    # u        - the vector u which has the size nx + 1
    # Mc       - number of coarse intermediate nodes plus 1
    # nxc      - number of coarse spatial nodes
    # Mf       - number of fine intermediate nodes plus 1
    # nxf      - number of fine spatial nodes
    #
    # Output:
    # returns the restriction of vector u on a coarse level with injection
                
    uRestr = np.zeros(nxc * Mc, dtype='cfloat')

    R = np.zeros((nxc, nxf))
    
    # matrix choose only low frequencies because the vector has components in spectral space (pseudospectral discretization)
    R[0:nxc//2, 0:nxc//2] = np.eye(nxc//2)
    R[(nxc - nxc//2):nxc, (nxf - nxc//2):nxf] = np.eye(nxc//2)
    R = 1/2 * R
    ratio = nxf//nxc
    
    
    if Mc == 1 and Mf == 1:
        tmp_u = np.fft.ifft(u)
        uRestr = np.fft.fft(tmp_u[::ratio])
        #uRestr = R.dot(u)
    
    else:
        for l in range(0, Mf):
            if l == 0:
                tmp_u = np.fft.ifft(u[l*nxf:l*nxf+nxf])
                uRestr[l*nxc:l*nxc+nxc] = np.fft.fft(tmp_u[::ratio])
                #uRestr[l*nxc:l*nxc+nxc] = R.dot(u[l*nxf:l*nxf+nxf])
                
            elif l%2 == 0 and l>0:
                tmp2_u = np.fft.ifft(u[l*nxf:l*nxf+nxf])
                uRestr[(l//2)*nxc:(l//2)*nxc+nxc] = np.fft.fft(tmp2_u[::ratio])
                #uRestr[(l//2)*nxc:(l//2)*nxc+nxc] = R.dot(u[l*nxf:l*nxf+nxf])
                
            else:
                continue 
                

    return uRestr