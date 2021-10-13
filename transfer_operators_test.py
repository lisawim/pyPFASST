import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
import scipy
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def get_transfer_matrix_Q(f_nodes, c_nodes):
        """
        Helper routine to quickly define transfer matrices between sets of nodes (fully Lagrangian)
        Args:
            f_nodes: fine nodes
            c_nodes: coarse nodes
        Returns:
            matrix containing the interpolation weights
        """
        nnodes_f = len(f_nodes)
        nnodes_c = len(c_nodes)

        tmat = np.zeros((nnodes_f, nnodes_c))

        for i in range(nnodes_f):
            xi = f_nodes[i]
            for j in range(nnodes_c):
                den = 1.0
                num = 1.0
                for k in range(nnodes_c):
                    if k == j:
                        continue
                    else:
                        den *= c_nodes[j] - c_nodes[k]
                        num *= xi - c_nodes[k]
                tmat[i, j] = num / den

        return tmat

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
    
    ##################################################################################################
    
    Pcoll = get_transfer_matrix_Q(tf, tc)
    #Pcoll = get_transfer_matrix_Q(np.array([0.,0.17267316464601135,0.5,0.8273268353539887,1.]), np.array([0., 0.5, 1.]))
    
    ###################################################################################################
        
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
        
        tmp_uf = np.concatenate((uf0, np.concatenate((uf2, uf4), axis=0)), axis=0)
        
        #np.set_printoptions(precision=30)
        #print(Pcoll)
        #print('Prolong u-uold')
        #print(uf0[:10])
        #print()
        #print(uf2[:10])
        #print()
        #print(uf4[:10])
        #print()
        
        #print()
        #print(get_transfer_matrix_Q(np.array([0.,0.17267316464601135,0.5,0.8273268353539887,1.]), np.array([0., 0.5, 1.])))
        #print()
        
        
        ###############################################################################################
        
        #uInt = np.kron(Pcoll, np.identity(nxf)).dot(np.concatenate((uf0, np.concatenate((uf2, uf4), axis=0)), axis=0))
        
        for l in range(0, Mf):
            for j in range(0, Mc):
                #uInt[l*nxf:l*nxf+nxf] += Pcoll[l, j] * tmp_uf[j*nxf:j*nxf+nxf]
                uf[l*nxf:l*nxf+nxf] += Pcoll[l, j] * tmp_uf[j*nxf:j*nxf+nxf]
                
        uInt = uf
        
        ###############################################################################################
        
                       
        #for l in range(0, nxf):
        #    uf1[l] = polynomial_interpolation(tf[0], uf0[l], tf[2], uf2[l], tf[4], uf4[l], tf[1])
        #    uf3[l] = polynomial_interpolation(tf[0], uf0[l], tf[2], uf2[l], tf[4], uf4[l], tf[3])
        
        # The interpolated u's to the different time nodes will be collected
        #uf0uf1 = np.concatenate((uf0, uf1), axis=0)
        #uf0uf1uf2 = np.concatenate((uf0uf1, uf2), axis=0)
        #uf0uf1uf2uf3 = np.concatenate((uf0uf1uf2, uf3), axis=0)
        #uInt = np.concatenate((uf0uf1uf2uf3, uf4), axis=0)         
    
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

    R = np.zeros((nxc, nxf))
    
    Rcoll = get_transfer_matrix_Q(tc, tf)
    
    # matrix choose only low frequencies because the vector has components in spectral space (pseudospectral discretization)
    R[0:nxc//2, 0:nxc//2] = np.eye(nxc//2)
    R[(nxc - nxc//2):nxc, (nxf - nxc//2):nxf] = np.eye(nxc//2)
    R = 1/2 * R
    
    coarsening_factor = nxf//nxc
    
    
    if Mc == 1 and Mf == 1:
        uRestr = u[::coarsening_factor]
        #uRestr = R.dot(u)
    
    else:
        # Restriction in space
        #for l in range(0, Mf):
            #if l == 0:
            #    tmp_u = u[l*nxf:l*nxf+nxf]
            #    uRestr[l*nxc:l*nxc+nxc] = tmp_u[::coarsening_factor]
                
            #elif l%2 == 0 and l>0:
            #    tmp_u = u[l*nxf:l*nxf+nxf]
            #    uRestr[(l//2)*nxc:(l//2)*nxc+nxc] = tmp_u[::coarsening_factor]
                
            #else:
            #    continue 
            
            #tmp_u = u[l*nxf:l*nxf+nxf]
            #tmp_uf[l*nxc:l*nxc+nxc] = tmp_u[::coarsening_factor]
            
        #tmp_uf[0:nxc] = u[0:nxf][::coarsening_factor]
        #tmp_uf[nxc:2*nxc] = u[nxf:2*nxf][::coarsening_factor]
        #tmp_uf[2*nxc:3*nxc] = u[2*nxf:3*nxf][::coarsening_factor]
        #tmp_uf[3*nxc:4*nxc] = u[3*nxf:4*nxf][::coarsening_factor]
        #tmp_uf[4*nxc:5*nxc] = u[4*nxf:5*nxf][::coarsening_factor]
        
        tmp_uc[0:nxf] = u[0:nxf]
        tmp_uc[nxf:2*nxf] = u[2*nxf:3*nxf]
        tmp_uc[2*nxf:3*nxf] = u[4*nxf:5*nxf]
        
        #for m in range(0, Mc):
        #    print(tmp_uc[m*nxf:m*nxf+nxf][:10])
        #    print()
        
        uRestr[0:nxc] = tmp_uc[0:nxf][::coarsening_factor]
        uRestr[nxc:2*nxc] = tmp_uc[nxf:2*nxf][::coarsening_factor]
        uRestr[2*nxc:3*nxc] = tmp_uc[2*nxf:3*nxf][::coarsening_factor]
        
        #for m in range(0, Mc):
        #    print(uRestr[m*nxc:m*nxc+nxc][:10])
        #    print()
                

    return uRestr