import numpy as np
from mpi4py import MPI
from numpy import kron, matmul, transpose
from numpy.fft import fft, ifft
from numpy.linalg import inv
from FAS_correction_test import FAS
from int_nodes_test import int_nodes
from spectral_int_matrix_test import spectral_int_matrix
from transfer_operators_test import interpolation, restriction
from differential_operators_test import differentialA
from resolved_run_test import single_SDC
from sweep_test import coarse_sweep, fine_sweep
#np.set_printoptions(precision=20)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')
import sys

def pfasst(comm, dt, dtc, dtf, func, K, L, nG, nxc, nxf, nu, Mc, Mf, prediction_on, rank, size, T, tc, tf, typeODE, u0c, u0f, v, xc, xf):
    
    # initialization of vector of solution values and vector of function values of solution values
    uf = np.zeros(nxf * Mf, dtype='float')
    uc = np.zeros(nxc * Mc, dtype='float')
    uc_init = np.zeros(nxc * Mc, dtype='float')
    uhat_solve = np.zeros(nxf * Mf, dtype='cfloat')
    
    # spread initial condition to each collocation node -- restrict fine vector to yield coarse vector
    if rank == 0:
        for m in range(0, Mf):
            uf[m*nxf:m*nxf+nxf] = u0f
            
        uc = restriction(uf, Mc, nxc, Mf, nxf)
        uc_init = uc
        
    else:
        uf = None
        uc = None
        uc_init = None
           
    uf = comm.bcast(uf, root=0)
    uc = comm.bcast(uc, root=0)
    uc_init = comm.bcast(uc_init, root=0)
    
    # define the (constant) coefficient matrices and spread to all processors/ranks
    if rank == 0:
        AEf, AIf = differentialA(L, nu, nxf, typeODE, v)
        AEc, AIc = differentialA(L, nu, nxc, typeODE, v)
        
    else:
        AEf = None
        AIf = None
        AEc = None
        AIc = None

    AEf = comm.bcast(AEf, root=0)
    AIf = comm.bcast(AIf, root=0)
    AEc = comm.bcast(AEc, root=0)
    AIc = comm.bcast(AIc, root=0)

    
    # spectral integration matrices SE, SI on fine and on coarse level
    Qf, QEf, QIf, Sf, SEf, SIf = spectral_int_matrix(Mf, dt, dtf, tf[rank*Mf:rank*Mf+Mf])
    Qc, QEc, QIc, Sc, SEc, SIc = spectral_int_matrix(Mc, dt, dtc, tc[rank*Mc:rank*Mc+Mc])

    # FAS correction term
    tau = FAS(AIc, AIf, AEc, AEf, dt, func, Mc, nxc, Mf, nxf, nu, Qf, Qc, Sf, Sc, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf+Mf], typeODE, uc, uf, xc, xf)
       
    uc_MTilde = np.zeros(nxc, dtype='float')
    uc_MTilde = u0c
    
    # initial value for the resolved run (single-SDC on fine grid)
    u_M_solve = np.zeros(nxf, dtype='float')
    u_M_solve = u0f
    
    
    # Prediction phase
    if prediction_on == True:
    

        for j in range(0, rank+1):
    
            # Step (1)
            # Default tag j-1
            if (j > 0 and rank > 0):
                uc_MTilde = comm.recv(source=rank-1, tag=j-1)
            
            else:
                uc_MTilde = u0c
            
            # Step (2) - Coarse SDC sweep
            uc = coarse_sweep(AEc, AIc, dt, dtc, func, Mc, nG, nu, nxc, Sc, Qc, QEc, QIc, tau, tc[rank*Mc:rank*Mc+Mc], typeODE, uc, uc_MTilde, xc)
                   
            # Step (3) Default tag j   
            if rank < (size - 1):
                comm.send(uc[Mc*nxc-nxc:Mc*nxc], dest=rank+1, tag=j)
            
        uf = uf + interpolation(uc - uc_init, u0c - uc_MTilde, dtf, Mc, nxc, Mf, nxf, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf + Mf])
    
        uf = fine_sweep(AEf, AIf, dt, dtf, func, Mf, 1, nu, nxf, Sf, Qf, QEf, QIf, tf[rank*Mf:rank*Mf+Mf], typeODE, uf, u0f, xf)
        
    else:
        pass
    
    
    # single SDC on fine level
    #if K > 0:
    #    for j in range(0, rank+1):
    #        uhat_solve = single_SDC(AEf, AIf, dtf, func, K, L, Mf, nu, nxf, rank, dt, typeODE, tf[j*Mf:j*Mf+Mf], ufhat, u_M_solve, xf)
            
    #        u_M_solve = uhat_solve[Mf*nxf - nxf:Mf*nxf]
    
    # PFASST ITERATIONS
    for k in range(1, K+1):
        
        # Coarse level
        
        # restrict the fine values
        uc = restriction(uf, Mc, nxc, Mf, nxf)
        
        # FAS correction
        tau_PFASST = FAS(AIc, AIf, AEc, AEf, dt, func, Mc, nxc, Mf, nxf, nu, Qf, Qc, Sf, Sc, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf+Mf], typeODE, uc, uf, xc, xf)
        
        # receive coarse initial value from previous process
        if rank > 0:
            uc_MTilde = comm.recv(source=rank-1, tag=100*k)
            
        else:
            uc_MTilde = u0c
            
        # nG coarse SDC sweeps
        uc_prime = coarse_sweep(AEc, AIc, dt, dtc, func, Mc, nG, nu, nxc, Sc, Qc, QEc, QIc, tau_PFASST, tc[rank*Mc:rank*Mc+Mc], typeODE, uc, uc_MTilde, xc)
        
        # send initial value to next process
        if rank < (size - 1):
            comm.send(uc_prime[Mc*nxc-nxc:Mc*nxc], dest=rank+1, tag=100*k)
            
            
        # Fine level
        
        # interpolate coarse correction
        delta = interpolation(uc_prime - uc, np.zeros(nxc), dtf, Mc, nxc, Mf, nxf, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf + Mf])
        
        uf_prime = uf + delta
        #print(rank, type(delta[Mf*nxf-1]))
        
        # interpolate coarse correction
        #delta1 = interpolation(uchat_prime[0:nxc] - uchat[0:nxc], np.zeros(nxc), dtf, Mc, nxc, Mf, nxf, tf[rank*Mf:rank*Mf + Mf])
        delta1 = interpolation(uc_prime[Mc*nxc-nxc:Mc*nxc] - uc[Mc*nxc-nxc:Mc*nxc], np.zeros(nxc), dtf, Mc, nxc, Mf, nxf, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf + Mf])
        
        # send initial value to next process
        if rank < (size - 1):
            comm.send(uf_prime[Mf*nxf-nxf:Mf*nxf], dest=rank+1, tag=1+100*k)
        
        # receive new fine initial value from previous process
        if rank > 0:
            uf_M = comm.recv(source=rank-1, tag=1+100*k)
            
        else:
            uf_M = u0f
        
        #print(rank, type(uf_prime[Mf*nxf-nxf]))
        
        uf_prime[Mf*nxf-nxf:Mf*nxf] = uf[Mf*nxf-nxf:Mf*nxf] + delta1
        
        # fine Sweep
        uf = fine_sweep(AEf, AIf, dt, dtf, func, Mf, 1, nu, nxf, Sf, Qf, QEf, QIf, tf[rank*Mf:rank*Mf+Mf], typeODE, uf_prime, uf_M, xf)
            
            
    # returns only the solution at end of sub intervals
    uc_M = uc[Mc * nxc - nxc:Mc * nxc]
    uf_M = uf[Mf * nxf - nxf:Mf * nxf]
    
    return AIf, AIc, uf_M, uc_M, u_M_solve
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        