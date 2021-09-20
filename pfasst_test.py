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
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

def pfasst(comm, dt, dtc, dtf, func, K, L, nG, nxc, nxf, nu, Mc, Mf, rank, size, T, tc, tf, typeODE, u0hatc, u0hatf, v, xc, xf):
    sum = 0
    number = 0
    
    # initialization of vector of solution values and vector of function values of solution values
    ufhat = np.zeros(nxf * Mf, dtype='cfloat')
    uchat = np.zeros(nxc * Mc, dtype='cfloat')
    uchat_init = np.zeros(nxc * Mc, dtype='cfloat')
    uhat_solve = np.zeros(nxf * Mf, dtype='cfloat')
    
    # spread initial condition to each collocation node -- restrict fine vector to yield coarse vector
    if rank == 0:
        for m in range(0, Mf):
            ufhat[m*nxf:m*nxf+nxf] = u0hatf           
            
        uchat = restriction(ufhat, Mc, nxc, Mf, nxf)
        uchat_init = uchat
        
    else:
        ufhat = None
        uchat = None
        uchat_init = None
           
    ufhat = comm.bcast(ufhat, root=0)
    uchat = comm.bcast(uchat, root=0)
    uchat_init = comm.bcast(uchat_init, root=0)  
    
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
    tau = FAS(AIc, AIf, AEc, AEf, dt, func, Mc, nxc, Mf, nxf, nu, Qf, Qc, Sf, Sc, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf+Mf], typeODE, uchat, ufhat, xc, xf)
       
    uc_MTilde = np.zeros(nxc, dtype='cfloat')
    uc_MTilde = u0hatc
    
    # initial value for the resolved run (single-SDC on fine grid)
    u_M_solve = np.zeros(nxf, dtype='cfloat')
    u_M_solve = u0hatf
    
    # INITIALIZATION PROCEDURE
    for j in range(0, rank+1):
        sum = sum + 1
        # Step (1)
        if (j > 0 and rank > 0):
            uc_MTilde = comm.recv(source=rank-1, tag=j-1)
            
        else:
            uc_MTilde = u0hatc
            
        # Step (2) - Coarse SDC sweep
        uchat = coarse_sweep(AEc, AIc, dt, dtc, func, Mc, nG, nu, nxc, Sc, Qc, tau, tc[rank*Mc:rank*Mc+Mc], typeODE, uchat, uc_MTilde, xc)
        
        #if rank == 0:
        #    print("Rank 0")
        #    for m in range(0, Mc):
        #        uchat_m = ifft(uchat[m*nxc:m*nxc+nxc])
        #        print()
        #        print(uchat_m[:10])
                
        #if rank == 1:
        #    print("Rank 1")
        #    for m in range(0, Mc):
        #        uchat_m = ifft(uchat[m*nxc:m*nxc+nxc])
        #        print()
        #        print(uchat_m[:10])
            
        # Step (3)    
        if rank < (size - 1):
            comm.send(uchat[Mc*nxc-nxc:Mc*nxc], dest=rank+1, tag=j)
            
    
    # for each rank a resolved run for each substep with SDC is computed to calculate errors - the actual value uhat isn't required (for directIMEX)
    #if K > 0:
    #    for j in range(0, rank+1):
    #        uhat_solve = single_SDC(AEf, AIf, dtf, func, K, L, Mf, nu, nxf, rank, dt, typeODE, tf[j*Mf:j*Mf+Mf], ufhat, u_M_solve, xf)
            
    #        u_M_solve = uhat_solve[Mf*nxf - nxf:Mf*nxf]
    
    
    # Return value after the initialization procedure, if K=0 is chosen
    #if K == 0:
    #    ufhat = interpolation(uchat, u0hatc, dtf, Mc, nxc, Mf, nxf, tf[rank*Mf:rank*Mf + Mf])
        
    #else:
    #    ufhat = ufhat + interpolation(uchat - uchat_init, u0hatc - uc_MTilde, dtf, Mc, nxc, Mf, nxf, tf[rank*Mf:rank*Mf + Mf])
        
    ufhat = ufhat + interpolation(uchat - uchat_init, u0hatc - uc_MTilde, dtf, Mc, nxc, Mf, nxf, tf[rank*Mf:rank*Mf + Mf])
    
    ufhat = fine_sweep(AEf, AIf, dt, dtf, func, Mf, 1, nu, nxf, Sf, Qf, tf[rank*Mf:rank*Mf+Mf], typeODE, ufhat, u0hatf, xf)

    uf_M = np.zeros(nxf, dtype='cfloat')
    uf_M = u0hatf
    

    # PFASST ITERATIONS
    for k in range(1, K+1):
        # Step (1) - Perform one fine sweep
        ufhat_prime = fine_sweep(AEf, AIf, dt, dtf, func, Mf, 1, nu, nxf, Sf, Qf, tf[rank*Mf:rank*Mf+Mf], typeODE, ufhat, uf_M, xf)

        # Step (2) - Restrict values ufhat_prime
        uchat_prime = restriction(ufhat_prime, Mc, nxc, Mf, nxf)

        # Step (3) - Compute FAS correction tau
        tau_PFASST = FAS(AIc, AIf, AEc, AEf, dt, func, Mc, nxc, Mf, nxf, nu, Qf, Qc, Sf, Sc, tc[rank*Mc:rank*Mc+Mc], tf[rank*Mf:rank*Mf+Mf], typeODE, restriction(ufhat_prime, Mc, nxc, Mf, nxf), ufhat_prime, xc, xf)
        
        # Step (4) - Receive uf_M from processor n-1
        if rank > 0:
            uf_M = comm.recv(source=rank-1, tag=k)
            
        else:
            uf_M = u0hatf

        # Step (5) - Perform nG coarse sweeps 
        uchat = uchat_prime
        
        uchat = coarse_sweep(AEc, AIc, dt, dtc, func, Mc, nG, nu, nxc, Sc, Qc, tau_PFASST, tc[rank*Mc:rank*Mc+Mc], typeODE, uchat, restriction(uf_M, 1, nxc, 1, nxf), xc)

        # Step (6) - Interpolate last component of uchat_prime - uchat in space and add to ufhat_prime to yield ufhat
        corrcM = interpolation(uchat[Mc*nxc - nxc:Mc*nxc] - uchat_prime[Mc*nxc - nxc:Mc*nxc], np.zeros(nxc), dtf, Mc, nxc, Mf, nxf, tf[rank*Mf:rank*Mf + Mf]) 
        ufhat[Mf * nxf - nxf:Mf * nxf] = ufhat_prime[Mf * nxf - nxf:Mf * nxf] + corrcM

        # Step (7) - Send last component of ufhat to processor n+1
        if rank < (size - 1):
            comm.send(ufhat[Mf*nxf-nxf:Mf*nxf], dest=rank+1, tag=k)
                
        # Step (8) - Interpolate the difference uchat_prime-uchat at nodes 0 < m < M and add it to ufhat_prime to yield ufhat
        corrc = interpolation(uchat - uchat_prime, np.zeros(nxc), dtf, Mc, nxc, Mf, nxf, tf[rank*Mf:rank*Mf + Mf])
        ufhat[0:Mf * nxf - nxf] = ufhat_prime[0:Mf * nxf - nxf] + corrc[0:Mf * nxf - nxf]
    
    # returns only the solution at end of sub intervals
    uc_M = uchat[Mc * nxc - nxc:Mc * nxc]
    uf_M = ufhat[Mf * nxf - nxf:Mf * nxf]
    
    
    return AIf, AIc, uf_M, uc_M, u_M_solve
