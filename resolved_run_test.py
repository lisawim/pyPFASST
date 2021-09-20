import numpy as np
from differential_operators_test import differentialA
from spectral_int_matrix_test import spectral_int_matrix
from sweep_test import fine_sweep

def single_SDC(AE_solve, AI_solve, dt_solve, func, K, L, Mf, nu, nxf, rank, dt, typeODE, t_solve, uhat, u_M_solve, xf):
    
    Q_solve, QE_solve, QI_solve, S_solve, SE_solve, SI_solve = spectral_int_matrix(Mf, dt, dt_solve, t_solve)
    
    uhat_solve = np.zeros(Mf * nxf, dtype='cfloat')
    
    # Change the iterations for F serial
    uhat_solve = fine_sweep(AE_solve, AI_solve, dt, dt_solve, func, Mf, K, nu, nxf, S_solve, Q_solve, t_solve, typeODE, uhat, u_M_solve, xf)
    
    return uhat_solve