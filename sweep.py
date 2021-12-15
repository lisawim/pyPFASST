import numpy as np
from numpy import kron, transpose, matmul
from numpy.fft import fft, ifft
from numpy.linalg import inv
from residual import residual
import scipy
import sys
import warnings
warnings.filterwarnings('ignore')

    

def coarse_sweep(AEc, AIc, dt, dtc, func, Mc, nG, nu, nxc, Sc, Qc, QEc, QIc, tau, tc_int, typeODE, ucIter, u0c, xc):
    
    """
    Input:
        AEc          -    Operator for gradient on coarse level
        AIc          -    Operator for Laplacian on coarse level
        dt           -    time step
        dtc          -    sub time step on coarse level
        func         -    function which is used for initial condition
        Mc           -    number of coarse collocation nodes
        nG           -    number of coarse sweeps
        nu           -    diffusion coefficient
        nxc          -    number of coarse DoF
        Sc           -    coarse spectral integration matrix 'node to node'
        Qc           -    coarse integration matrix 't0 to node'
        QEc          -    coarse explicit integration matrix
        QIc          -    coarse implicit integration matrix
        tau          -    FAS correction
        tc_int       -    time interval on coarse level
        typeODE      -    equation to be solved
        ucIter       -    current vector of u on coarse level
        u0c          -    initial condition on coarse level
        xc           -    x-values on coarse level
        
    Return:
        Solution after nG coarse sweeps
    """
    
    def rhs(func, nu, x, t):
        
        """
        Input:
            func         - function for initial condition
            nu           - diffusion coefficient
            x            - x-values
            t            - time
            
        Return:
            the nomhomogeneous in right-hand side of ODE
        """
        
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='float')
        sigma = 0.004
        
        if func == 'exp':
            # RHS for Burgers
            u = np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            dt_u = -np.exp(-((x-0.5)**2)/sigma) * np.sin(t)
            dx_u = -((2*(x-0.5))/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            ddx_u = (((4*(x-0.5)**2)/sigma**2) - 2/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
        
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
            
        elif func == 'poly':
            u = x**4 * (1-x)**4 * np.cos(t)
            dt_u = -x**4 * (1-x)**4 * np.sin(t)
            dx_u = -4 * (1-x)**3 * x**3 * (2*x-1) * np.cos(t)
            ddx_u = 4 * (1-x)**2 * x**2 * (14*x**2 - 14*x + 3) * np.cos(t)
            
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
                       
        elif func == 'sin_heat':
            arr = -np.sin(np.pi * 4 * x) * (np.sin(t) - nu * (np.pi * 4) ** 2 * np.cos(t))
            
        return arr

    # Compute nonhomogeneous term for right-hand side of ODE
    rhsFc = np.zeros(nxc * Mc, dtype='float')
    
    for l in range(0, Mc):
        rhsFc[l*nxc:l*nxc + nxc] = rhs(func, nu, xc, tc_int[l])
        
    # Coarse SDC sweep  
    if typeODE == 'heat':
        uc_new = np.zeros(Mc * nxc, dtype='float')
        fI = np.zeros(Mc*nxc, dtype='float')
        rhs = np.zeros(nxc, dtype='float')
        
        # nG coarse sweeps
        for n in range(0, nG):
            
            for m in range(0, Mc):
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(ucIter[m*nxc:m*nxc+nxc])))
                
            # Compute integrals with subtraction
            Qkmc = np.zeros(Mc * nxc, dtype='float')
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fI[j*nxc:j*nxc+nxc]
                    
                    Qkmc[l*nxc:l*nxc+nxc] -= dt * QIc[l, j] * fI[j*nxc:j*nxc+nxc]
            
        
            for m in range(0, Mc):
                
                # Compute right-hand side for linear system 
                tmp = u0c + tau[m*nxc:m*nxc+nxc] + Qkmc[m*nxc:m*nxc+nxc]
                
                for j in range(0, m):
                    tmp += dt * QIc[m, j] * fI[j*nxc:j*nxc+nxc]
                    
                uc_new[m*nxc:m*nxc+nxc] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIc[m, m] * np.diag(AIc))))
            
                # new implicit function values
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(uc_new[m*nxc:m*nxc+nxc])))
        
            ucIter = uc_new
            
    elif typeODE == 'heat_forced':
        uc_new = np.zeros(Mc * nxc, dtype='float')
        fE = np.zeros(Mc*nxc, dtype='float')
        fI = np.zeros(Mc*nxc, dtype='float')
        rhs = np.zeros(nxc, dtype='float')
        
        # nG coarse sweeps           
        for n in range(0, nG):
            
            for m in range(0, Mc):
                fE[m*nxc:m*nxc+nxc] = rhsFc[m*nxc:m*nxc+nxc]
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(ucIter[m*nxc:m*nxc+nxc])))

            # Compute integrals with subtraction
            Qkmc = np.zeros(Mc * nxc, dtype='float')
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * (fE[j*nxc:j*nxc+nxc] + fI[j*nxc:j*nxc+nxc])
                    
                    Qkmc[l*nxc:l*nxc+nxc] -= dt * (QEc[l, j] * fE[j*nxc:j*nxc+nxc] + QIc[l, j] * fI[j*nxc:j*nxc+nxc])

                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mc):
                
                # Compute right-hand side for linear system
                tmp = u0c + tau[m*nxc:m*nxc+nxc] + Qkmc[m*nxc:m*nxc+nxc]
                
                for j in range(0, m):
                    tmp += dt * ( QEc[m, j] * fE[j*nxc:j*nxc+nxc] + QIc[m, j] * fI[j*nxc:j*nxc+nxc] )
            
                uc_new[m*nxc:m*nxc+nxc] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIc[m, m] * np.diag(AIc))))
            
                # new explicit function values
                fE[m*nxc:m*nxc+nxc] = rhsFc[m*nxc:m*nxc + nxc] 
            
                # new implicit function values
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(uc_new[m*nxc:m*nxc+nxc])))
                
            ucIter = uc_new

        
    elif typeODE == 'Burgers':      
        # IMEX Version (Burgers' equation)- uses the iterative method with S and SE
        AEcuchat = np.zeros(nxc * Mc, dtype='cfloat')
        nsc = np.zeros(nxc * Mc, dtype='cfloat')
        nschat = np.zeros(nxc * Mc, dtype='float')
        Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
        uc_new = np.zeros(Mc * nxc, dtype='cfloat')
        fE = np.zeros(Mc*nxc, dtype='float')
        fI = np.zeros(Mc*nxc, dtype='float')
        newAEcuhat = np.zeros(nxc, dtype='cfloat')
        newnsc = np.zeros(nxc, dtype='cfloat')
        
        
        # nG coarse sweeps           
        for n in range(0, nG):
            
            # evaluation of the nonstiff term in spectral space
            for l in range(0, Mc):
                AEcuchat[l*nxc:l*nxc + nxc] = AEc.dot(fft(ucIter[l*nxc:l*nxc + nxc]))
                nsc[l*nxc:l*nxc + nxc] = -np.multiply(ucIter[l*nxc:l*nxc + nxc], ifft(AEcuchat[l*nxc:l*nxc + nxc]))
                nschat[l*nxc:l*nxc + nxc] = ifft(fft(nsc[l*nxc:l*nxc + nxc]))
            
            for m in range(0, Mc):
                fE[m*nxc:m*nxc+nxc] = nschat[m*nxc:m*nxc+nxc] + rhsFc[m*nxc:m*nxc+nxc]
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(ucIter[m*nxc:m*nxc+nxc])))

                
            # Compute integrals with subtraction
            Qkmc = np.zeros(Mc * nxc, dtype='float')
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * (fE[j*nxc:j*nxc+nxc] + fI[j*nxc:j*nxc+nxc])
                    
                    Qkmc[l*nxc:l*nxc+nxc] -= dt * (QEc[l, j] * fE[j*nxc:j*nxc+nxc] + QIc[l, j] * fI[j*nxc:j*nxc+nxc])
                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mc):   
                                                      
                # Compute right-hand side for linear system
                tmp = u0c + tau[m*nxc:m*nxc+nxc] + Qkmc[m*nxc:m*nxc+nxc]
                                                                      
                for j in range(0, m):
                    tmp += dt * ( QEc[m, j] * fE[j*nxc:j*nxc+nxc] + QIc[m, j] * fI[j*nxc:j*nxc+nxc] )
                                                                      
                uc_new[m*nxc:m*nxc+nxc] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIc[m, m] * np.diag(AIc))))                                      
            
                # new explicit function values
                newAEcuhat = AEc.dot(fft(uc_new[m*nxc:m*nxc+nxc]))
                newnsc = -np.multiply(uc_new[m*nxc:m*nxc+nxc], ifft(newAEcuhat))
                fE[m*nxc:m*nxc+nxc] = ifft(fft(newnsc)) + rhsFc[m*nxc:m*nxc+nxc] 
            
                # new implicit function values
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(uc_new[m*nxc:m*nxc+nxc])))
                
            ucIter = uc_new
            
    elif typeODE == 'Burgers_poly':      
        # IMEX Version (Burgers' equation)- uses the iterative method with S and SE
        AEcuchat = np.zeros(nxc * Mc, dtype='cfloat')
        nsc = np.zeros(nxc * Mc, dtype='cfloat')
        nschat = np.zeros(nxc * Mc, dtype='float')
        Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
        uc_new = np.zeros(Mc * nxc, dtype='cfloat')
        fE = np.zeros(Mc*nxc, dtype='float')
        fI = np.zeros(Mc*nxc, dtype='float')
        newAEcuhat = np.zeros(nxc, dtype='cfloat')
        newnsc = np.zeros(nxc, dtype='cfloat')
        
        
        # nG coarse sweeps           
        for n in range(0, nG):
            
            # evaluation of the nonstiff term in spectral space
            for l in range(0, Mc):
                AEcuchat[l*nxc:l*nxc + nxc] = AEc.dot(fft(ucIter[l*nxc:l*nxc + nxc]))
                nsc[l*nxc:l*nxc + nxc] = -np.multiply(ucIter[l*nxc:l*nxc + nxc], ifft(AEcuchat[l*nxc:l*nxc + nxc]))
                nschat[l*nxc:l*nxc + nxc] = ifft(fft(nsc[l*nxc:l*nxc + nxc]))
            
            for m in range(0, Mc):
                fE[m*nxc:m*nxc+nxc] = nschat[m*nxc:m*nxc+nxc] + rhsFc[m*nxc:m*nxc+nxc]
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(ucIter[m*nxc:m*nxc+nxc])))

                
            # Compute integrals with subtraction
            Qkmc = np.zeros(Mc * nxc, dtype='float')
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * (fE[j*nxc:j*nxc+nxc] + fI[j*nxc:j*nxc+nxc])
                    
                    Qkmc[l*nxc:l*nxc+nxc] -= dt * (QEc[l, j] * fE[j*nxc:j*nxc+nxc] + QIc[l, j] * fI[j*nxc:j*nxc+nxc])
                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mc):   
                                                      
                # Compute right-hand side for linear system
                tmp = u0c + tau[m*nxc:m*nxc+nxc] + Qkmc[m*nxc:m*nxc+nxc]
                                                                      
                for j in range(0, m):
                    tmp += dt * ( QEc[m, j] * fE[j*nxc:j*nxc+nxc] + QIc[m, j] * fI[j*nxc:j*nxc+nxc] )
                                                                      
                uc_new[m*nxc:m*nxc+nxc] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIc[m, m] * np.diag(AIc))))                                      
            
                # new explicit function values
                newAEcuhat = AEc.dot(fft(uc_new[m*nxc:m*nxc+nxc]))
                newnsc = -np.multiply(uc_new[m*nxc:m*nxc+nxc], ifft(newAEcuhat))
                fE[m*nxc:m*nxc+nxc] = ifft(fft(newnsc)) + rhsFc[m*nxc:m*nxc+nxc] 
            
                # new implicit function values
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(uc_new[m*nxc:m*nxc+nxc])))
                
            ucIter = uc_new

    elif typeODE == 'advdif':
        # IMEX Version (advection diffusion equation)- uses the iterative method with S and SE
        
        # Initialisations
        uc_new = np.zeros(Mc * nxc, dtype='float')
        fE = np.zeros(Mc * nxc, dtype='float')
        fI = np.zeros(Mc * nxc, dtype='float')
        rhs = np.zeros(nxc, dtype='float')
            
        # nG coarse sweeps    
        for n in range(0, nG):
            
            for m in range(0, Mc):
                fE[m*nxc:m*nxc+nxc] = ifft(AEc.dot(fft(ucIter[m*nxc:m*nxc+nxc])))
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(ucIter[m*nxc:m*nxc+nxc])))
                
                
            # Compute integrals with subtraction
            Qkmc = np.zeros(Mc * nxc, dtype='float')
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * (fE[j*nxc:j*nxc+nxc] + fI[j*nxc:j*nxc+nxc])
                    
                    Qkmc[l*nxc:l*nxc+nxc] -= dt * (QEc[l, j] * fE[j*nxc:j*nxc+nxc] + QIc[l, j] * fI[j*nxc:j*nxc+nxc])
            

            for m in range(0, Mc):  
                
                # Compute right-hand side for linear system
                tmp = u0c + tau[m*nxc:m*nxc+nxc] + Qkmc[m*nxc:m*nxc+nxc]
                
                for j in range(0, m):
                    tmp += dt * ( QEc[m, j] * fE[j*nxc:j*nxc+nxc] + QIc[m, j] * fI[j*nxc:j*nxc+nxc] )
                    
                uc_new[m*nxc:m*nxc+nxc] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIc[m, m] * np.diag(AIc))))
                
                # new explicit function values
                fE[m*nxc:m*nxc+nxc] = ifft(AEc.dot(fft(uc_new[m*nxc:m*nxc+nxc]))) 
            
                # new implicit function values
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(uc_new[m*nxc:m*nxc+nxc]))) 

            ucIter = uc_new
                
                   
    return uc_new


def fine_sweep(AEf, AIf, dt, dtf, func, Mf, nF, nu, nxf, Sf, Qf, QEf, QIf, tf_int, typeODE, ufIter, u0f, xf):
    
    """
    Input:
        AEf          -    Operator for gradient on fine level
        AIf          -    Operator for Laplacian on fine level
        dt           -    time step
        dtf          -    sub time step on fine level
        func         -    function which is used for initial condition
        Mf           -    number of fine collocation nodes
        nF           -    number of fine sweeps
        nu           -    diffusion coefficient
        nxf          -    number of fine DoF
        Sf           -    fine spectral integration matrix 'node to node'
        Qf           -    fine integration matrix 't0 to node'
        QEf          -    fine explicit integration matrix
        QIf          -    fine implicit integration matrix
        tf_int       -    time interval on fine level
        typeODE      -    equation to be solved
        ufIter       -    current vector of u on fine level
        u0f          -    initial condition on fine level
        xf           -    x-values on fine level
        
    Return:
        Solution after nF fine sweeps and the residual
    """
    
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='float')
        sigma = 0.004
        
        if func == 'exp':
            # RHS for Burgers
            u = np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            dt_u = -np.exp(-((x-0.5)**2)/sigma) * np.sin(t)
            dx_u = -((2*(x-0.5))/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            ddx_u = (((4*(x-0.5)**2)/sigma**2) - 2/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
        
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
            
        elif func == 'poly':            
            u = x**4 * (1-x)**4 * np.cos(t)
            dt_u = -x**4 * (1-x)**4 * np.sin(t)
            dx_u = -4 * (1-x)**3 * x**3 * (2*x-1) * np.cos(t)
            ddx_u = 4 * (1-x)**2 * x**2 * (14*x**2 - 14*x + 3) * np.cos(t)
            
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
            
        elif func == 'sin_heat':
            arr = -np.sin(np.pi * 4 * x) * (np.sin(t) - nu * (np.pi * 4) ** 2 * np.cos(t))
            
        return arr
    
    # Compute nonhomogeneous term for right-hand side of ODE   
    rhsFf = np.zeros(Mf * nxf, dtype='float')
    
    for l in range(0, Mf):
        rhsFf[l*nxf:l*nxf + nxf] = rhs(func, nu, xf, tf_int[l])
    
    # Fine sweep
    if typeODE == 'heat':
        uf_new = np.zeros(Mf * nxf, dtype='float')
        fI = np.zeros(Mf*nxf, dtype='float')
        rhs = np.zeros(nxf, dtype='float')
        
        # nG coarse sweeps
        for n in range(0, nF):
            
            for m in range(0, Mf):
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                
            # Compute integrals with subtraction
            Qkmf = np.zeros(Mf * nxf, dtype='float')
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fI[j*nxf:j*nxf+nxf]
                    
                    Qkmf[l*nxf:l*nxf+nxf] -= dt * QIf[l, j] * fI[j*nxf:j*nxf+nxf]
            
        
            for m in range(0, Mf):
                
                # Compute right-hand side for linear system
                tmp = u0f + Qkmf[m*nxf:m*nxf+nxf]
                
                for j in range(0, m):
                    tmp += dt * QIf[m, j] * fI[j*nxf:j*nxf+nxf]
                    
                uf_new[m*nxf:m*nxf+nxf] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIf[m, m] * np.diag(AIf))))
            
                # new implicit function values
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
        
            ufIter = uf_new
            
    elif typeODE == 'heat_forced':
        uf_new = np.zeros(Mf*nxf, dtype='float')
        fE = np.zeros(Mf*nxf, dtype='float')
        fI = np.zeros(Mf*nxf, dtype='float')
        rhs = np.zeros(nxf, dtype='float')
        
        # nG coarse sweeps           
        for n in range(0, nF):
            
            for m in range(0, Mf):
                fE[m*nxf:m*nxf+nxf] = rhsFf[m*nxf:m*nxf+nxf]
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))

            # Compute integrals with subtraction
            Qkmf = np.zeros(Mf * nxf, dtype='float')
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * (fE[j*nxf:j*nxf+nxf] + fI[j*nxf:j*nxf+nxf])
                    
                    Qkmf[l*nxf:l*nxf+nxf] -= dt * (QEf[l, j] * fE[j*nxf:j*nxf+nxf] + QIf[l, j] * fI[j*nxf:j*nxf+nxf])

                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mf):
                
                # Compute right-hand side for linear system
                tmp = u0f + Qkmf[m*nxf:m*nxf+nxf]
                
                for j in range(0, m):
                    tmp += dt * ( QEf[m, j] * fE[j*nxf:j*nxf+nxf] + QIf[m, j] * fI[j*nxf:j*nxf+nxf] )
            
                uf_new[m*nxf:m*nxf+nxf] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIf[m, m] * np.diag(AIf))))
            
                # new explicit function values
                fE[m*nxf:m*nxf+nxf] = rhsFf[m*nxf:m*nxf+nxf] 
            
                # new implicit function values
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
                
            ufIter = uf_new
        
    elif typeODE == 'Burgers':                                                                                          
        # IMEX Version (Burgers' equation)- uses the iterative method with S and SE
        
        # Initialisations
        AEfufhat = np.zeros(Mf * nxf, dtype='cfloat')
        nsf = np.zeros(Mf * nxf, dtype='cfloat')
        nsfhat = np.zeros(Mf * nxf, dtype='float')
        uf_new = np.zeros(Mf * nxf, dtype='cfloat')
        fE = np.zeros(Mf * nxf, dtype='float')
        fI = np.zeros(Mf * nxf, dtype='float')
        newAEfuhat = np.zeros(nxf, dtype='cfloat')
        newnsf = np.zeros(nxf, dtype='cfloat')
                    
        for n in range(0, nF):
                                                      
            # evaluation of the nonstiff term in spectral space
            for l in range(0, Mf):
                AEfufhat[l*nxf:l*nxf + nxf] = AEf.dot(fft(ufIter[l*nxf:l*nxf + nxf]))
                nsf[l*nxf:l*nxf + nxf] = -np.multiply(ufIter[l*nxf:l*nxf + nxf], ifft(AEfufhat[l*nxf:l*nxf + nxf]))
                nsfhat[l*nxf:l*nxf + nxf] = ifft(fft(nsf[l*nxf:l*nxf + nxf]))
            
            for m in range(0, Mf):
                fE[m*nxf:m*nxf+nxf] = nsfhat[m*nxf:m*nxf+nxf] + rhsFf[m*nxf:m*nxf+nxf]
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                

                
            # Compute integrals with subtraction
            Qkmf = np.zeros(Mf * nxf, dtype='float')
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * (fE[j*nxf:j*nxf+nxf] + fI[j*nxf:j*nxf+nxf])
                    
                    Qkmf[l*nxf:l*nxf+nxf] -= dt * (QEf[l, j] * fE[j*nxf:j*nxf+nxf] + QIf[l, j] * fI[j*nxf:j*nxf+nxf])
        
            #print(np.shape(u0f))       
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mf):   
                                                      
                # Compute right-hand side for linear system
                tmp = u0f + Qkmf[m*nxf:m*nxf+nxf]
                                                                      
                for j in range(0, m):
                    tmp += dt * ( QEf[m, j] * fE[j*nxf:j*nxf+nxf] + QIf[m, j] * fI[j*nxf:j*nxf+nxf] )
                                                                      
                uf_new[m*nxf:m*nxf+nxf] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIf[m, m] * np.diag(AIf))))                                      
            
                # new explicit function values
                newAEfuhat = AEf.dot(fft(uf_new[m*nxf:m*nxf+nxf]))
                newnsf = -np.multiply(uf_new[m*nxf:m*nxf+nxf], ifft(newAEfuhat))
                fE[m*nxf:m*nxf+nxf] = ifft(fft(newnsf)) + rhsFf[m*nxf:m*nxf+nxf] 
            
                # new implicit function values
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
                
            ufIter = uf_new
            
    elif typeODE == 'Burgers_poly':                                                                                          
        # IMEX Version (Burgers' equation)- uses the iterative method with S and SE
        
        # Initialisations
        AEfufhat = np.zeros(Mf * nxf, dtype='cfloat')
        nsf = np.zeros(Mf * nxf, dtype='cfloat')
        nsfhat = np.zeros(Mf * nxf, dtype='float')
        uf_new = np.zeros(Mf * nxf, dtype='cfloat')
        fE = np.zeros(Mf * nxf, dtype='float')
        fI = np.zeros(Mf * nxf, dtype='float')
        newAEfuhat = np.zeros(nxf, dtype='cfloat')
        newnsf = np.zeros(nxf, dtype='cfloat')
                    
        for n in range(0, nF):
                                                      
            # evaluation of the nonstiff term in spectral space
            for l in range(0, Mf):
                AEfufhat[l*nxf:l*nxf + nxf] = AEf.dot(fft(ufIter[l*nxf:l*nxf + nxf]))
                nsf[l*nxf:l*nxf + nxf] = -np.multiply(ufIter[l*nxf:l*nxf + nxf], ifft(AEfufhat[l*nxf:l*nxf + nxf]))
                nsfhat[l*nxf:l*nxf + nxf] = ifft(fft(nsf[l*nxf:l*nxf + nxf]))
            
            for m in range(0, Mf):
                fE[m*nxf:m*nxf+nxf] = nsfhat[m*nxf:m*nxf+nxf] + rhsFf[m*nxf:m*nxf+nxf]
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                

                
            # Compute integrals with subtraction
            Qkmf = np.zeros(Mf * nxf, dtype='float')
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * (fE[j*nxf:j*nxf+nxf] + fI[j*nxf:j*nxf+nxf])
                    
                    Qkmf[l*nxf:l*nxf+nxf] -= dt * (QEf[l, j] * fE[j*nxf:j*nxf+nxf] + QIf[l, j] * fI[j*nxf:j*nxf+nxf])
        
            #print(np.shape(u0f))       
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mf):   
                                                      
                # Compute right-hand side for linear system
                tmp = u0f + Qkmf[m*nxf:m*nxf+nxf]
                                                                      
                for j in range(0, m):
                    tmp += dt * ( QEf[m, j] * fE[j*nxf:j*nxf+nxf] + QIf[m, j] * fI[j*nxf:j*nxf+nxf] )
                                                                      
                uf_new[m*nxf:m*nxf+nxf] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIf[m, m] * np.diag(AIf))))                                      
            
                # new explicit function values
                newAEfuhat = AEf.dot(fft(uf_new[m*nxf:m*nxf+nxf]))
                newnsf = -np.multiply(uf_new[m*nxf:m*nxf+nxf], ifft(newAEfuhat))
                fE[m*nxf:m*nxf+nxf] = ifft(fft(newnsf)) + rhsFf[m*nxf:m*nxf+nxf] 
            
                # new implicit function values
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
                
            ufIter = uf_new
            
    elif typeODE == 'advdif':                                                           
        # IMEX Version (advection diffusion equation)- uses the iterative method with S and SE
        
        # Initialisations
        uf_new = np.zeros(Mf * nxf, dtype='float')
        fE = np.zeros(Mf * nxf, dtype='float')
        fI = np.zeros(Mf * nxf, dtype='float')
        rhs = np.zeros(nxf, dtype='float')
            
        for n in range(0, nF):
            
            for m in range(0, Mf):
                fE[m*nxf:m*nxf+nxf] = ifft(AEf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                
            # Compute integrals with subtraction
            Qkmf = np.zeros(Mf * nxf, dtype='float')
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * (fE[j*nxf:j*nxf+nxf]+ fI[j*nxf:j*nxf+nxf])
                    
                    Qkmf[l*nxf:l*nxf+nxf] -= dt * (QEf[l, j] * fE[j*nxf:j*nxf+nxf] + QIf[l, j] * fI[j*nxf:j*nxf+nxf])

                
            for m in range(0, Mf):    
                
                # Compute right-hand side for linear system
                tmp = u0f + Qkmf[m*nxf:m*nxf+nxf]
                
                for j in range(0, m):
                    tmp += dt * (QEf[m, j] * fE[j*nxf:j*nxf+nxf] + QIf[m, j] * fI[j*nxf:j*nxf+nxf])
                    
                uf_new[m*nxf:m*nxf+nxf] = ifft(fft(tmp) * (1.0 / (1.0 - dt * QIf[m, m] * np.diag(AIf))))
                
                
                # new explicit function values
                fE[m*nxf:m*nxf+nxf] = ifft(AEf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
            
                # new implicit function values
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
            
            ufIter = uf_new
            
    # Residual
    res = residual(dt, fE, fI, Mf, nxf, Qf, u0f, uf_new)
         
    return uf_new, res
