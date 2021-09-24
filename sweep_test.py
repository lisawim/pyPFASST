import numpy as np
from numpy import kron, transpose, matmul
from numpy.fft import fft, ifft
from numpy.linalg import inv
import scipy
import warnings
warnings.filterwarnings('ignore')

def coarse_sweep(AEc, AIc, dt, dtc, func, Mc, nG, nu, nxc, Sc, Qc, QEc, QIc, tau, tc_int, typeODE, ucIter, uc_MTilde, xc):
    
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='cfloat')
        sigma = 0.004
        
        if func == 'exp':
            # RHS for Burgers
            u = np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            dt_u = -np.exp(-((x-0.5)**2)/sigma) * np.sin(t)
            dx_u = -((2*(x-0.5))/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            ddx_u = (((4*(x-0.5)**2)/sigma**2) - 2/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
        
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
            #arr = -np.exp(-((x-0.5)**2)/0.004) * np.sin(t) + (np.exp(-((x-0.5)**2)/0.004) * np.cos(t))**2 * (-(2*(x-0.5))/0.004) - nu * np.cos(t) * (((2*(x-0.5))/0.004) * ((2*(x-0.5))/0.004) * np.exp(-((x-0.5)**2)/0.004) - 2/0.004 * np.exp(-((x-0.5)**2)/0.004))
            
        elif func == 'poly':
            arr = nu**2 * x**4 *(x**4 - 1) * np.sin(nu*t) + nu * x**4 * (1 - x**4) * np.cos(nu*t) * 4 * nu * x**3 * (1 - 2*x**4) * np.cos(nu*t) - nu**2 * 4 * x**2 * (3 - 14*x**4) * np.cos(nu*t)
            
        elif func == 'sin_heat':
            arr = -np.sin(np.pi * 4 * x) * (np.sin(t) - nu * (np.pi * 4) ** 2 * np.cos(t))
            
        return arr

    # Initialise solution vector and vector of right-hand side
    uc = np.zeros(nxc * Mc, dtype='cfloat')
    
    rhsFc = np.zeros(nxc * Mc, dtype='cfloat')
    
    for l in range(0, Mc):
        rhsFc[l*nxc:l*nxc + nxc] = fft(rhs(func, nu, xc, tc_int[l]))
        
    # Coarse SDC sweep  
    if typeODE == 'heat':
        uc_new = np.zeros(Mc * nxc, dtype='cfloat')
        fI = np.zeros(Mc*nxc, dtype='cfloat')
        
        # nG coarse sweeps
        for n in range(0, nG):
            # Compute integrals 'node to node'
            Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
            for l in range(0, Mc-1):
                for j in range(0, Mc):
                    Skmc[l*nxc:l*nxc+nxc] += dt * Sc[l, j] * (AIc.dot(ucIter[j*nxc:j*nxc+nxc]))
            
            uc_new[0:nxc] = uc_MTilde
        
            fI[0:nxc] = AIc.dot(uc_MTilde)
        
            for m in range(0, Mc-1):
                rhs = uc_new[m*nxc:m*nxc+nxc] + dtc[m] * (- AIc.dot(ucIter[(m+1)*nxc:(m+1)*nxc+nxc])) + Skmc[m*nxc:m*nxc+nxc] + tau[m*nxc:m*nxc+nxc]
            
                uc_new[(m+1)*nxc:(m+1)*nxc+nxc] = np.linalg.solve(np.identity(nxc) - dtc[m] * AIc, rhs) 
            
                # new implicit function values
                fI[(m+1)*nxc:(m+1)*nxc+nxc] = AIc.dot(uc_new[(m+1)*nxc:(m+1)*nxc+nxc])
        
            ucIter = uc_new
            
    elif typeODE == 'heat_forced':
        uc_new = np.zeros(Mc * nxc, dtype='cfloat')
        fE = np.zeros(Mc*nxc, dtype='cfloat')
        fI = np.zeros(Mc*nxc, dtype='cfloat')
        
        # nG coarse sweeps           
        for n in range(0, nG):
            # Compute integrals
            Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
            for m in range(0, Mc-1):
                for j in range(0, Mc):
                    Skmc[m*nxc:m*nxc+nxc] += dt * Sc[m, j] * (rhsFc[j*nxc:j*nxc + nxc] + AIc.dot(ucIter[j*nxc:j*nxc+nxc]))

            uc_new[0:nxc] = uc_MTilde
            
            fE[0:nxc] = rhsFc[0:nxc]
            
            fI[0:nxc] = AIc.dot(uc_MTilde)
                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mc-1):                   
                rhs = uc_new[m*nxc:m*nxc+nxc] + dtc[m] * (fE[m*nxc:m*nxc+nxc] - rhsFc[m*nxc:m*nxc + nxc] - AIc.dot(ucIter[(m+1)*nxc:(m+1)*nxc+nxc])) + Skmc[m*nxc:m*nxc+nxc] + tau[m*nxc:m*nxc+nxc]
            
                uc_new[(m+1)*nxc:(m+1)*nxc+nxc] = np.linalg.solve(np.identity(nxc) - dtc[m] * AIc, rhs)
            
                # new explicit function values
                fE[(m+1)*nxc:(m+1)*nxc+nxc] = rhsFc[(m+1)*nxc:(m+1)*nxc + nxc] 
            
                # new implicit function values
                fI[(m+1)*nxc:(m+1)*nxc+nxc] = AIc.dot(uc_new[(m+1)*nxc:(m+1)*nxc+nxc])
                
            ucIter = uc_new
        
    elif typeODE == 'Burgers':      
        # IMEX Version (Burgers' equation)- uses the iterative method with S and SE
        AEcuchat = np.zeros(nxc * Mc, dtype='cfloat')
        nsc = np.zeros(nxc * Mc, dtype='cfloat')
        nschat = np.zeros(nxc * Mc, dtype='cfloat')
        Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
        uc_new = np.zeros(Mc * nxc, dtype='cfloat')
        fE = np.zeros(Mc*nxc, dtype='cfloat')
        fI = np.zeros(Mc*nxc, dtype='cfloat')
        newAEcuhat = np.zeros(nxc, dtype='cfloat')
        newnsc = np.zeros(nxc, dtype='cfloat')
        
        #for m in range(0, Mc):
        #    print(ifft(nschat[m*nxc:m*nxc + nxc] + rhsFc[m*nxc:m*nxc + nxc])[:10])
        
        # nG coarse sweeps           
        for n in range(0, nG):
            # evaluation of the nonstiff term in spectral space
            for l in range(0, Mc):
                if l == 0:
                    AEcuchat[0:nxc] = AEc.dot(uc_MTilde)
                    nsc[0:nxc] = -np.multiply(ifft(uc_MTilde), ifft(AEcuchat[0:nxc]))
                    
                else:
                    AEcuchat[l*nxc:l*nxc + nxc] = AEc.dot(ucIter[l*nxc:l*nxc + nxc])
                    nsc[l*nxc:l*nxc + nxc] = -np.multiply(ifft(ucIter[l*nxc:l*nxc + nxc]), ifft(AEcuchat[l*nxc:l*nxc + nxc]))
                        
                nschat[l*nxc:l*nxc + nxc] = fft(nsc[l*nxc:l*nxc + nxc])
                
            # Compute integrals
            Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
            for m in range(0, Mc-1):
                for j in range(0, Mc):
                    Skmc[m*nxc:m*nxc+nxc] += dt * Sc[m, j] * (nschat[j*nxc:j*nxc + nxc] + rhsFc[j*nxc:j*nxc + nxc] + AIc.dot(ucIter[j*nxc:j*nxc+nxc]))

            uc_new[0:nxc] = uc_MTilde
            
            fE[0:nxc] = nschat[0:nxc]
            
            fI[0:nxc] = AIc.dot(uc_MTilde)
                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mc-1):                   
                rhs = uc_new[m*nxc:m*nxc+nxc] + dtc[m] * (fE[m*nxc:m*nxc+nxc] - nschat[m*nxc:m*nxc+nxc] - AIc.dot(ucIter[(m+1)*nxc:(m+1)*nxc+nxc])) + Skmc[m*nxc:m*nxc+nxc] + tau[m*nxc:m*nxc+nxc]
            
                uc_new[(m+1)*nxc:(m+1)*nxc+nxc] = np.linalg.solve(np.identity(nxc) - dtc[m] * AIc, rhs)
            
                # new explicit function values
                newAEcuhat = AEc.dot(uc_new[(m+1)*nxc:(m+1)*nxc+nxc])
                newnsc = -np.multiply(ifft(uc_new[(m+1)*nxc:(m+1)*nxc+nxc]), ifft(newAEcuhat))
                fE[(m+1)*nxc:(m+1)*nxc+nxc] = fft(newnsc) 
            
                # new implicit function values
                fI[(m+1)*nxc:(m+1)*nxc+nxc] = AIc.dot(uc_new[(m+1)*nxc:(m+1)*nxc+nxc])
                
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
                
            # Compute integrals
            #Skmc = np.zeros((Mc - 1) * nxc, dtype='cfloat')
            #for l in range(0, Mc-1):
            #    for j in range(0, Mc):
            #        Skmc[l*nxc:l*nxc+nxc] += dt * Sc[l, j] * (ifft(AEc.dot(fft(ucIter[j*nxc:j*nxc+nxc]))) + ifft(AIc.dot(fft(ucIter[j*nxc:j*nxc+nxc]))))
            
            Qkmc = np.zeros(Mc * nxc, dtype='float')
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * (fE[j*nxc:j*nxc+nxc] + fI[j*nxc:j*nxc+nxc])
                    
            
            for l in range(0, Mc):
                for j in range(0, Mc):
                    Qkmc[l*nxc:l*nxc+nxc] -= dt * (QEc[l, j] * fE[j*nxc:j*nxc+nxc] + QIc[l, j] * fI[j*nxc:j*nxc+nxc])
                    
                Qkmc[l*nxc:l*nxc+nxc] += uc_MTilde
                    
                Qkmc[l*nxc:l*nxc+nxc] += tau[l*nxc:l*nxc+nxc]
                                            
            #uc_new[0:nxc] = uc_MTilde
            
            #fE[0:nxc] = ifft(AEc.dot(fft(uc_MTilde)))     
            
            #fI[0:nxc] = ifft(AIc.dot(fft(uc_MTilde)))
            
            # sweep
            #for m in range(0, Mc-1):
            #    rhs = uc_new[m*nxc:m*nxc+nxc] + dtc[m] * (fE[m*nxc:m*nxc+nxc] - ifft(AEc.dot(fft(ucIter[m*nxc:m*nxc+nxc]))) - ifft(AIc.dot(fft(ucIter[(m+1)*nxc:(m+1)*nxc+nxc])))) + Skmc[m*nxc:m*nxc+nxc] + tau[m*nxc:m*nxc+nxc]
            
                #tmp = np.linalg.solve(np.identity(nxc) - dtc[m]*AIc, fft(rhs))
            #    tmp = fft(rhs) * (1.0 / (1.0 - dtc[m] * np.diag(AIc)))
            #    uc_new[(m+1)*nxc:(m+1)*nxc+nxc] = ifft(tmp)
            
                # new explicit function values
            #    fE[(m+1)*nxc:(m+1)*nxc+nxc] = ifft(AEc.dot(fft(uc_new[(m+1)*nxc:(m+1)*nxc+nxc]))) 
            
                # new implicit function values
            #    fI[(m+1)*nxc:(m+1)*nxc+nxc] = ifft(AIc.dot(fft(uc_new[(m+1)*nxc:(m+1)*nxc+nxc])))
            

            for m in range(0, Mc):  
                rhs = Qkmc[m*nxc:m*nxc+nxc]
                
                for j in range(0, m):
                    rhs += dt * ( QEc[m, j] * fE[j*nxc:j*nxc+nxc] + QIc[m, j] * fI[j*nxc:j*nxc+nxc] )
                    
                tmp = np.linalg.solve(np.identity(nxc) - dt * QIc[m, m] * AIc, fft(rhs))
                uc_new[m*nxc:m*nxc+nxc] = ifft(tmp)
                
                # new explicit function values
                fE[m*nxc:m*nxc+nxc] = ifft(AEc.dot(fft(uc_new[m*nxc:m*nxc+nxc]))) 
            
                # new implicit function values
                fI[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(uc_new[m*nxc:m*nxc+nxc]))) 
                
            ucIter = uc_new
                
                   
    return uc_new


def fine_sweep(AEf, AIf, dt, dtf, func, Mf, nF, nu, nxf, Sf, Qf, QEf, QIf, tf_int, typeODE, ufIter, uf_M, xf):
    
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='cfloat')
        sigma = 0.004
        
        if func == 'exp':
            # RHS for Burgers
            u = np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            dt_u = -np.exp(-((x-0.5)**2)/sigma) * np.sin(t)
            dx_u = -((2*(x-0.5))/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
            ddx_u = (((4*(x-0.5)**2)/sigma**2) - 2/sigma) * np.exp(-((x-0.5)**2)/sigma) * np.cos(t)
        
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
            #arr = -np.exp(-((x-0.5)**2)/0.004) * np.sin(t) + (np.exp(-((x-0.5)**2)/0.004) * np.cos(t))**2 * (-(2*(x-0.5))/0.004) - nu * np.cos(t) * (((2*(x-0.5))/0.004) * ((2*(x-0.5))/0.004) * np.exp(-((x-0.5)**2)/0.004) - 2/0.004 * np.exp(-((x-0.5)**2)/0.004))
            
        elif func == 'poly':
            arr = nu**2 * x**4 *(x**4 - 1) * np.sin(nu*t) + nu * x**4 * (1 - x**4) * np.cos(nu*t) * 4 * nu * x**3 * (1 - 2*x**4) * np.cos(nu*t) - nu**2 * 4 * x**2 * (3 - 14*x**4) * np.cos(nu*t)
            
        elif func == 'sin_heat':
            arr = -np.sin(np.pi * 4 * x) * (np.sin(t) - nu * (np.pi * 4) ** 2 * np.cos(t))
            
        return arr
    
    # Initialise solution vector and vector of right-hand side
    uf = np.zeros(Mf * nxf, dtype='cfloat')
        
    rhsFf = np.zeros(Mf * nxf, dtype='cfloat')
    
    for l in range(0, Mf):
        rhsFf[l*nxf:l*nxf + nxf] = fft(rhs(func, nu, xf, tf_int[l]))
    
    # Fine sweep
    if typeODE == 'heat':
        # heat equation
        uf_new = np.zeros(Mf * nxf, dtype='cfloat')
        fI = np.zeros(Mf * nxf, dtype='cfloat')
        
        for n in range(0, nF):
            # Compute integrals 'node to node'
            Skmf = np.zeros((Mf - 1) * nxf, dtype='cfloat')
            for l in range(0, Mf-1):
                for j in range(0, Mf):
                    Skmf[l*nxf:l*nxf+nxf] += dt * Sf[l, j] * (AIf.dot(ufIter[j*nxf:j*nxf+nxf]))
            
            uf_new[0:nxf] = uf_M
        
            fI[0:nxf] = AIf.dot(uf_M)
        
            for m in range(0, Mf-1):
                rhs = uf_new[m*nxf:m*nxf+nxf] + dtf[m] * (- AIf.dot(ufIter[(m+1)*nxf:(m+1)*nxf+nxf])) + Skmf[m*nxf:m*nxf+nxf]
            
                uf_new[(m+1)*nxf:(m+1)*nxf+nxf] = np.linalg.solve(np.identity(nxf) - dtf[m] * AIf, rhs) 
            
                # new implicit function values
                fI[(m+1)*nxf:(m+1)*nxf+nxf] = AIf.dot(uf_new[(m+1)*nxf:(m+1)*nxf+nxf])
                
            ufIter = uf_new
            
    elif typeODE == 'heat_forced':
        uf_new = np.zeros(Mf*nxf, dtype='cfloat')
        fE = np.zeros(Mf*nxf, dtype='cfloat')
        fI = np.zeros(Mf*nxf, dtype='cfloat')
        
        # nG coarse sweeps           
        for n in range(0, nF):
            # Compute integrals
            Skmf = np.zeros((Mf - 1) * nxf, dtype='cfloat')
            for m in range(0, Mf-1):
                for j in range(0, Mf):
                    Skmf[m*nxf:m*nxf+nxf] += dt * Sf[m, j] * (rhsFf[j*nxf:j*nxf + nxf] + AIf.dot(ufIter[j*nxf:j*nxf+nxf]))

            uf_new[0:nxf] = uf_M
            
            fE[0:nxf] = rhsFf[0:nxf]
            
            fI[0:nxf] = AIf.dot(uf_M)
                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mf-1):                   
                rhs = uf_new[m*nxf:m*nxf+nxf] + dtf[m] * (fE[m*nxf:m*nxf+nxf] - rhsFf[m*nxf:m*nxf + nxf] - AIf.dot(ufIter[(m+1)*nxf:(m+1)*nxf+nxf])) + Skmf[m*nxf:m*nxf+nxf]
            
                uf_new[(m+1)*nxf:(m+1)*nxf+nxf] = np.linalg.solve(np.identity(nxf) - dtf[m] * AIf, rhs)
            
                # new explicit function values
                fE[(m+1)*nxf:(m+1)*nxf+nxf] = rhsFf[(m+1)*nxf:(m+1)*nxf + nxf] 
            
                # new implicit function values
                fI[(m+1)*nxf:(m+1)*nxf+nxf] = AIf.dot(uf_new[(m+1)*nxf:(m+1)*nxf+nxf])
                
            ufIter = uf_new
        
    elif typeODE == 'Burgers':                                                                                          
        # IMEX Version (Burgers' equation)- uses the iterative method with S and SE
        
        # Initialisations
        AEfufhat = np.zeros(Mf * nxf, dtype='cfloat')
        nsf = np.zeros(Mf * nxf, dtype='cfloat')
        nsfhat = np.zeros(Mf * nxf, dtype='cfloat')
        uf_new = np.zeros(Mf * nxf, dtype='cfloat')
        fE = np.zeros(Mf * nxf, dtype='cfloat')
        fI = np.zeros(Mf * nxf, dtype='cfloat')
        newAEfuhat = np.zeros(nxf, dtype='cfloat')
        newnsf = np.zeros(nxf, dtype='cfloat')
            
        #uf_new = ufIter
                    
        for n in range(0, nF):
            # evaluation of the nonstiff term in spectral space
            for l in range(0, Mf):
                if l == 0:
                    AEfufhat[0:nxf] = AEf.dot(uf_M)
                    nsf[0:nxf] = -np.multiply(ifft(uf_M), ifft(AEfufhat[0:nxf]))
                    
                else:
                    AEfufhat[l*nxf:l*nxf + nxf] = AEf.dot(ufIter[l*nxf:l*nxf + nxf])
                    nsf[l*nxf:l*nxf + nxf] = -np.multiply(ifft(ufIter[l*nxf:l*nxf + nxf]), ifft(AEfufhat[l*nxf:l*nxf + nxf]))
                        
                nsfhat[l*nxf:l*nxf + nxf] = fft(nsf[l*nxf:l*nxf + nxf])
                
            # Compute integrals
            Skmf = np.zeros((Mf - 1) * nxf, dtype='cfloat')
            for m in range(0, Mf-1):
                for j in range(0, Mf):
                    Skmf[m*nxf:m*nxf+nxf] += dt * Sf[m, j] * (nsfhat[j*nxf:j*nxf + nxf] + rhsFf[j*nxf:j*nxf + nxf] + AIf.dot(ufIter[j*nxf:j*nxf+nxf]))

            uf_new[0:nxf] = uf_M
            
            fE[0:nxf] = nsfhat[0:nxf] 
            
            fI[0:nxf] = AIf.dot(uf_M)
                
                   
            # Solving Burgers' equation with right-hand side or without it
            for m in range(0, Mf-1):   
                rhs = uf_new[m*nxf:m*nxf+nxf] + dtf[m] * (fE[m*nxf:m*nxf+nxf] - nsfhat[m*nxf:m*nxf+nxf] - AIf.dot(ufIter[(m+1)*nxf:(m+1)*nxf+nxf])) + Skmf[m*nxf:m*nxf+nxf]
            
                uf_new[(m+1)*nxf:(m+1)*nxf+nxf] = np.linalg.solve(np.identity(nxf) - dtf[m] * AIf, rhs)
            
                # new explicit function values
                newAEfuhat = AEf.dot(uf_new[(m+1)*nxf:(m+1)*nxf+nxf])
                newnsf = -np.multiply(ifft(uf_new[(m+1)*nxf:(m+1)*nxf+nxf]), ifft(newAEfuhat))
                fE[(m+1)*nxf:(m+1)*nxf+nxf] = fft(newnsf) 
            
                # new implicit function values
                fI[(m+1)*nxf:(m+1)*nxf+nxf] = AIf.dot(uf_new[(m+1)*nxf:(m+1)*nxf+nxf])
                
            ufIter = uf_new
            
    elif typeODE == 'advdif':                                                           
        # IMEX Version (advection diffusion equation)- uses the iterative method with S and SE
        
        # Initialisations
        tmp_uf_new = np.zeros((Mf+1) * nxf, dtype='float')
        uf_new = np.zeros(Mf * nxf, dtype='float')
        fE = np.zeros(Mf * nxf, dtype='float')
        fI = np.zeros(Mf * nxf, dtype='float')
        rhs = np.zeros(nxf, dtype='float')
            
        for n in range(0, nF):
            
            for m in range(0, Mf):
                fE[m*nxf:m*nxf+nxf] = ifft(AEf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(ufIter[m*nxf:m*nxf+nxf])))
                
            # Compute integrals
            #Skmf = np.zeros((Mf - 1) * nxf, dtype='cfloat')
            #for l in range(0, Mf-1):
            #    for j in range(0, Mf):
            #        Skmf[l*nxf:l*nxf+nxf] += dt * Sf[l, j] * (ifft(AEf.dot(fft(ufIter[j*nxf:j*nxf+nxf]))) + ifft(AIf.dot(fft(ufIter[j*nxf:j*nxf+nxf]))))
            
            Qkmf = np.zeros(Mf * nxf, dtype='float')
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * (fE[j*nxf:j*nxf+nxf]+ fI[j*nxf:j*nxf+nxf])
                    
            for l in range(0, Mf):
                for j in range(0, Mf):
                    Qkmf[l*nxf:l*nxf+nxf] -= dt * (QEf[l, j] * fE[j*nxf:j*nxf+nxf] + QIf[l, j] * fI[j*nxf:j*nxf+nxf])
                    
                Qkmf[l*nxf:l*nxf+nxf] += uf_M
                    
                                                                            
            #uf_new[0:nxf] = uf_M
            
            #fE[0:nxf] = ifft(AEf.dot(uf_M))     
            
            #fI[0:nxf] = ifft(AIf.dot(uf_M))
            
            # sweep
            #for m in range(0, Mf-1):
            #    rhs = uf_new[m*nxf:m*nxf+nxf] + dtf[m] * (fE[m*nxf:m*nxf+nxf] - ifft(AEf.dot(fft(ufIter[m*nxf:m*nxf+nxf]))) - ifft(AIf.dot(fft(ufIter[(m+1)*nxf:(m+1)*nxf+nxf])))) + Skmf[m*nxf:m*nxf+nxf]
            
                #tmp = np.linalg.solve(np.identity(nxf) - dtf[m] * AIf, fft(rhs))
            #    tmp = fft(rhs) * (1.0 / (1.0 - dtf[m] * np.diag(AIf)))
            #    uf_new[(m+1)*nxf:(m+1)*nxf+nxf] = ifft(tmp)
            
                # new explicit function values
            #    fE[(m+1)*nxf:(m+1)*nxf+nxf] = ifft(AEf.dot(fft(uf_new[(m+1)*nxf:(m+1)*nxf+nxf])))
            
                # new implicit function values
            #    fI[(m+1)*nxf:(m+1)*nxf+nxf] = ifft(AIf.dot(fft(uf_new[(m+1)*nxf:(m+1)*nxf+nxf])))

                
            for m in range(0, Mf):    
                rhs = Qkmf[m*nxf:m*nxf+nxf]
                
                for j in range(0, m):
                    rhs += dt * (QEf[m, j] * fE[j*nxf:j*nxf+nxf] + QIf[m, j] * fI[j*nxf:j*nxf+nxf])
                    
                tmp = np.linalg.solve(np.identity(nxf) - dt * QIf[m, m] * AIf, fft(rhs))
                uf_new[m*nxf:m*nxf+nxf] = ifft(tmp)
                
                # new explicit function values
                fE[m*nxf:m*nxf+nxf] = ifft(AEf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
            
                # new implicit function values
                fI[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf_new[m*nxf:m*nxf+nxf])))
            
            ufIter = uf_new
         
    return uf_new