import numpy as np
from numpy.fft import fft, ifft
from numpy import kron
from transfer_operators import restriction
import warnings
warnings.filterwarnings('ignore')

def FAS(AIc, AIf, AEc, AEf, dt, func, Mc, nxc, Mf, nxf, nu, Qf, Qc, Sf, Sc, tc_int, tf_int, typeODE, uc, uf, xc, xf):
    
    """
    Input:
        AIc          -    Operator for Laplacian on coarse level
        AIf          -    Operator for gradient on coarse level
        AEc          -    Operator for Laplacian on fine level
        AEf          -    Operator for gradient on fine level
        dt           -    time step
        func         -    function for initial condition
        Mc           -    number of coarse collocation nodes
        nxc          -    number of coarse DoF
        Mf           -    number of fine collocation nodes
        nxf          -    number of fine DoF
        nu           -    diffusion coefficient
        Qf           -    fine integration matrix 't0 to node'
        Qc           -    coarse integration matrix 't0 to node'
        Sf           -    fine spectral integration matrix 'node to node'
        Sc           -    coarse coarse integration matrix 'node to node'
        tc_int       -    time interval on coarse level
        tf_int       -    time interval on fine level
        typeODE      -    equation to be solved
        uc           -    current vector of u on coarse level
        uf           -    current vector of u on fine level
        xc           -    x-values on coarse level
        xf           -    x-values on fine level
        
    Return:
        FAS correction
    """
    
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='float')
        sigma = 0.004
        
        if func == 'exp':
            arr = -np.exp(-((x-0.5)**2)/0.004) * np.sin(t) + (np.exp(-((x-0.5)**2)/0.004) * np.cos(t))**2 * (-(2*(x-0.5))/0.004) - nu * np.cos(t) * (((2*(x-0.5))/0.004) * ((2*(x-0.5))/0.004) * np.exp(-((x-0.5)**2)/0.004) - 2/0.004 * np.exp(-((x-0.5)**2)/0.004))
            
        elif func == 'poly':
            #arr = nu**2 * x**4 *(x**4 - 1) * np.sin(nu*t) + nu * x**4 * (1 - x**4) * np.cos(nu*t) * 4 * nu * x**3 * (1 - 2*x**4) * np.cos(nu*t) - nu**2 * 4 * x**2 * (3 - 14*x**4) * np.cos(nu*t)
            
            u = x**4 * (1-x)**4 * np.cos(t)
            dt_u = -x**4 * (1-x)**4 * np.sin(t)
            dx_u = -4 * (1-x)**3 * x**3 * (2*x-1) * np.cos(t)
            ddx_u = 4 * (1-x)**2 * x**2 * (14*x**2 - 14*x + 3) * np.cos(t)
            
            arr = (dt_u) + u * (dx_u) - nu * (ddx_u)
            
        elif func == 'sin_heat':
            arr = -np.sin(np.pi * 4 * x) * (np.sin(t) - nu * (np.pi * 4) ** 2 * np.cos(t))
            
        return arr

    
    restr_SF = np.zeros((Mc-1) * nxc, dtype='float')
    
    fexplc = np.zeros(Mc*nxc, dtype='float')
    fimplc = np.zeros(Mc*nxc, dtype='float')
    
    fexplf = np.zeros(Mf*nxf, dtype='float')
    fimplf = np.zeros(Mf*nxf, dtype='float')
    
    
    rhsFc = np.zeros(nxc * Mc, dtype='float')
    rhsFf = np.zeros(nxf * Mf, dtype='float')
    
    for m in range(0, Mc):
        rhsFc[m*nxc:m*nxc + nxc] = rhs(func, nu, xc, tc_int[m])
        
    for m in range(0, Mf):
        rhsFf[m*nxf:m*nxf + nxf] = rhs(func, nu, xf, tf_int[m])
    
    # tau for different equations (heat equation or Burgers' equation)
    if typeODE == 'heat':
        # 'node to node'
        #tau = np.zeros(nxc * (Mc-1), dtype='float')
        
        # 't0 to node'
        tau = np.zeros(nxc * Mc, dtype='float')
        
        
        # restrict fine u in space and time
        tmp_u = np.zeros(Mc*nxc, dtype='float')        
        tmp_u = restriction(uf, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
        
        # Compute coarse function values
        for m in range(0, Mc):
            fimplc[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
            
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='float')
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = fimplc[m*nxc:m*nxc+nxc]
        
        
        # Integrate from 't0 to node' on coarse level
        Qc_int = np.zeros(Mc * nxc, dtype='float')
        for l in range(0, Mc):
            for j in range(0, Mc):
                Qc_int[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fevalc[j*nxc:j*nxc+nxc]
                
        
        # Compute fine function values
        for m in range(0, Mf):
            fimplf[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf[m*nxf:m*nxf+nxf])))
            
            
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='float')        
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf+nxf] = fimplf[m*nxf:m*nxf+nxf]
            
        
        # Integrate from 't0 to node' on fine level
        Qf_int = np.zeros(Mf * nxf, dtype='float')
        for l in range(0, Mf):
            for j in range(0, Mf):
                Qf_int[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fevalf[j*nxf:j*nxf+nxf]
                
                                
        # Restrict the integral of fine function values in space and time
        restr_QF = restriction(Qf_int, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
                
        # tau with 't0 to node'
        tau = (restr_QF - Qc_int)
        
        
        # Conversion to 'node to node'
        #for m in range(0, Mc-1):
        #    restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
        #    QFc[m*nxc:m*nxc+nxc] = QFc[(m+1)*nxc:(m+1)*nxc+nxc] - QFc[m*nxc:m*nxc+nxc] 
        
        
        #tau = dt * (restr_QF - QFc)                              
        
    elif typeODE == 'heat_forced':
        # 'node to node'
        #tau = np.zeros(nxc * (Mc-1), dtype='float')
        
        # 't0 to node'
        tau = np.zeros(nxc * Mc, dtype='float')
        
        
        # restrict fine u in space and time
        tmp_u = np.zeros(Mc*nxc, dtype='float')        
        tmp_u = restriction(uf, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
        
        # Compute coarse function values
        for m in range(0, Mc):
            #fexplc[m*nxc:m*nxc+nxc] = ifft(AEc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
            fexplc[m*nxc:m*nxc+nxc] = rhsFc[m*nxc:m*nxc+nxc]
            fimplc[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
            
            
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='float')
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = fexplc[m*nxc:m*nxc+nxc] + fimplc[m*nxc:m*nxc+nxc]
            
            
        # Integrate from 't0 to node' on coarse level
        Qc_int = np.zeros(Mc * nxc, dtype='float')
        for l in range(0, Mc):
            for j in range(0, Mc):
                Qc_int[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fevalc[j*nxc:j*nxc+nxc]
                
                
        # Compute fine function values
        for m in range(0, Mf):
            #fexplf[m*nxf:m*nxf+nxf] = ifft(AEf.dot(fft(uf[m*nxf:m*nxf+nxf])))
            fexplf[m*nxf:m*nxf+nxf] = rhsFf[m*nxf:m*nxf+nxf]
            fimplf[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf[m*nxf:m*nxf+nxf])))
            
        
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='float')        
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf+nxf] = fexplf[m*nxf:m*nxf+nxf] + fimplf[m*nxf:m*nxf+nxf]
            
            
        # Integrate from 't0 to node' on fine level
        Qf_int = np.zeros(Mf * nxf, dtype='float')
        for l in range(0, Mf):
            for j in range(0, Mf):
                Qf_int[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fevalf[j*nxf:j*nxf+nxf]
                
                
        # Restrict the integral of fine function values in space and time
        restr_QF = restriction(Qf_int, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
        
        # tau with 't0 to node'
        tau = (restr_QF - Qc_int)
        
        
        # Conversion to 'node to node'
        #for m in range(0, Mc-1):
        #    restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
        #    QFc[m*nxc:m*nxc+nxc] = QFc[(m+1)*nxc:(m+1)*nxc+nxc] - QFc[m*nxc:m*nxc+nxc]   
        
        
        #tau = dt * (restr_QF - QFc)
        
    elif typeODE == 'Burgers':

        AEcuchat = np.zeros(nxc * Mc, dtype='cfloat')
        nsc = np.zeros(nxc * Mc, dtype='cfloat')
        nschat = np.zeros(nxc * Mc, dtype='float')
        AEfufhat = np.zeros(nxf * Mf, dtype='cfloat')
        nsf = np.zeros(nxf * Mf, dtype='cfloat')
        nsfhat = np.zeros(nxf * Mf, dtype='float')
        
        
        # 'node to node'
        #tau = np.zeros(nxc * (Mc-1), dtype='float')
        
        # 't0 to node'
        tau = np.zeros(nxc * Mc, dtype='float')
        
        
        # restrict fine u in space and time
        tmp_u = np.zeros(Mc*nxc, dtype='float')        
        tmp_u = restriction(uf, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
                
        for m in range(0, Mc):
            AEcuchat[m*nxc:m*nxc + nxc] = AEc.dot(fft(tmp_u[m*nxc:m*nxc + nxc]))
            nsc[m*nxc:m*nxc + nxc] = -np.multiply(tmp_u[m*nxc:m*nxc + nxc], ifft(AEcuchat[m*nxc:m*nxc + nxc]))
            nschat[m*nxc:m*nxc + nxc] = ifft(fft(nsc[m*nxc:m*nxc + nxc]))
            
        
        # Compute coarse function values
        for m in range(0, Mc):
            fexplc[m*nxc:m*nxc+nxc] = nschat[m*nxc:m*nxc + nxc] + rhsFc[m*nxc:m*nxc+nxc]
            fimplc[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
            
                
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='float')
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = fexplc[m*nxc:m*nxc+nxc] + fimplc[m*nxc:m*nxc+nxc]
            
            
        # Integrate from 't0 to node' on coarse level
        Qc_int = np.zeros(Mc * nxc, dtype='float')
        for l in range(0, Mc):
            for j in range(0, Mc):
                Qc_int[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fevalc[j*nxc:j*nxc+nxc]
                
                
        for m in range(0, Mf):
            AEfufhat[m*nxf:m*nxf + nxf] = AEf.dot(fft(uf[m*nxf:m*nxf + nxf]))
            nsf[m*nxf:m*nxf + nxf] = -np.multiply(uf[m*nxf:m*nxf + nxf], ifft(AEfufhat[m*nxf:m*nxf + nxf]))
            nsfhat[m*nxf:m*nxf + nxf] = ifft(fft(nsf[m*nxf:m*nxf + nxf]))
            
        
        # Compute fine function values
        for m in range(0, Mf):
            fexplf[m*nxf:m*nxf+nxf] = nsfhat[m*nxf:m*nxf + nxf] + rhsFf[m*nxf:m*nxf+nxf]
            fimplf[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf[m*nxf:m*nxf+nxf])))
            
                
        # Evaluation of function values on fine level                          
        fevalf = np.zeros(nxf * Mf, dtype='float')
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf + nxf] = fexplf[m*nxf:m*nxf+nxf] + fimplf[m*nxf:m*nxf+nxf]
            
            
        # Integrate from 't0 to node' on coarse level
        Qf_int = np.zeros(Mf * nxf, dtype='float')
        for l in range(0, Mf):
            for j in range(0, Mf):
                Qf_int[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fevalf[j*nxf:j*nxf+nxf]
                
                
        # Restrict the integral of fine function values in space and time
        restr_QF = restriction(Qf_int, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
        
        # tau with 't0 to node'
        tau = (restr_QF - Qc_int)
        
        
        # Conversion to 'node to node'
        #for m in range(0, Mc-1):
        #    restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
        #    Qc_int[m*nxc:m*nxc+nxc] = Qc_int[(m+1)*nxc:(m+1)*nxc+nxc] - Qc_int[m*nxc:m*nxc+nxc]
        
        
        # tau with 'node to node'
        #tau = restr_QF - Qc_int
        
    elif typeODE == 'Burgers_poly':

        AEcuchat = np.zeros(nxc * Mc, dtype='cfloat')
        nsc = np.zeros(nxc * Mc, dtype='cfloat')
        nschat = np.zeros(nxc * Mc, dtype='float')
        AEfufhat = np.zeros(nxf * Mf, dtype='cfloat')
        nsf = np.zeros(nxf * Mf, dtype='cfloat')
        nsfhat = np.zeros(nxf * Mf, dtype='float')
        
        
        # 'node to node'
        #tau = np.zeros(nxc * (Mc-1), dtype='float')
        
        # 't0 to node'
        tau = np.zeros(nxc * Mc, dtype='float')
        
        
        # restrict fine u in space and time
        tmp_u = np.zeros(Mc*nxc, dtype='float')        
        tmp_u = restriction(uf, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
                
        for m in range(0, Mc):
            AEcuchat[m*nxc:m*nxc + nxc] = AEc.dot(fft(tmp_u[m*nxc:m*nxc + nxc]))
            nsc[m*nxc:m*nxc + nxc] = -np.multiply(tmp_u[m*nxc:m*nxc + nxc], ifft(AEcuchat[m*nxc:m*nxc + nxc]))
            nschat[m*nxc:m*nxc + nxc] = ifft(fft(nsc[m*nxc:m*nxc + nxc]))
            
        
        # Compute coarse function values
        for m in range(0, Mc):
            fexplc[m*nxc:m*nxc+nxc] = nschat[m*nxc:m*nxc + nxc] + rhsFc[m*nxc:m*nxc+nxc]
            fimplc[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
            
                
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='float')
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = fexplc[m*nxc:m*nxc+nxc] + fimplc[m*nxc:m*nxc+nxc]
            
            
        # Integrate from 't0 to node' on coarse level
        Qc_int = np.zeros(Mc * nxc, dtype='float')
        for l in range(0, Mc):
            for j in range(0, Mc):
                Qc_int[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fevalc[j*nxc:j*nxc+nxc]
                
                
        for m in range(0, Mf):
            AEfufhat[m*nxf:m*nxf + nxf] = AEf.dot(fft(uf[m*nxf:m*nxf + nxf]))
            nsf[m*nxf:m*nxf + nxf] = -np.multiply(uf[m*nxf:m*nxf + nxf], ifft(AEfufhat[m*nxf:m*nxf + nxf]))
            nsfhat[m*nxf:m*nxf + nxf] = ifft(fft(nsf[m*nxf:m*nxf + nxf]))
            
        
        # Compute fine function values
        for m in range(0, Mf):
            fexplf[m*nxf:m*nxf+nxf] = nsfhat[m*nxf:m*nxf + nxf] + rhsFf[m*nxf:m*nxf+nxf]
            fimplf[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf[m*nxf:m*nxf+nxf])))
            
                
        # Evaluation of function values on fine level                          
        fevalf = np.zeros(nxf * Mf, dtype='float')
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf + nxf] = fexplf[m*nxf:m*nxf+nxf] + fimplf[m*nxf:m*nxf+nxf]
            
            
        # Integrate from 't0 to node' on coarse level
        Qf_int = np.zeros(Mf * nxf, dtype='float')
        for l in range(0, Mf):
            for j in range(0, Mf):
                Qf_int[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fevalf[j*nxf:j*nxf+nxf]
                
                
        # Restrict the integral of fine function values in space and time
        restr_QF = restriction(Qf_int, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
        
        # tau with 't0 to node'
        tau = (restr_QF - Qc_int)
        
        
        # Conversion to 'node to node'
        #for m in range(0, Mc-1):
        #    restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
        #    Qc_int[m*nxc:m*nxc+nxc] = Qc_int[(m+1)*nxc:(m+1)*nxc+nxc] - Qc_int[m*nxc:m*nxc+nxc]
        
        
        # tau with 'node to node'
        #tau = restr_QF - Qc_int
        
    elif typeODE == 'advdif':
        # 'node to node'
        #tau = np.zeros(nxc * (Mc-1), dtype='float')
        
        # 't0 to node'
        tau = np.zeros(nxc * Mc, dtype='float')
        
        
        # restrict fine u in space and time
        tmp_u = np.zeros(Mc*nxc, dtype='float')        
        tmp_u = restriction(uf, Mc, nxc, Mf, nxf, tc_int, tf_int)
        
            
        # Compute coarse function values
        for m in range(0, Mc):
            fexplc[m*nxc:m*nxc+nxc] = ifft(AEc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
            fimplc[m*nxc:m*nxc+nxc] = ifft(AIc.dot(fft(tmp_u[m*nxc:m*nxc+nxc])))
        
        
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='float')
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = fexplc[m*nxc:m*nxc+nxc] + fimplc[m*nxc:m*nxc+nxc]
        
        
        # Integrate from 't0 to node' on coarse level
        Qc_int = np.zeros(Mc * nxc, dtype='float')
        for l in range(0, Mc):
            for j in range(0, Mc):
                Qc_int[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fevalc[j*nxc:j*nxc+nxc]
                
                
        # Compute fine function values
        for m in range(0, Mf):
            fexplf[m*nxf:m*nxf+nxf] = ifft(AEf.dot(fft(uf[m*nxf:m*nxf+nxf])))
            fimplf[m*nxf:m*nxf+nxf] = ifft(AIf.dot(fft(uf[m*nxf:m*nxf+nxf])))
                
                
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='float')            
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf+nxf] = fexplf[m*nxf:m*nxf+nxf] + fimplf[m*nxf:m*nxf+nxf]
        
        
        # Integrate from 't0 to node' on fine level
        Qf_int = np.zeros(Mf * nxf, dtype='float')
        for l in range(0, Mf):
            for j in range(0, Mf):
                Qf_int[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fevalf[j*nxf:j*nxf+nxf]
        
        
        # Restrict the integral of fine function values in space and time
        restr_QF = restriction(Qf_int, Mc, nxc, Mf, nxf, tc_int, tf_int)

        
        # tau with 't0 to node'
        tau = (restr_QF - Qc_int)
        
        
        # Conversion to 'node to node'
        #for m in range(0, Mc-1):
        #    restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
        #    Qc_int[m*nxc:m*nxc+nxc] = Qc_int[(m+1)*nxc:(m+1)*nxc+nxc] - Qc_int[m*nxc:m*nxc+nxc]
        
        
        # tau with 'node to node'
        #tau = restr_QF - Qc_int

        
    return tau
    
