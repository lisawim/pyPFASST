import numpy as np
from numpy.fft import fft, ifft
from numpy import kron
from transfer_operators_test import restriction

def FAS(AIc, AIf, AEc, AEf, dt, func, Mc, nxc, Mf, nxf, nu, Qf, Qc, Sf, Sc, tc_int, tf_int, typeODE, uchat, ufhat, xc, xf):
    
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='cfloat')
        sigma = 0.004
        
        if func == 'exp':
            arr = -np.exp(-((x-0.5)**2)/0.004) * np.sin(t) + (np.exp(-((x-0.5)**2)/0.004) * np.cos(t))**2 * (-(2*(x-0.5))/0.004) - nu * np.cos(t) * (((2*(x-0.5))/0.004) * ((2*(x-0.5))/0.004) * np.exp(-((x-0.5)**2)/0.004) - 2/0.004 * np.exp(-((x-0.5)**2)/0.004))
            
        elif func == 'poly':
            arr = nu**2 * x**4 *(x**4 - 1) * np.sin(nu*t) + nu * x**4 * (1 - x**4) * np.cos(nu*t) * 4 * nu * x**3 * (1 - 2*x**4) * np.cos(nu*t) - nu**2 * 4 * x**2 * (3 - 14*x**4) * np.cos(nu*t)
            
        elif func == 'sin_heat':
            arr = -np.sin(np.pi * 4 * x) * (np.sin(t) - nu * (np.pi * 4) ** 2 * np.cos(t))
            
        return arr
    
    def get_transfer_matrix_Q(tc, tf):
        num_tc = np.shape(tc)[0]
        num_tf = np.shape(tf)[0]
        
        tmat = np.zeros((num_tc, num_tf))
        
        for i in range(num_tc):
            
            xi = tc[i]
            
            for j in range(num_tf):
                den = 1.0
                num = 1.0
                
                for k in range(num_tf):
                    if k == j:
                        continue
                        
                    else:
                        den *= tf[j] - tf[k]
                        num *= xi - tf[k]
                        
                tmat[i, j] = num/den
                
        return tmat
    
    Rcoll = get_transfer_matrix_Q(tc_int, tf_int)
    
    restr_SF = np.zeros((Mc-1) * nxc, dtype='cfloat')
    
    rhsFc = np.zeros(nxc * Mc, dtype='cfloat')
    rhsFf = np.zeros(nxf * Mf, dtype='cfloat')
    
    for m in range(0, Mc):
        rhsFc[m*nxc:m*nxc + nxc] = fft(rhs(func, nu, xc, tc_int[m]))
        
    for m in range(0, Mf):
        rhsFf[m*nxf:m*nxf + nxf] = fft(rhs(func, nu, xf, tf_int[m]))
    
    # tau for different equations (heat equation or Burgers' equation)
    if typeODE == 'heat':
        tau = np.zeros(nxc * (Mc-1), dtype='cfloat')
        
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='cfloat')
        
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf + nxf] = AIf.dot(ufhat[m*nxf:m*nxf + nxf])
                
        # Integrate from 't0 to node' on fine level
        restr_QF = restriction(kron(Qf, np.identity(nxf)).dot(fevalf), Mc, nxc, Mf, nxf)
                                       
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='cfloat')
            
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = AIc.dot(uchat[m*nxc:m*nxc + nxc])
        
        # Integrate from 't0 to node' on coarse level
        QFc = kron(Qc, np.identity(nxc)).dot(fevalc)
        
        # Conversion to 'node to node'
        for m in range(0, Mc-1):
            restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
            QFc[m*nxc:m*nxc+nxc] = QFc[(m+1)*nxc:(m+1)*nxc+nxc] - QFc[m*nxc:m*nxc+nxc]        
        
        tau = dt * (restr_QF - QFc)
        
    elif typeODE == 'heat_forced':
        tau = np.zeros(nxc * (Mc-1), dtype='cfloat')
        
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='cfloat')
        
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf + nxf] = AIf.dot(ufhat[m*nxf:m*nxf + nxf]) + rhsFf[m*nxf:m*nxf+nxf]
                
        # Integrate from 't0 to node' on fine level
        restr_QF = restriction(kron(Qf, np.identity(nxf)).dot(fevalf), Mc, nxc, Mf, nxf)
                                       
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='cfloat')
            
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = AIc.dot(uchat[m*nxc:m*nxc + nxc]) + rhsFc[m*nxc:m*nxc+nxc]
        
        # Integrate from 't0 to node' on coarse level
        QFc = kron(Qc, np.identity(nxc)).dot(fevalc)
        
        # Conversion to 'node to node'
        for m in range(0, Mc-1):
            restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
            QFc[m*nxc:m*nxc+nxc] = QFc[(m+1)*nxc:(m+1)*nxc+nxc] - QFc[m*nxc:m*nxc+nxc]        
        
        tau = dt * (restr_QF - QFc)
        
    elif typeODE == 'Burgers':
        tau = np.zeros(nxc * (Mc-1), dtype='cfloat')
        
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='cfloat')
        
        AEfufhat = np.zeros(nxf * Mf, dtype='cfloat')
        nsf = np.zeros(nxf * Mf, dtype='cfloat')
        nsfhat = np.zeros(nxf * Mf, dtype='cfloat')
        
        for m in range(0, Mf):
            AEfufhat[m*nxf:m*nxf + nxf] = AEf.dot(ufhat[m*nxf:m*nxf + nxf])
            nsf[m*nxf:m*nxf + nxf] = -np.multiply(ifft(ufhat[m*nxf:m*nxf + nxf]), ifft(AEfufhat[m*nxf:m*nxf + nxf]))
            nsfhat[m*nxf:m*nxf + nxf] = fft(nsf[m*nxf:m*nxf + nxf])
            
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf + nxf] = nsfhat[m*nxf:m*nxf + nxf] + AIf.dot(ufhat[m*nxf:m*nxf + nxf]) + rhsFf[m*nxf:m*nxf+nxf]
                
        # Integrate from 't0 to node' on fine level
        restr_QF = restriction(kron(Qf, np.identity(nxf)).dot(fevalf), Mc, nxc, Mf, nxf)
        
        # Integrate from 'node to node' on fine level
        SF = kron(Sf, np.identity(nxf)).dot(fevalf)
        restr_SF[0:nxc] = restriction(SF[0:nxf] + SF[nxf:2*nxf], 1, nxc, 1, nxf)
        restr_SF[nxc:2*nxc] = restriction(SF[2*nxf:3*nxf] + SF[3*nxf:4*nxf], 1, nxc, 1, nxf)
                                       
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='cfloat')
        
        AEcuchat = np.zeros(nxc * Mc, dtype='cfloat')
        nsc = np.zeros(nxc * Mc, dtype='cfloat')
        nschat = np.zeros(nxc * Mc, dtype='cfloat')
        
        for m in range(0, Mc):
            AEcuchat[m*nxc:m*nxc + nxc] = AEc.dot(uchat[m*nxc:m*nxc + nxc])
            nsc[m*nxc:m*nxc + nxc] = -np.multiply(ifft(uchat[m*nxc:m*nxc + nxc]), ifft(AEcuchat[m*nxc:m*nxc + nxc]))
            nschat[m*nxc:m*nxc + nxc] = fft(nsc[m*nxc:m*nxc + nxc])
            
        for m in range(0, Mc):
            fevalc[m*nxc:m*nxc + nxc] = nschat[m*nxc:m*nxc + nxc] + AIc.dot(uchat[m*nxc:m*nxc + nxc]) + rhsFc[m*nxc:m*nxc+nxc]
        
        # Integrate from 't0 to node' on coarse level
        QFc = kron(Qc, np.identity(nxc)).dot(fevalc)
        
        # Integrate from 'node to node' on coarse level
        SFc = kron(Sc, np.identity(nxc)).dot(fevalc)
        
        # Conversion to 'node to node'
        for m in range(0, Mc-1):
            restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
            QFc[m*nxc:m*nxc+nxc] = QFc[(m+1)*nxc:(m+1)*nxc+nxc] - QFc[m*nxc:m*nxc+nxc]        
        
        tau = dt * (restr_QF - QFc)
        
    elif typeODE == 'advdif':
        tau = np.zeros(nxc * (Mc-1), dtype='cfloat')
        
        #for m in range(0, Mf):
        #    tmp = ufhat[m*nxf:m*nxf+nxf]
        #    print(ifft(tmp)[:10])
        #    print()
        
        # restrict fine u in space
        tmp_u = np.zeros(Mf*nxc, dtype='cfloat')
        for m in range(0, Mf):
            tmp_u[m*nxc:m*nxc+nxc] = restriction(ufhat[m*nxf:m*nxf+nxf], 1, nxc, 1, nxf)
            
        # restrict this u in time on collocation nodes
        tmp2_u = np.zeros(Mc*nxc, dtype='cfloat')
        
        #tmp2_u[0:nxc] = np.fft.ifft(ufhat[0:nxf])[::2]
        #tmp2_u[nxc:2*nxc] = np.fft.ifft(ufhat[2*nxf:3*nxf])[::2]
        #tmp2_u[2*nxc:3*nxc] = np.fft.ifft(ufhat[4*nxf:5*nxf])[::2]
        for n in range(0, Mc):
            for m in range(0, Mf):
                tmp2_u[n*nxc:n*nxc+nxc] += Rcoll[n, m] * tmp_u[m*nxc:m*nxc+nxc]
        
        # Evaluation of function values on coarse level                          
        fevalc = np.zeros(nxc * Mc, dtype='cfloat')
        for m in range(0, Mc):
            #fevalc[m*nxc:m*nxc + nxc] = AEc.dot(uchat[m*nxc:m*nxc + nxc]) + AIc.dot(uchat[m*nxc:m*nxc + nxc])
            fevalc[m*nxc:m*nxc + nxc] = AEc.dot(tmp2_u[m*nxc:m*nxc + nxc]) + AIc.dot(tmp2_u[m*nxc:m*nxc + nxc])
            #fevalc[m*nxc:m*nxc + nxc] = np.fft.ifft(fevalc[m*nxc:m*nxc + nxc])
                
                
        # Evaluation of function values on fine level
        fevalf = np.zeros(nxf * Mf, dtype='cfloat')            
        for m in range(0, Mf):
            fevalf[m*nxf:m*nxf + nxf] = AEf.dot(ufhat[m*nxf:m*nxf + nxf]) + AIf.dot(ufhat[m*nxf:m*nxf + nxf])
        
        # Integrate from 't0 to node' on fine level
        Qf_int = np.zeros(Mf * nxf, dtype='cfloat')
        for l in range(0, Mf):
            for j in range(0, Mf):
                Qf_int[l*nxf:l*nxf+nxf] += dt * Qf[l, j] * fevalf[j*nxf:j*nxf+nxf]
                
        #for m in range(0, Mf):
        #    print("m=", m)
        #    tmp = Qf_int[m*nxf:m*nxf+nxf]
        #    print(ifft(tmp)[:10])
        #    print()
                    
        # Integrate from 't0 to node' on fine level and restrict
        restr_QF = restriction(Qf_int, Mc, nxc, Mf, nxf)
        
        # restriction not in spectral space via injection
        tmp_restr_QF_ifft = np.zeros(Mc*nxc, dtype='float')
                           
        #tmp_restr_QF_ifft[0:nxc] = (Qf_int[0:nxf])[::2]
        #tmp_restr_QF_ifft[nxc:2*nxc] = (Qf_int[2*nxf:3*nxf])[::2]
        #tmp_restr_QF_ifft[2*nxc:3*nxc] = (Qf_int[4*nxf:5*nxf])[::2]
        
        tmp_restr_QF = np.zeros(Mc * nxc, dtype='float')
        for m in range(0, Mc):
            tmp_restr_QF[m*nxc:m*nxc+nxc] = np.fft.ifft(restr_QF[m*nxc:m*nxc+nxc])
            #print(restr_QF[m*nxc:m*nxc+nxc][125:135])
            
        Qc_int = np.zeros(Mc * nxc, dtype='cfloat')
        for l in range(0, Mc):
            for j in range(0, Mc):
                Qc_int[l*nxc:l*nxc+nxc] += dt * Qc[l, j] * fevalc[j*nxc:j*nxc+nxc]
                    
        # Integrate from 't0 to node' on coarse level
        QFc = kron(Qc, np.identity(nxc)).dot(fevalc)
        
        tmp_Qc_int = np.zeros(Mc * nxc, dtype='float')
        tmp_tau = np.zeros(Mc * nxc, dtype='float')
        for m in range(0, Mc):
            tmp_Qc_int[m*nxc:m*nxc+nxc] = np.fft.ifft(Qc_int[m*nxc:m*nxc+nxc])
            #tmp_Qc_int[m*nxc:m*nxc+nxc] = Qc_int[m*nxc:m*nxc+nxc]
            #tauFG = tmp_restr_QF[m*nxc:m*nxc+nxc]
            #tauG = tmp_Qc_int[m*nxc:m*nxc+nxc]
            #Gtau = tauFG - tauG
            #print(np.fft.fft(Gtau)[:10])
            #print(tmp_Qc_int[m*nxc:m*nxc+nxc][125:135])
            tmp_tau[m*nxc:m*nxc+nxc] = tmp_restr_QF[m*nxc:m*nxc+nxc] - tmp_Qc_int[m*nxc:m*nxc+nxc]
        
        #tmp_tau = (restr_QF - QFc)
        #tmp_tau = (restr_QF - Qc_int)
        #tmp_tau = tmp_restr_QF_ifft - tmp_Qc_int
        tmp_tau = (tmp_restr_QF - tmp_Qc_int)
        
        np.set_printoptions(precision=20)
        for m in range(0, Mc):
            #print()
            print(tmp_tau[m*nxc:m*nxc+nxc][:10])
            #print("restr_QF=", tmp_restr_QF[m*nxc:m*nxc+nxc][:10])
            #print("Qc_int=", tmp_Qc_int[m*nxc:m*nxc+nxc][:10])
            #print()
            
        #for m in range(0, Mc):
            #tmp = Qc_int[m*nxc:m*nxc+nxc]
            #tmp = tmp2_u[m*nxc:m*nxc+nxc]
            #print("Qc_int=", ifft(tmp)[:10])
            #print()
        
        # Conversion to 'node to node'
        for m in range(0, Mc-1):
            restr_QF[m*nxc:m*nxc+nxc] = restr_QF[(m+1)*nxc:(m+1)*nxc+nxc] - restr_QF[m*nxc:m*nxc+nxc]
            #QFc[m*nxc:m*nxc+nxc] = QFc[(m+1)*nxc:(m+1)*nxc+nxc] - QFc[m*nxc:m*nxc+nxc]
            Qc_int[m*nxc:m*nxc+nxc] = Qc_int[(m+1)*nxc:(m+1)*nxc+nxc] - Qc_int[m*nxc:m*nxc+nxc]
        
        #tau = dt * (restr_QF - QFc)
        tau = restr_QF - Qc_int

        
    return tau
    