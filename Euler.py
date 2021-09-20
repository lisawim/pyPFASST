import numpy as np
from differential_operators_test import differentialA

def expEuler(dt, func, L, nu, u0hat, t0, T, typeODE, v, x):
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='cfloat')
        sigma = 0.004
        
        if func == "exp":
            arr = -np.exp(-((x-0.5)**2)/0.004) * np.sin(t) + (np.exp(-((x-0.5)**2)/0.004) * np.cos(t))**2 * (-(2*(x-0.5))/0.004) - nu * np.cos(t) * (((2*(x-0.5))/0.004) * ((2*(x-0.5))/0.004) * np.exp(-((x-0.5)**2)/0.004) - 2/0.004 * np.exp(-((x-0.5)**2)/0.004))
            
        elif func == "poly":
            f = lambda x, t: nu**2 * x**4 *(x**4 - 1) * np.sin(nu*t) + nu * x**4 * (1 - x**4) * np.cos(nu*t) * 4 * nu * x**3 * (1 - 2*x**4) * np.cos(nu*t) - nu**2 * 4 * x**2 * (3 - 14*x**4) * np.cos(nu*t)
            arr = f(x, t)
            
        return arr
    
    uhat = u0hat
    nx = np.shape(u0hat)[0]
    AE, AI = differentialA(L, nu, nx, typeODE, v)
    
    AEcuchat = np.zeros(nx, dtype='cfloat')
    nsc = np.zeros(nx, dtype='cfloat')
    nschat = np.zeros(nx, dtype='cfloat')
    rhsF = np.zeros(nx, dtype='cfloat')
        
    # evaluation of the nonstiff term in spectral space
    AEcuchat = AE.dot(uhat)
    nsc = -np.multiply(np.fft.ifft(uhat), np.fft.ifft(AEcuchat))
    nschat = np.fft.fft(nsc)
    
    i = 0
    
    while t0 <= T:
        # Update of right-hand side
        rhsF = np.fft.fft(rhs(func, nu, x, t0))
        
        # explicit Euler
        if typeODE == "Burgers":
            uhat = uhat + dt * (AI.dot(uhat) + nschat + rhsF)
        
            # Update of nonlinear term
            AEcuchat = AE.dot(uhat)
            nsc = -np.multiply(np.fft.ifft(uhat), np.fft.ifft(AEcuchat))
            nschat = np.fft.fft(nsc)
            
        elif typeODE == "heat":
            uhat = uhat + dt * (AI.dot(uhat))
        
        i = i + 1
        t0 = t0 + dt
        
    return uhat, AI

def impEuler(dt, func, L, nu, u0hat, t0, T, t_int, typeODE, v, x):
    def rhs(func, nu, x, t):
        n = np.shape(x)[0]
        arr = np.zeros(n, dtype='cfloat')
        sigma = 0.004
        
        if func == "exp":
            arr = -np.exp(-((x-0.5)**2)/0.004) * np.sin(t) + (np.exp(-((x-0.5)**2)/0.004) * np.cos(t))**2 * (-(2*(x-0.5))/0.004) - nu * np.cos(t) * (((2*(x-0.5))/0.004) * ((2*(x-0.5))/0.004) * np.exp(-((x-0.5)**2)/0.004) - 2/0.004 * np.exp(-((x-0.5)**2)/0.004))
            
        elif func == "poly":
            f = lambda x, t: nu**2 * x**4 *(x**4 - 1) * np.sin(nu*t) + nu * x**4 * (1 - x**4) * np.cos(nu*t) * 4 * nu * x**3 * (1 - 2*x**4) * np.cos(nu*t) - nu**2 * 4 * x**2 * (3 - 14*x**4) * np.cos(nu*t)
            arr = f(x, t)
            
        return arr
    
    uhat = u0hat
    nx = np.shape(u0hat)[0]
    AE, AI = differentialA(L, nu, nx, typeODE, v)
    
    AEcuchat = np.zeros(nx, dtype='cfloat')
    nsc = np.zeros(nx, dtype='cfloat')
    nschat = np.zeros(nx, dtype='cfloat')
    rhsF = np.zeros(nx, dtype='cfloat')
        
    # evaluation of the nonstiff term in spectral space
    AEcuchat = AE.dot(uhat)
    nsc = -np.multiply(np.fft.ifft(uhat), np.fft.ifft(AEcuchat))
    nschat = np.fft.fft(nsc)
    
    # counter
    i = 0
            
    while t0 <= T:
        rhsF = np.fft.fft(rhs(func, nu, x, t0))
        
        if typeODE == "Burgers":
            uhat = uhat - np.linalg.inv(np.identity(nx) - dt * AI - dt * AE).dot(uhat - uhat - dt * nschat - dt * AI.dot(uhat) - dt * rhsF)
            
        elif typeODE == "heat":
            uhat = uhat - np.linalg.inv(np.identity(nx) - dt * AI).dot(uhat - uhat - dt * AI.dot(uhat))
            
        AEcuchat = AE.dot(uhat)
        nsc = -np.multiply(np.fft.ifft(uhat), np.fft.ifft(AEcuchat))
        nschat = np.fft.fft(nsc) 
        
        i = i + 1
        t0 = t0 + dt
        
    return uhat, AI