import numpy as np

def exact_sol(func, nu, T, x):
    n = np.shape(x)[0]
    u_exact = np.zeros(n)
    
    if func == 'exp':
        u_exact = np.exp(-((x-0.5)**2)/0.004) * np.cos(T)
        
    elif func == 'poly':
        u_exact = nu * x**4 * (1 - x**4) * np.cos(nu*T)
               
    elif func == 'sin_advdif':
        # exact solution for advection-diffusion equation (was proven)
        u_exact = np.sin(2 * np.pi * 1 * (x - T)) * np.exp(-T * nu * (2 * np.pi * 1)**2) # k = 1
        #u_exact = np.sin(2 * np.pi * 4 * (x - T)) * np.exp(-T * nu * (2 * np.pi * 4)**2) # k = 4
        
    elif func == 'sin_heat':    
        # exact solution for heat equation forced
        u_exact = np.sin(np.pi * 4 * x) * np.cos(T)
        
    return u_exact