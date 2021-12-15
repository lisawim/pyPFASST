import numpy as np

def ic(x, func, nu):
    
    """
    Input:
        x            -    x-values
        func         -    function for initial condition (fixed for a specific ODE)
        nu           -    diffusion coefficient
        
    Return:
        initial condition for a problem
    """


    n = np.shape(x)[0]
    u0 = np.zeros(n)
    L = np.zeros(1)
    
    if func == 'exp':
        sigma = 0.004
        u0 = np.exp(-((x - 0.5)**2)/sigma)
        L = 1
        
    elif func == 'sin':
        u0 = np.sin(x)
        L = 2*np.pi
        
    elif func == 'poly':
        u0 = x**4 * (1-x)**4
        L = 1
        
    elif func == 'sin_advdif':
        # initial condition for advection diffusion equation
        u0 = np.sin(2 * np.pi * 1 * (x - 0)) * np.exp(-0 * nu * (2 * np.pi * 1)**2)  # k = 1
        #u0 = np.sin(2 * np.pi * 4 * (x - 0)) * np.exp(-0 * nu * (2 * np.pi * 4)**2)  # k = 4
        L = 1
        
    elif func == 'sin_heat':    
        # initial condition for heat equation forced
        u0 = np.sin(np.pi * 4 * x) * np.cos(0)
        L = 1

    return u0, L