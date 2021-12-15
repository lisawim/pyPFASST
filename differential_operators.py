import numpy as np

def differentialA(L, nu, nx, typeODE, v):
    
    """
    Input:
        L            - period of function
        nu           - diffusion coefficient
        nx           - spatial degrees of freedom
        typeODE      - type of ODE to be solved
        v            - advection speed
        
    Return:
        Differential operators for implicit and explicit piece
    """
                
    if nx%2 == 0:
        a = np.concatenate((np.arange(0, (nx / 2)), 0), axis=None)
        #a = np.arange(0, (nx / 2) + 1)
        b = np.arange(((nx / 2) + 1 - nx), 0)
        
    else:
        a = np.arange(0, (nx - 1)/2 + 1)
        b = np.arange(-(nx - 1)/2, 0)
    

    AE = np.zeros((nx, nx), dtype='cfloat')
    AI = np.zeros((nx, nx), dtype='cfloat')
    
    k = (2 * np.pi)/L
    
    # Gradient
    AE = k * 1j * np.diag(np.concatenate((a, b), axis=None))
    
    # Laplacian   
    AI = nu * (-1) * k**2 * np.diag(np.concatenate(((a ** 2), (b ** 2)), axis=None))
    
    if typeODE == "advdif":
        AE = (-v) * AE
        
    return AE, AI
