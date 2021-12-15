import numpy as np


def int_nodes(a, b, M):
    
    """
    Input:
        a, b         - left and right point of an interval
        M            - number of collocation nodes
        
    Return:
        Gauss-Lobatto nodes
    """

    xtilde = np.zeros(M)
    x = np.zeros(M)

    if M == 3:
        # Gauss-Lobatto nodes for M=3
                
        x[0] = -1
        x[1] = 0
        x[2] = 1
        for i in range(0, M):
            xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    elif M == 4:

        x[0] = -1
        x[1] = -np.sqrt(1/5)
        x[2] = -x[1]
        x[3] = -x[0]
        for i in range(0, M):
            xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    elif M == 5:
        # Gauss-Lobatto nodes for M=5
      
        x[0] = -1
        x[1] = -np.sqrt(3/7)
        x[2] = 0
        x[3] = np.sqrt(3/7)
        x[4] = 1
        for i in range(0, M):
            xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    else:
        print("No nodes implemented.")
    
    
    return xtilde