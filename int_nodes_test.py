import numpy as np

# Input:
# a, b - Interval of the quadrature nodes which are determined
# x    - nodes which are transformed

def int_nodes(a, b, n):
    xtilde = np.zeros(n)

    if n == 3:
        # Gauss-Lobatto nodes for n=3
        x = np.zeros(n)
        
        x[0] = -1
        x[1] = 0
        x[2] = 1
        for i in range(0, np.shape(x)[0]):
            xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    elif n == 4:
        x = np.zeros(n)
        
        x[0] = -1
        x[1] = -np.sqrt(1/5)
        x[2] = -x[1]
        x[3] = -x[0]
        for i in range(0, np.shape(x)[0]):
            xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    elif n == 5:
        # Gauss-Lobatto nodes for n=5
        x = np.zeros(n)
        
        x[0] = -1
        x[1] = -np.sqrt(3/7)
        x[2] = 0
        x[3] = np.sqrt(3/7)
        x[4] = 1
        for i in range(0, np.shape(x)[0]):
            xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    else:
        print("No nodes implemented.")

    return xtilde