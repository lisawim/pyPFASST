import numpy as np
import numpy.polynomial.legendre as legendre


def int_nodes(a, b, n):
    
    # Input:
    # a, b - Interval of the quadrature nodes which are determined
    # n    - number of nodes
    
    xtilde = np.zeros(n)

    #if n == 3:
        # Gauss-Lobatto nodes for n=3
    #    x = np.zeros(n)
        
    #    x[0] = -1
    #    x[1] = 0
    #    x[2] = 1
    #    for i in range(0, np.shape(x)[0]):
    #        xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    #elif n == 4:
    #    x = np.zeros(n)
        
    #    x[0] = -1
    #    x[1] = -np.sqrt(1/5)
    #    x[2] = -x[1]
    #    x[3] = -x[0]
    #    for i in range(0, np.shape(x)[0]):
    #        xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    #elif n == 5:
        # Gauss-Lobatto nodes for n=5
    #    x = np.zeros(n)
        
    #    x[0] = -1
    #    x[1] = -np.sqrt(3/7)
    #    x[2] = 0
    #    x[3] = np.sqrt(3/7)
    #    x[4] = 1
    #    for i in range(0, np.shape(x)[0]):
    #        xtilde[i] = (b - a) / 2 * x[i] + (a + b) / 2
            
    #else:
    #    print("No nodes implemented.")

    
    roots = legendre.legroots(legendre.legder(np.array([0] * (n - 1) + [1], dtype=np.float64)))
    xtilde = np.array(np.append([-1.0], np.append(roots, [1.0])), dtype=np.float64)

    xtilde = (a * (1 - xtilde) + b * (1 + xtilde)) / 2
    
    
    return xtilde