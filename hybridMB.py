import numpy as np

from hybridMB_node import hybridMB_node


def hybridMB(X, gamma):
    M_tolerance = 1.01
    d_init_factor = 2
    n = len(X[0])
    
    Theta = np.zeros(shape=(n,n))
    for i in range(n):
        Theta[i] = hybridMB_node(X, i, gamma, M_tolerance, d_init_factor)
        
    for i in range(n):
        for j in range(i):
            if (abs(Theta)[i][j]<abs(Theta)[j][i]):
                Theta[j][i]=Theta[i][j]
            else:
                Theta[i][j]=Theta[j][i]

    return Theta