import numpy as np


def greedy_j(X, i):
    dots = np.matmul(X[:, i], X)
    print(dots)
    norms = np.sqrt(sum(X*X))
    score = np.divide (abs(dots), norms)
    score[i] = 0
    
    j= np.where(score == np.amax(score))[0][0]
    
    return j

    
    