import numpy as np
from sklearn.linear_model import LinearRegression


def pick (X, S):
    m= int(sum(sum(S)))
    n = len(X)
    X_s = np.zeros(shape=(n, m))
    
    j=0
    for p in range (len(S)):
        if (S[p]==1):
            
            X_s[:, j] = X[:, p]
            j=j+1
        
    return X_s


def OMP (T, X, Y):
    m = len(X[0]) 
    S= np.zeros(shape=(m, 1))
    
    for t in range (T):
        mini = np.zeros(shape=(m, 1))
        
        for j in range(m):
            if (S[j]==0):
                S[j]=1
                X_s= pick (X, S)
                model = LinearRegression().fit(X_s , Y) 
                intercept = model.intercept_
                mini[j] = np.sum(intercept*intercept)
                S[j]=0
            else:
                mini[j]= 1000000
                
        result=np.where(mini == np.amin(mini))
        J = result[0][0]
        S[J]=1
       # print (mini)
    return S