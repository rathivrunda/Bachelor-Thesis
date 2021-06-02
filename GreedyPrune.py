
import numpy as np

from OMP import OMP
from sklearn.linear_model import LinearRegression


def convar (samp, i, S):
    n = len(samp)
    s =int( sum(sum(S)))
 
    X_i = samp[:, i]
    X_i0 = np.zeros(shape=(n, s))
    
    if (s==0):
        v = np.var(X_i)
        return v
    
    j=0
    for p in range (len(S)):
        if (S[p]==1):
            if (p<i):
                p=p
            else:
                p=p+1
            X_i0[:, j] = samp[:, p]
            j=j+1
            
    model = LinearRegression().fit(X_i0, X_i) 
    
    A=model.coef_
    B=model.intercept_
    
#    print(np.shape(A))
#    print(np.shape(X_i0))
#    print(np.shape(X_i))
#    print(np.shape(B))
    EXY= np.dot(X_i0, A) + B
    "fitting with R=1 for different sample sizes"
#    if (i==0): print(model.score(X_i0, X_i))
    var1= X_i - EXY
#    if (i<3):
#        print (X_i)
#        print(EXY)
    
    " var1 is somehow 0-vector"
    covar= np.mean(var1 * var1)
    
    return covar
    
"if conditional variance is 0, estimate of diagonal term comes out to be infinity" 
def GreedyPrune(i, samp, T, nu):
    
    Y = samp[:, i]
    n = len(samp)
    m= len(samp[0])
    X = np.zeros(shape=(n, m-1))
    
    for j in range (m-1):
        if (j<i):
            X[:, j] = samp[:, j]
        else:
            X[:, j] = samp[:, j+1]
            
    S = OMP (T, X, Y) 
    #print("Greedy S")
    #print(S)
    theta = 1/ convar(samp, i, S)
    #print("nu/theta")
    #print(nu/theta)
    
    for j in range (m-1):
        if (S[j] == 1):
            s = convar(samp, i, S)
            #print(s)
            #print("difference in covariance for")
            #print (j)
            S_ = S
            S_[j]=0
            
            s_= convar(samp, i, S_)
            #print(s_ - s)
            #print(nu/theta)
            if ((s_ - s) < nu/theta):
                S[j]=0      
            else :
                S[j]=1

    return S       
        
