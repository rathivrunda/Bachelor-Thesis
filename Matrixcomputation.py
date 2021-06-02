
from GreedyPrune import convar
import numpy as np
from OMP import pick
from sklearn.linear_model import LinearRegression


def Precision(Lap, samp):
    
    m=len(Lap)

    O_hat = np.zeros(shape=(m, m))
    for i in range (m):
        preS =np.reshape((Lap[i]), (-1, 1))
        S = np.delete(preS, i , axis=0)
#        if (i<2): print(convar(samp, i, S))
        O_hat[i][i] = 1/convar(samp, i, S)
        
        if(sum(Lap[i])>0):
            X_i= samp[:, i]
            X_s= pick(samp, preS)
            model = LinearRegression().fit(X_s, X_i)
            w = model.coef_
            
            #print(w)
            
            k=0
            for j in range(m):
                if(Lap[i][j]==1):
                    O_hat[i][j]=(-1)*(w[k])*(O_hat[i][i])
                    #print (O_hat[i][j])
                    k=k+1
                    
    for i in range(m):
        for j in range(i):
            if (abs(O_hat)[i][j]<abs(O_hat)[j][i]):
                O_hat[j][i]=O_hat[i][j]
            else:
                O_hat[i][j]=O_hat[j][i]
    return O_hat