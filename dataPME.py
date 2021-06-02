

"""conditions on ggm
 k non-degenerate
 attractive"""
 
import numpy as np
import math
from GreedyPrune import GreedyPrune
from Matrixcomputation import Precision
from sklearn.covariance import GraphicalLasso
from numpy.random import randn

#"enter sigma, kappa, d, n manually"
#
#"covariance matrix"
#Sigma = np.array([[1, 0.5, 0, 0], 
#                  [0.5, 1, 0, 0], 
#                  [0, 0, 1, 0.5],
#                  [0, 0, 0.5, 1]])
#Omega = np.linalg.inv(Sigma)
#print(Omega)

"dimension"
m=20

"number of samples"
n=20

"non degeneraty"
kappa = 0.25

"maximum degree"
d= 3

#sparsity = order of (d log(2d/k^2))
T= math.ceil (d * (math.log((2*d/(kappa*kappa)), 10)))
#print (T)
# nu =  (k^2)/sqrt(32)
nu = (kappa**2)/math.sqrt(32)




#generate random samples

"adding columns of univariate gaussian to get multivariate"
samp =  randn(n, m)
samp[:, 1]=samp[:, 1] + 4*samp[:, 0]
samp[:, 2]= samp[:, 2] + 2*samp[:, 0]
#samp[:, 3]= samp[:, 3] +samp[:, 4]


#for i in range (n):
#    samp[i]=np.random.multivariate_normal(np.zeros(shape=(m)), Sigma)
    
model= GraphicalLasso(alpha=0.1).fit(samp)
print("Glasso estimate")
print(model.precision_)
#form graph laplacian with computed neighbourhoods
Lap = np.zeros(shape=(m, m))

for i in range(m):
    A= GreedyPrune(i, samp, T, nu)
    B= np.insert(A, i, 0)
    Lap[i]=B  

for i in range(m):
    for j in range(i):
        if (Lap[i][j]!=Lap[j][i]):
            Lap[i][j]=0
            Lap[j][i]=0
#print(Lap)  


#compute precision from graph laplacian
O_hat = Precision(Lap, samp)
print("GreedyPrune estimate")
print(O_hat)