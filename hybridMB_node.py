import numpy as np
import math as mt
from greedy_j import greedy_j
from glmnet_python import glmnet
from glmnet_python import glmnetCoef

def hybridMB_node(X, i, gamma, M_tolerance, d_init_factor):
    n= len(X[0])
    m= len(X)
    J = greedy_j(X, i)    
    #projection of j
    dots_j = np.matmul(X[:, J], X)
    X_proj = X - np.outer(X[:, J], (dots_j /(dots_j[J])) )
    # set column j back to original
    X_reg = X_proj
    X_reg[:, J] = X[:, J]
    X_reg_norms = np.zeros(shape=(n, 1))
    X_reg_norms = np.sqrt(sum(X_reg*X_reg))
    
    penalty_fact = np.zeros(n) +1
    penalty_fact[J] = 0
    
    X_prediction = np.transpose(np.transpose(X_reg) / X_reg_norms)
    X_prediction[:, i] = np.zeros(m) 
    Y_prediction = X_reg[:, i]
    inc = 4
    exc = np.array([i])
    Lambda = np.array([2 * max(abs(sum(X_prediction* Y_prediction)))/100])
    
    
    
    # start with lambda that is too big, then stop when it's small enough
    iterations = 16 # shouldn't be important
    for iter in range(iterations) :
        fit = glmnet(
                x= X_prediction,
                y= Y_prediction,
                family="mgaussian",
                alpha=1,
                penalty_factor = penalty_fact,
                lambdau=Lambda,
                standardize = False,
                exclude = exc
                )
        
        rsltbeta= glmnetCoef(fit)
        sigma_hat = np.linalg.norm(Y_prediction - np.matmult(X_prediction, rsltbeta), 'fro') /mt.sqrt(m)
        l = sum(abs(rsltbeta))
        if (gamma * sigma_hat <= l) :
            # l1 norm is big, stop shrinking
            break
    
        Lambda = Lambda/ inc
  
    upper = Lambda * inc
    lower = Lambda
    while (upper > lower * M_tolerance) :
        mid = (upper + lower)/2
        fit = glmnet(
            x= X_prediction,
            y= Y_prediction,
            family="mgaussian",
            alpha=1,
            weights = penalty_fact,
            lambdau=mid,
            standardize = False,
            exclude=  exc             
            )
        rsltbeta= glmnetCoef(fit)
    l = sum(abs(rsltbeta))
    #print(rslt$beta[j])
    #print(l)
    sigma_hat = np.linalg.norm(Y_prediction - np.matmult(X_prediction, rsltbeta), 'fro') / mt.sqrt(m)
    if (gamma * sigma_hat <= l) :
      # l1 norm is big, use more regularization
      lower = mid
    else :
      upper = mid
    
    #print(mid)
  
    beta = rsltbeta[:,1]/ X_reg_norms
    a_j =  beta[J] + dots_j[i] / dots_j[J] - np.matmult(beta,  dots_j) / (dots_j[J])
    
    beta[i] = -1
    beta[J] = a_j
  
    theta_hat = -beta / sigma_hat ^ 2
    return (theta_hat)