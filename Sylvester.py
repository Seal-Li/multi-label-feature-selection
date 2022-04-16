# -*- coding:utf-8 -*-

import numpy as np

#### Solving Sylvester Equation ####
def MaxEig(X):
    eig, vec = np.linalg.eig(X)
    max_loc = np.argmax(eig)
    return eig[max_loc]
 
   
def ConvergenceFactor(A, B, C, D):
    first = MaxEig(A @ A.T)*MaxEig(B @ B.T)
    second = MaxEig(C @ C.T)*MaxEig(D @ D.T)
    return 2/(first + second)


def Sylvester(A, B, C, D, F):
    M = np.kron(B.T, A) + np.kron(D.T, C)
    rank = np.linalg.matrix_rank(M)
    order = M.shape[0]
    p = A.shape[1]
    m = B.shape[0]
    if rank==order:
        return (np.linalg.inv(M) @ (F.T).reshape(-1,1)).reshape(m,p).T
    num = 10
    X0 = np.random.uniform(0, 1, (p,m))
    u = ConvergenceFactor(A, B, C, D)
    for _ in range(num):
        X1 = X0 + u*A.T @ (F - A @ X0 @ B - C @ X0 @ D) @ B.T
        X2 = X0 + u*C.T @ (F - A @ X0 @ B - C @ X0 @ D) @ D.T
        X0 = (X1 + X2)/2
    return X0

