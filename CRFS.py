# -*- coding: utf-8 -*-
import numpy as np

def CorrEntropy(X, Y, U, sigma=1):
    n, p = X.shape
    P = np.eye(n)
    ele = X @ U - Y
    for i in range(n):
        P[i,i] = np.exp(-np.sum(ele[i,:]**2))
    return P

def UpdateD(U):
    p = U.shape[0]
    D = np.eye(p)
    eps = 1e-64
    for i in range(p):
        ele = np.sqrt(np.sum(U[i,:]**2))
        D[i,i] = (2*ele + eps)
    return D

def Obj(X, Y, U, P, Q, alpha):
    n, p = X.shape
    item1 = 0
    item2 = alpha*np.trace(U.T @ Q @ U)
    for i in range(n):
        item1 = item1 + np.exp(-np.sum((X[i,:] @ U - Y[i,:])**2))
    return 1 - item1 + item2

def CRFS(X, Y, alpha):
    n, p = X.shape
    m = Y.shape[1]
    U = np.zeros((p,m))
    it = 50
    obj = np.zeros((it,))
    for i in range(it):
        P = CorrEntropy(X, Y, U)
        Q = UpdateD(U)
        temp = np.linalg.inv(X.T @ P @ X + alpha*Q)
        U = temp @ X.T @ P @ Y
        obj[i] = Obj(X, Y, U, P, Q, alpha)
    return U, obj
