# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg


def Laplacian(A):
    n = A.shape[0]
    A = (A + A.T)/2
    P = np.zeros_like(A)
    for i in range(n):
        P[i,i] = np.sum(A[i,:])
    return P - A


def Obj(X, Y, W, H, L, L0, F, Q, alpha, beta, gamma):
    item1 = np.trace(F.T @ L @ F)
    item2 = np.sum((H @ X @ W - H @ F)**2)
    item3 = alpha*np.sum((F-Y)**2)
    item4 = beta*np.trace(F @ L0 @ F.T)
    item5 = gamma*np.trace(W.T @ Q @ W)
    return item1 + item2 + item3 + item4 + item5


def UpdateD(W):
    p, m = W.shape
    D = np.eye(p)
    eps = 1e-50
    for i in range(p):
        ele = np.sqrt(np.sum(W[i,:]**2))
        ele = max(ele, eps)
        D[i,i] = 1/(2*ele)
    return D


def Solve(X, Y, L, L0, alpha, beta, gamma):
    n, p = X.shape
    m = Y.shape[1]
    H = np.eye(n) - np.ones((n, 1)) @ np.ones((1,n))/n
    W = np.random.normal(0, 1, (p,m))
    it = 20
    obj = np.zeros((it,))
    for i in range(it):
        A = (L + H +alpha*np.eye(n))
        D = beta*L0
        E = H @ X @ W + alpha*Y
        F = linalg.solve_sylvester(A, D, E)
        Q = UpdateD(W)
        W = np.linalg.inv(X.T @ H @ X + gamma*Q) @ X.T @ H @ F
        b = (np.ones((1,n)) @ F - np.ones((1,n)) @ X @ W)/n
        obj[i] = Obj(X, Y, W, H, L, L0, F, Q, alpha, beta, gamma)
    return W, b, obj
