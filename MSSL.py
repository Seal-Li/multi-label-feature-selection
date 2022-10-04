# -*- coding: utf-8 -*-
import numpy as np


def Laplacian(A):
    n = A.shape[0]
    A = (A + A.T)/2
    P = np.zeros_like(A)
    for i in range(n):
        P[i,i] = np.sum(A[i,:])
    return P - A


def Solve(X, Y, L, D, H, alpha, beta):
    n, p = X.shape
    En = np.ones((n,1))
    item1 = np.linalg.inv(D @ ((X.T @ H @ X) + alpha*L) + beta*np.eye(p))
    item2 = D @ X.T @ H @ Y
    W = item1 @ item2
    b = (Y.T @ En - W.T @ X.T @ En)/n
    return W, b


def Update(W):
    p = W.shape[0]
    D = np.eye(p)
    eps = 1e-50
    for i in range(p):
        ele = np.sqrt(np.sum(W[i,:]**2))
        if ele < eps:
            ele = eps
        else:
            D[i,i] = 1/(2*ele)
    return D


def Obj(X, Y, W, b, L, D, alpha, beta):
    n = X.shape[0]
    En = np.ones((n,1))
    item1 = (np.trace(Y.T @ Y))/2
    item2 = np.trace(W.T @ X.T @ Y)
    item3 = (np.trace(W.T @ (X.T @ X + alpha*L + beta*D) @ W))/2
    item4 = np.trace(W.T @ X.T @ En @ b.T)
    item5 = np.trace(Y.T @ En @ b.T)
    item6 = np.trace(b @ En.T @ En @ b.T)/2
    return item1 - item2 + item3 + item4 - item5 + item6


def Iteration(X, Y, L, D, H, alpha, beta):
    it = 20
    obj = np.zeros((it,1))
    for i in range(it):
        W,b = Solve(X, Y, L, D, H, alpha, beta)
        obj[i] = Obj(X, Y, W, b, L, D, alpha, beta)
        D = Update(W)
    return W, b, obj
