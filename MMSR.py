# -*- coding: utf-8 -*-
import numpy as np


def RandomWalk(A, step=100):
    n = A.shape[0]
    C = np.zeros_like(A)
    nodes = np.arange(n)
    for i in range(n):
        if (A[i,:]==0).all():
            A[i,:] = np.random.uniform(0, 1, (1,n))

    for i in range(n):
        prob = (A[i,:]/np.sum(A[i,:])).reshape(n,)
        for _ in range(step):
            index = np.random.choice(nodes, 1, p=prob)
            C[i,index] = C[i,index] + 1
            A_ind = A[index,:]
            A_ind[0,index] = 0
            prob = (A_ind/np.sum(A_ind)).reshape(n,)
    return (C + C.T)/2

def Laplacian(A):
    n = A.shape[0]
    A = (A + A.T)/2
    P = np.zeros_like(A)
    for i in range(n):
        P[i,i] = np.sum(A[i,:])
    return P - A

def Solve(X, Y, L, D, V, H, alpha, beta, gamma):
    n, p = X.shape
    En = np.ones((n,1))
    temp = X.T @ H @ X + alpha*X.T @ L @ X + beta*np.eye(p) + gamma*D
    W = np.linalg.inv(temp) @ X.T @ H @ Y
    b = (En.T @ V @ Y - En.T @ V @ X @ W)/n
    
    return W, b, V

def Update(W):
    p = W.shape[0]
    D = np.eye(p)
    eps = 1e-64
    for i in range(p):
        ele = np.sqrt(np.sum(W[i,:]**2) + eps)
        D[i,i] = 1/(2*ele)
    return D

def Obj(X, Y, W, b, V, H, L, D, alpha, beta, gamma):
    n, p = X.shape
    En = np.ones((n,1))
    item1 = np.trace(W.T @ X.T @ V @ X @ W)
    item2 = np.trace(b.T @ En.T @ V @ X @ W)
    item3 = np.trace(Y.T @ V @ X @ W)
    item4 = np.trace(b.T @ En.T @ V @ En @ b)
    item5 = np.trace(Y.T @ V @ En @ b)
    item6 = np.trace(Y.T @ V @ Y)
    item7 = np.trace(W.T @ X.T @ L @ X @ W)
    item8 = np.trace(W.T @ W)
    item9 = np.trace(W.T @ D @ W)
    return item1 / 2 + item2 - item3 / 2 + item4 / 2 - item5 + item6 / 2 + alpha * item7 / 2 + beta * item8 / 2 + gamma * item9 / 2

def Iteration(X, Y, L, D, V, H, alpha, beta, gamma):
    it = 30
    obj = np.zeros((it,1))
    for i in range(it):
        W, b, V = Solve(X, Y, L, D, V, H, alpha, beta, gamma)
        obj[i] = Obj(X, Y, W, b, V, H, L, D, alpha, beta, gamma)
        D = Update(W)
    return W, b, V, obj
