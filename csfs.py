import numpy as np
from numpy.linalg import inv


class CSFS(object):
    def __init__(self, x, y, S=None, max_iter=50, eps=1e-64):
        self.x = x.T
        self.y = y
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.m = y.shape[1]
        self.S = S if S is not None else np.eye(self.n)
        self.weight = np.ones((1, self.n)) @ self.S @ np.ones((self.n, 1))
        self.H = np.eye(self.n) - (np.ones((self.n, 1)) @ np.ones((1, self.n)) @ self.S) / self.weight
        self.max_iter = max_iter
        self.eps = eps        
    
    def updateD(self, W, D):
        for i in range(self.p):
            denominator = np.sqrt(np.sum(W[i, :] ** 2))
            D[i, i] = 1 / (2 * denominator + self.eps)
        return D
    
    def updateb(self, W, F):
        item1 = F.T  @ self.S @ np.ones((self.n, 1))
        item2 = W.T @ self.x @ self.S @ np.ones((self.n, 1))
        return (item1 + item2) / self.weight
    
    def updateF(self, F):
        F[F>1] = 1
        F[F<0] = 0
        return F
    
    def csfs(self, mu):
        D = np.eye(self.p)
        F = self.y
        item1 = self.x @ self.H @ self.S @ self.H @ self.x.T
        item2 = self.x @ self.H @ self.S @ self.H @ self.y
        for _ in range(self.max_iter):
            W = inv(item1 + mu * D) @ item2
            b = self.updateb(W, F)
            F = self.updateF(self.x.T @ W  + np.ones((self.n, 1)) @ b.T)
        return W, F
