import numpy as np


class MIFS(object):
    def __init__(self, x, y, L, 
                 lr_V=1e-10, lr_B=1e-10, lr_W=1e-10, 
                 maxIter=100, eps=1e-64):
        self.x = x
        self.y = y
        self.L = L
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.m = y.shape[1]
        self.lr_V = lr_V
        self.lr_B = lr_B
        self.lr_W = lr_W
        self.maxIter = maxIter
        self.eps = eps

    def update(self, W, D, V, B, dw, dv, db):
        V = V - self.lr_V * dv
        B = B - self.lr_B * db
        W = W - self.lr_W * dw
        
        for i in range(self.p):
            denominator = np.sqrt(np.sum(W[i, :] ** 2))
            D[i,i] = 1 / (2 * denominator + self.eps)
        return W, D, V, B
    
    def derives(self, W, D, V, B, alpha, beta, gamma):
        dw = 2 * (self.x.T @ (self.x @ W - V) + gamma * D @ W)
        dv = 2 * ((V - self.x @ W) + alpha * (V @ B - self.y) @ B.T + beta * self.L @ V)
        db = 2 * alpha * V.T @ (V @ B - self.y)
        return dw, dv, db

    def mifs(self, alpha, beta, gamma):
        np.random.seed(20221028)
        W = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y
        V = np.random.randint(0, 2, (self.n, self.m))
        B = np.eye(self.m)
        D = np.eye(self.p)
        for _ in range(self.maxIter):
            dw, dv, db = self.derives(W, D, V, B, alpha, beta, gamma)
            W, D, V, B = self.update(W, D, V, B, dw, dv, db)
        return W, B
    
    def loss(self, W, D, V, B, L, alpha, beta, gamma):
        item1 = np.sum((self.x @ W - V) ** 2)
        item2 = alpha * (np.sum((self.y - V @ B) ** 2))
        item3 = beta * np.trace(V.T @ L @ V)
        item4 = 2 * gamma * np.trace(W.T @ D @ W)
        return item1 + item2 + item3 + item4