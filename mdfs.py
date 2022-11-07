import numpy as np
from scipy import linalg


class MDFS(object):
    def __init__(self, x, y, L, L0, max_iter=50, eps=1e-64):
        self.x = x
        self.y = y
        self.L = L
        self.L0 = L0
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.m = y.shape[1]
        self.H = np.eye(self.n) - np.ones((self.n, 1)) @ np.ones((1, self.n)) / self.n
        self.max_iter = max_iter
        self.eps = eps
    
    def update(self, W):
        D = np.eye(self.p)
        for i in range(self.p):
            denominator = np.sqrt(np.sum(W[i, :] ** 2))
            D[i,i] = 1 / (2 * denominator + self.eps)
        return D

    def mdfs(self, alpha, beta, gamma):
        np.random.seed(20221028)
        W = np.random.normal(0, 1, (self.p, self.m))
        A = (self.L + self.H + alpha * np.eye(self.n))
        D = beta * self.L0
        for _ in range(self.max_iter):
            E = self.H @ self.x @ W + alpha * self.y
            F = linalg.solve_sylvester(A, D, E)
            Q = self.update(W)
            W = np.linalg.inv(self.x.T @ self.H @ self.x + gamma * Q) @ self.x.T @ self.H @ F
            b = (np.ones((1, self.n)) @ F - np.ones((1, self.n)) @ self.x @ W) / self.n
        return W, b

    def loss(self, W, F, Q, alpha, beta, gamma):
        item1 = np.trace(F.T @ self.L @ F)
        item2 = np.sum((self.H @ self.x @ self.W - self.H @ F) ** 2)
        item3 = alpha * np.sum((F - self.y) ** 2)
        item4 = beta * np.trace(F @ self.L0 @ F.T)
        item5 = gamma * np.trace(W.T @ Q @ W)
        return item1 + item2 + item3 + item4 + item5