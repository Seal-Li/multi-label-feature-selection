import numpy as np


class MSSL(object):
    def __init__(self, x, y, L, max_iter=20, eps=1e-64):
        self.x = x
        self.y = y
        self.L = L
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.m = y.shape[1]
        self.H = np.eye(x.shape[0]) - (np.ones((x.shape[0], 1)) @ np.ones((x.shape[0], 1)).T) / self.n
        self.eps = eps
        self.max_iter = max_iter

    def mmsl(self, alpha, beta):
        En = np.ones((self.n, 1))
        D = np.eye(self.p)
        item1 = np.linalg.inv(D @ ((self.x.T @ self.H @ self.x) + alpha * self.L) + beta * np.eye(self.p))
        for _ in range(self.max_iter):
            item2 = D @ self.x.T @ self.H @ self.y
            W = item1 @ item2
            b = (self.y.T @ En - W.T @ self.x.T @ En) / self.n
            D = self.update(W)
        return W, b

    def update(self, W):
        D = np.eye(self.p)
        for i in range(self.p):
            denominator = np.sqrt(np.sum(W[i, :] ** 2))
            D[i,i] = 1 / (2 * denominator + self.eps)
        return D

    def loss(self, W, b, D, alpha, beta):
        En = np.ones((self.n, 1))
        item1 = (np.trace(self.y.T @ self.y)) / 2
        item2 = np.trace(W.T @ self.x.T @ self.y)
        item3 = (np.trace(W.T @ (self.x.T @ self.x + alpha * self.L + beta * D) @ W)) / 2
        item4 = np.trace(W.T @ self.x.T @ En @ b.T)
        item5 = np.trace(self.y.T @ En @ b.T)
        item6 = np.trace(b @ En.T @ En @ b.T) / 2
        return item1 - item2 + item3 + item4 - item5 + item6
