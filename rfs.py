import numpy as np
from numpy.linalg import inv

class RFS(object):
    def __init__(self, x, y, maxIter=50, eps=1e-20):
        self.x = x.T
        self.y = y
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.m = y.shape[1]
        self.maxIter = maxIter
        self.eps = eps

    def update(self, U):
        D = np.eye(self.n + self.p)
        for i in range(self.n + self.p):
            denomenator = np.sqrt(np.sum(U[i, :] ** 2))
            D[i, i] = (2 * denomenator + self.eps)
        return D
    
    def rfs(self, gamma):
        I = np.eye(self.n)
        A = np.concatenate((self.x.T, gamma * I), axis=1)
        D = np.eye(self.n + self.p)
        U = np.ones((self.n + self.p, self.m))
        for i in range(self.maxIter):
            U = inv(D) @ A.T @ inv(A @ inv(D) @ A.T) @ self.y
            D = self.update(U)
        return U[:self.p, :]