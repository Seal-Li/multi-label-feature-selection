import numpy as np


class RandomWalk(object):
    def __init__(self, S, walk_length=50, method="dfs"):
        self.S = S
        self.n = S.shape[0]
        self.walk_length = walk_length
        self.method = method

    def trans_prob(self):
        prob = np.zeros_like(self.S)
        for i in range(self.n):
            denominator = np.sum(self.S[i, :])
            if denominator == 0:
                prob[i, :] = np.ones((self.n, )) / self.n
            else:
                prob[i, :] = self.S[i, :] / denominator
        return prob

    def random_walk(self):
        np.random.seed(20221028)
        prob = self.trans_prob()
        nodes = np.arange(self.n)
        traces = np.zeros((self.n, self.n))
        for i in range(self.n):
            p = prob[i, :] / np.sum(prob[i, :])
            if self.method == "dfs":
                for _ in range(self.walk_length):
                    index = np.random.choice(nodes, 1, p=p)
                    traces[i, index] = traces[i, index] + 1
                    p = prob[index, :]
                    p[0, index] = 0
                    p = (p / np.sum(p)).reshape(self.n, )
            elif self.method == "bfs":
                for _ in range(self.walkLength):
                    index = np.random.choice(nodes, 1, p=p)
                    traces[i, index] += 1
        return (traces + traces.T) / 2

class MSFS(object):
    def __init__(self, x, y, L, max_iter=50, eps=1e-64):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.m = y.shape[1]
        self.L = L
        self. H = np.eye(self.n) - np.ones((self.n, 1)) @ np.ones((1, self.n)) / self.n
        self.max_iter = max_iter
        self.eps = eps

    def update_U(self, W, U):
        for i in range(self.p):
            denominator = np.sqrt(np.sum(W[i, :] ** 2))
            U[i, i] = 1 / (2 * denominator + self.eps)
        return U

    def loss(self, W, b, U, alpha, beta, rho):
        item1 = np.trace(W.T @ self.x.T @ self.x @ W) / 2
        item2 = np.trace(b.T @ np.ones((1, self.n)) @ self.x @ W)
        item3 = - np.trace(self.y.T @ self.x @ W)
        item4 = np.trace(b.T @ np.ones((1,self.n)) @ np.ones((self.n ,1)) @ b) / 2
        item5 = - np.trace(self.y.T @ np.ones((self.n, 1)) @ b)
        item6 = np.trace(self.y.T @ self.y) / 2
        item7 = alpha * np.trace(W.T @ self.x.T @ self.L @ self.x @ W) / 2
        item8 = beta * rho * np.trace(W.T @ U @ W) / 2
        item9 = beta * (1 - rho) * np.trace(W.T @ W) / 2
        return item1 + item2 + item3 + item4 + item5 + item6 + item7 + item8 + item9

    def msfs(self, alpha, beta, rho):
        np.random.seed(20221028)
        U = np.diag(np.random.uniform(1, 100, self.p))
        W = np.random.uniform(20, 100, (self.p, self.m))
        b = np.zeros((1, self.m))
        item1 = self.x.T @ (self.H + alpha * self.L) @ self.x
        item2 = beta * (1 - rho) * np.eye(self.p)
        for _ in range(self.max_iter):
            item3 = beta * rho * U
            item4 = self.x.T @ self.H @ self.y
            W = np.linalg.inv(item1 + item2 + item3) @ item4
            b = (np.ones((1, self.n)) @ self.y - np.ones((1, self.n)) @ self.x @ W) / self.n
            U = self.update_U(W, U)
        return W, b
