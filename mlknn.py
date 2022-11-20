import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class MLKNN(object):
    def __init__(self, x_train, y_train, x_test, k=10, s=1):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.k = k
        self.s = s
        self.n = x_train.shape[0]
        self.nt = x_test.shape[0]
        self.p = x_train.shape[1]
        self.m = y_train.shape[1]
        self.train_distances = euclidean_distances(self.x_train, self.x_train) + np.eye(self.n) * np.sum(x_train ** 2)
        self.test_distances = euclidean_distances(self.x_test, self.x_train)
        self.train_neibours = np.argsort(self.train_distances, axis=1)[:, :self.k].astype(int)
        self.test_neibours = np.argsort(self.test_distances, axis=1)[:, :self.k].astype(int)

    def prior_prob(self):
        Pr1 = (self.s + np.sum(self.y_train, axis=0)) / (2 * self.s + self.n)
        Pr0 = 1 - Pr1
        return Pr0, Pr1

    def posterior_prob(self):
        Post0 = np.zeros((self.m, self.k+1))
        Post1 = np.zeros((self.m, self.k+1))
        for l in range(self.m):
            c0 = np.zeros((self.k+1, ))
            c1 = np.zeros((self.k+1, ))
            for i in range(self.n):
                delta = np.sum(self.y_train[self.train_neibours[i, :], l])
                if self.y_train[i, l] == 1:
                    c1[int(delta)] += 1
                else:
                    c0[int(delta)] += 1
            for j in range(self.k + 1):
                Post1[l, j] = (self.s + c1[j]) / (self.s * (self.k + 1) + np.sum(c1))
                Post0[l, j] = (self.s + c0[j]) / (self.s * (self.k + 1) + np.sum(c0))
        return Post0, Post1

    def predict(self):
        Pr0, Pr1 = self.prior_prob()
        Post0, Post1 = self.posterior_prob()
        y_pre = np.zeros((self.nt, self.m))
        
        for i in range(self.nt):
            for l in range(self.m):
                ind = np.sum(self.y_train[self.test_neibours[i, :], l])
                y0 = Pr0[l] * Post0[l, int(ind)]
                y1 = Pr1[l] * Post1[l, int(ind)]
                y_pre[i, l] = 1 if (y1 > y0) else 0
        return y_pre
