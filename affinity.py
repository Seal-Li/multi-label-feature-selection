import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class Affinity(object):
    def __init__(self, data):
        self.data = data
        self.n = data.shape[0]
        self.p = data.shape[1]
    
    
    def kernel(self, k=10, sigma=1, keep_all=False):
        distances = euclidean_distances(self.data, self.data)
        distances = distances + np.eye(self.n) * np.max(distances) * k
        
        if keep_all:
            return np.exp(-distances / sigma ** 2)

        A = np.zeros_like(distances)
        kth_value = np.sort(distances, axis=1)[:, k]
        for i in range(self.n):
            index = np.where(distances[i, :] >= kth_value[i])
            A[i, :] = np.exp(-distances[i, :] / sigma ** 2)
            A[i, index] = 0
        return A

    def boolean(self, k=10):
        A = np.zeros((self.n, self.n))
        distances = euclidean_distances(self.data, self.data)
        distances = distances + np.eye(self.n) * np.max(distances) * k
        index = np.argsort(distances, axis=1)[:, :k]
        
        for i in range(self.n):
            A[i, index[i, :]] = 1
        return A

    def cosine(self, k=10):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                A[i, j] = self.data[i, :] @ self.data[j, :].T
                A[j, i] = A[i, j]
            A[i, i] = 0
        
        kth_value = np.sort(A, axis=1)[:, -k]
        for i in range(self.n):
            index = np.where(A[i, :] < kth_value[i])
            A[i, index] = 0
        return A

    def jaccard(self):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            item1 = self.data[i, :] @ self.data[i, :].T
            for j in range(i, self.n):
                element = self.data[i, :] @ self.data[j, :].T
                item2 = self.data[j, :] @ self.data[j, :].T
                item3 = self.data[i, :] @ self.data[j, :].T
                denominator = item1 + item2 - item3
                A[i, j] = element / denominator
                A[j, i] = A[i, j]
            A[i, i] = 0
        return A