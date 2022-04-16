# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

"""
示例：
A = AffinityMetrics(X).Boolean()
A = AffinityMetrics(X).Kernel()
A = AffinityMetrics(X).Joint(Y)
"""

class AffinityMetrics():
    def __init__(self, X):
        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
    
    def Boolean(self, k=10):
        """
        计算bool值近邻矩阵
        """
        A = np.zeros((self.n, self.n))
        distances = euclidean_distances(self.X, self.X)
        max_value = np.max(distances)
        distances = distances + np.eye(self.n)*max_value*k
        order = np.argsort(distances, axis=1)[:,:k]
        
        for i in range(self.n):
            A[i,order[i,:]] = 1
        return A
    
    def Kernel(self, k=10, sigma=1):
        """
        计算高斯核近邻矩阵
        """
        A = np.zeros((self.n, self.n))
        distances = euclidean_distances(self.X, self.X)
        max_value = np.max(distances)
        distances = distances + np.eye(self.n)*max_value*k
        kth_value = np.sort(distances, axis=1)[:,k]
        
        for i in range(self.n):
            index = np.where(distances[i,:]>=kth_value[i])
            A[i,:] = np.exp(-distances[i,:]/sigma)
            A[i,index] = 0
        return A
    
    def Cosine(self, k=10):
        """
        计算正弦值近邻矩阵
        """
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                A[i,j] = self.X[i,:] @ self.X[j,:].T
                A[j,i] = A[i,j]
            A[i,i] = 0
        
        kth_value = np.sort(A, axis=1)[:,-k]
        for i in range(self.n):
            index = np.where(A[i,:]<kth_value[i])
            A[i,index] = 0
        return A
    
    def Joint(self, Y, sigma=1):
        """
        计算基于变量的高斯核与基于标签的Jaccard指数的联合近邻矩阵
        """
        A = np.zeros((self.n, self.n))
        """计算gaussian kernel"""
        distances = euclidean_distances(self.X, self.X)
        
        """计算jaccard指数"""
        Jac = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i,self.n):
                ele = Y[i,:] @ Y[j,:].T
                den = Y[i,:] @ Y[i,:].T + Y[j,:] @ Y[j,:].T - Y[i,:] @ Y[j,:].T
                Jac[i,j] = ele/den
                Jac[j,i] = Jac[i,j]
            Jac[i,i] = 0
        A = Jac*np.exp(-distances/sigma)
        return A, Jac
