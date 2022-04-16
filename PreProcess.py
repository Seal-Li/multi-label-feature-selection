# -*- coding: utf-8 -*-
import numpy as np

"""
示例：
X = np.genfromtxt(filepath, delimiter=",", skip_header=1)
X_st = PreProcess().Standardization(X)
X_norm = PreProcess().Normalization(X)
"""

class PreProcess():
    def __init__(self):
        return None
    
    def Standardization(self, X):
        """
        数据标准化
        """
        n, p = X.shape
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        for i in range(p):
            X[:,i] = (X[:,i] - mean[i])/std[i]
        return X
    
    def Normalization(self, X):
        """
        数据归一化
        """
        n, p = X.shape
        Max = np.max(X, axis=0)
        Min = np.min(X, axis=0)
        for i in range(p):
            X[:,i] = (X[:,i] - Min[i])/(Max[i] - Min[i])
        return X
    
    def ANL(self, Y):
        """
        计算平均每个样本所属的标签数
        """
        n, m = Y.shape
        lab_num = np.sum(Y)
        return lab_num/n

    def PMC(self, Y):
        """
        计算同时属于多个标签（至少属于两个标签）的样本，占所有样本的比率
        """
        n = Y.shape[0]
        v = np.sum(Y, axis=1)
        m = np.sum(v==1)
        num = n - m
        return num/n
    
    def MissingTreatment(self, X, Y):
        n = X.shape[0]
        m = Y.shape[1]
        index = []
        for i in range(n):
            vx = np.sum(X[i,:])
            vy = np.sum(Y[i,:])
            if (vy==0) or (vx==0) or (vy==m):
                index.append(i)
            else:
                continue
        X = np.delete(X, index, axis=0)
        Y = np.delete(Y, index, axis=0)
        return X, Y
                
