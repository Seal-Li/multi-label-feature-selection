# -*- coding: utf-8 -*-
import numpy as np
from PreProcess import PreProcess
from Evaluation import Metrics

def Evaluate(Y, Y_pre):
    hammingloss = Metrics(Y, Y_pre).hamming_loss()
    rankingloss = Metrics(Y, Y_pre).ranking_loss()
    one_error = Metrics(Y, Y_pre).one_error()
    coverage = Metrics(Y, Y_pre).coverage()
    averageprecision = Metrics(Y, Y_pre).average_precision()
    
    loss=[hammingloss, rankingloss, one_error, coverage, averageprecision]
    return loss

def CorrEntropy(X, Y, U, sigma=1):
    n, p = X.shape
    P = np.eye(n)
    ele = X @ U - Y
    for i in range(n):
        P[i,i] = np.exp(-np.sum(ele[i,:]**2))
    return P

def UpdateD(U):
    p = U.shape[0]
    D = np.eye(p)
    eps = 1e-64
    for i in range(p):
        ele = np.sqrt(np.sum(U[i,:]**2))
        D[i,i] = (2*ele + eps)
    return D

def Obj(X, Y, U, P, Q, alpha):
    n, p = X.shape
    item1 = 0
    item2 = alpha*np.trace(U.T @ Q @ U)
    for i in range(n):
        item1 = item1 + np.exp(-np.sum((X[i,:] @ U - Y[i,:])**2))
    obj = 1 - item1 + item2
    return obj

def CRFS(X, Y, alpha):
    n, p = X.shape
    m = Y.shape[1]
    U = np.zeros((p,m))
    it = 50
    obj = np.zeros((it,))
    for i in range(it):
        P = CorrEntropy(X, Y, U)
        Q = UpdateD(U)
        temp = np.linalg.inv(X.T @ P @ X + alpha*Q)
        U = temp @ X.T @ P @ Y
        obj[i] = Obj(X, Y, U, P, Q, alpha)
    return U, obj
        
def FeatureSelection(W, n_fea):
    weights = np.sqrt(np.sum(W**2, axis=1))
    indexes = np.argsort(weights)[-n_fea:]
    W = W[indexes,:]
    return W, indexes

def main(x_train, y_train, x_test, y_test, alpha, n_fea):
    U, obj = CRFS(x_train, y_train, alpha)
    U, index = FeatureSelection(U, n_fea)
    x_train = x_train[:,index]
    n = x_train.shape[0]
    H = np.eye(n) - (np.ones((n,1)) @ np.ones((1,n)))/n
    W = np.linalg.inv(x_train.T @ H @ x_train) @ x_train.T @ H @ y_train
    b = (np.ones((1,n)) @ y_train - np.ones((1,n)) @ x_train @ W)/n
    x_test = x_test[:,index]
    n1 = x_test.shape[0]
    y_pre = x_test @ W + np.ones((n1,1)) @ b
    loss = Evaluate(y_test, y_pre)
    return loss

if __name__ == '__main__':
    #### 变量区 ####
    data_names = ["Business","Emotions","Health","Mediamill","Scene","Yeast"]
    p = [300,72,300,120,294,103]
    
    k = 4
    data_name = data_names[k]
    p = p[k]
    n_fea = 50
    #### 读取数据文件 ####
    train_path = r"C:\Users\dell\Desktop\datasets\{}\{}-train.csv".format(data_name, data_name)
    test_path = r"C:\Users\dell\Desktop\datasets\{}\{}-test.csv".format(data_name, data_name)
    data_train = np.genfromtxt(train_path, delimiter=",", skip_header=1)
    data_test = np.genfromtxt(test_path, delimiter=",", skip_header=1)
    x_train, y_train = PreProcess().MissingTreatment(data_train[:,:p], data_train[:,p:])
    x_test, y_test = PreProcess().MissingTreatment(data_test[:,:p], data_test[:,p:])
    n, p = x_train.shape
    
    alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1])
    Loss = np.zeros((len(alphas), 5))
    for i in range(len(alphas)):
        Loss[i,:] = main(x_train, y_train, x_test, y_test, alphas[i], n_fea)
    
    best_values = np.zeros((5,))
    for i in range(4):
        best_values[i] = np.min(Loss[:,i])
    best_values[4] = np.max(Loss[:,4])
        