# -*- coding: utf-8 -*-
import numpy as np
from PreProcess import PreProcess
from Evaluation import Metrics
from AffinityMetrics import AffinityMetrics
from scipy import linalg

def Laplacian(A):
    n = A.shape[0]
    A = (A + A.T)/2
    P = np.zeros_like(A)
    for i in range(n):
        P[i,i] = np.sum(A[i,:])
    L = P - A
    return L

def FeatureSelection(W, n_fea):
    weights = np.sqrt(np.sum(W**2, axis=1))
    index = np.argsort(weights)[-n_fea:]
    W = W[index,:]
    return W, index

def Predict(X, W, b, index):
    n = X.shape[0]
    En = np.ones((n,1))
    X = X[:,index]
    y_pre = X @ W + En @ b
    return y_pre

def Evaluate(truth, pred):
    hl = Metrics(truth, pred).hamming_loss()
    rl = Metrics(truth, pred).ranking_loss()
    oe = Metrics(truth, pred).one_error()
    cov = Metrics(truth, pred).coverage()
    ap = Metrics(truth, pred).average_precision()
    loss = [hl, rl, oe, cov, ap]
    return loss

def Obj(X, Y, W, H, L, L0, F, Q, alpha, beta, gamma):
    item1 = np.trace(F.T @ L @ F)
    item2 = np.sum((H @ X @ W - H @ F)**2)
    item3 = alpha*np.sum((F-Y)**2)
    item4 = beta*np.trace(F @ L0 @ F.T)
    item5 = gamma*np.trace(W.T @ Q @ W)
    obj = item1 + item2 + item3 + item4 + item5
    return obj

def UpdateD(W):
    p, m = W.shape
    D = np.eye(p)
    eps = 1e-50
    for i in range(p):
        ele = np.sqrt(np.sum(W[i,:]**2))
        if ele < eps:
            ele = eps
        D[i,i] = 1/(2*ele)
    return D

def Solve(X, Y, L, L0, alpha, beta, gamma):
    n, p = X.shape
    m = Y.shape[1]
    H = np.eye(n) - np.ones((n, 1)) @ np.ones((1,n))/n
    W = np.random.normal(0, 1, (p,m))
    it = 20
    obj = np.zeros((it,))
    for i in range(it):
        A = (L + H +alpha*np.eye(n))
        D = beta*L0
        E = H @ X @ W + alpha*Y
        F = linalg.solve_sylvester(A, D, E)
        Q = UpdateD(W)
        W = np.linalg.inv(X.T @ H @ X + gamma*Q) @ X.T @ H @ F
        b = (np.ones((1,n)) @ F - np.ones((1,n)) @ X @ W)/n
        obj[i] = Obj(X, Y, W, H, L, L0, F, Q, alpha, beta, gamma)
    return W, b, obj

if __name__ == '__main__':
    
    #### 变量区 ####
    data_names = ["Business","Emotions","Health","Mediamill","Scene","Yeast"]
    p = [300,72,300,120,294,103]
    k = 0
    data_name = data_names[k]
    p = p[k]
    n_fea = p
    alphas = np.array([1])
    beta = 1
    gammas = np.array([0.01])
    #### 读取数据文件 ####
    train_path = r"C:\Users\dell\Desktop\datasets\{}\{}-train.csv".format(data_name, data_name)
    test_path = r"C:\Users\dell\Desktop\datasets\{}\{}-test.csv".format(data_name, data_name)
    data_train = np.genfromtxt(train_path, delimiter=",", skip_header=1)
    data_test = np.genfromtxt(test_path, delimiter=",", skip_header=1)
    x_train, y_train = PreProcess().MissingTreatment(data_train[:,:p], data_train[:,p:])
    x_test, y_test = PreProcess().MissingTreatment(data_test[:,:p], data_test[:,p:])
    m = y_train.shape[1]
    
    A = AffinityMetrics(x_train).Kernel()
    A0 = AffinityMetrics(y_train.T).Kernel(m-1)
    L = Laplacian(A)
    L0 = Laplacian(A0)
    
    Loss = np.zeros((len(alphas), len(gammas), 5))
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            W, b, obj = Solve(x_train, y_train, L, L0, alphas[i], beta, gammas[j])
            W, index = FeatureSelection(W, n_fea)
            Y_pre = Predict(x_test, W, b, index)
            Loss[i,j,:] = Evaluate(y_test, Y_pre)
    
    best_values = np.zeros((5,))
    for i in range(4):
        best_values[i] = np.min(Loss[:,:,i])
    best_values[4] = np.max(Loss[:,:,4])



