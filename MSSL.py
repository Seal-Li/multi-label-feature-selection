# -*- coding: utf-8 -*-
import numpy as np
from AffinityMetrics import AffinityMetrics
from Evaluation import Metrics
from PreProcess import PreProcess


def Laplacian(A):
    n = A.shape[0]
    A = (A + A.T)/2
    P = np.zeros_like(A)
    for i in range(n):
        P[i,i] = np.sum(A[i,:])
    return P - A


def Solve(X, Y, L, D, H, alpha, beta):
    n, p = X.shape
    En = np.ones((n,1))
    item1 = np.linalg.inv(D @ ((X.T @ H @ X) + alpha*L) + beta*np.eye(p))
    item2 = D @ X.T @ H @ Y
    W = item1 @ item2
    b = (Y.T @ En - W.T @ X.T @ En)/n
    return W, b


def Update(W):
    p = W.shape[0]
    D = np.eye(p)
    eps = 1e-50
    for i in range(p):
        ele = np.sqrt(np.sum(W[i,:]**2))
        if ele < eps:
            ele = eps
        else:
            D[i,i] = 1/(2*ele)
    return D


def Obj(X, Y, W, b, L, D, alpha, beta):
    n = X.shape[0]
    En = np.ones((n,1))
    item1 = (np.trace(Y.T @ Y))/2
    item2 = np.trace(W.T @ X.T @ Y)
    item3 = (np.trace(W.T @ (X.T @ X + alpha*L + beta*D) @ W))/2
    item4 = np.trace(W.T @ X.T @ En @ b.T)
    item5 = np.trace(Y.T @ En @ b.T)
    item6 = np.trace(b @ En.T @ En @ b.T)/2
    return item1 - item2 + item3 + item4 - item5 + item6


def FeatureSelection(W, n_fea):
    weights = np.sqrt(np.sum(W**2, axis=1))
    index = np.argsort(weights)[-n_fea:]
    W = W[index,:]
    return W, index


def Predict(X, W, b, index):
    n = X.shape[0]
    En = np.ones((n,1))
    X = X[:,index]
    return X @ W + En @ b.T


def Evaluate(truth, pred):
    hl = Metrics(truth, pred).hamming_loss()
    rl = Metrics(truth, pred).ranking_loss()
    oe = Metrics(truth, pred).one_error()
    cov = Metrics(truth, pred).coverage()
    ap = Metrics(truth, pred).average_precision()
    return [hl, rl, oe, cov, ap]


def Iteration(X, Y, L, D, H, alpha, beta):
    it = 20
    obj = np.zeros((it,1))
    for i in range(it):
        W,b = Solve(X, Y, L, D, H, alpha, beta)
        obj[i] = Obj(X, Y, W, b, L, D, alpha, beta)
        D = Update(W)
    return W, b, obj


def main(x_train, y_train, x_test, y_test, L, n_fea, alpha, beta):
    n, p = x_train.shape
    D = np.eye(p)
    H = np.eye(n) - (np.ones((n,1)) @ np.ones((n,1)).T)/n
    W, b, obj = Iteration(x_train, y_train, L, D, H, alpha, beta)
    W, index = FeatureSelection(W, n_fea)
    Y_pre = Predict(x_test, W, b, index)
    loss = Evaluate(y_test, Y_pre)
    return loss, obj

import itertools
if __name__ == '__main__':
    #### 变量区 ####
    data_names = ["Birds", "Business","Emotions","Health","Mediamill","Scene","Yeast"]
    p = [260, 300,72,300,120,294,103]

    k = 6
    data_name = data_names[k]
    p = p[k]

    n_fea = p
    n_feas = [55,60,65,70,75,80,85,90,95,100]
    alphas = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2])
    betas = np.array([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5])

    #### 读取数据文件 ####
    train_path = f"C:\\Users\\dell\\Desktop\\datasets\\{data_name}\\{data_name}-train.csv"

    test_path = f"C:\\Users\\dell\\Desktop\\datasets\\{data_name}\\{data_name}-test.csv"

    data_train = np.genfromtxt(train_path, delimiter=",", skip_header=1)
    data_test = np.genfromtxt(test_path, delimiter=",", skip_header=1)
    x_train, y_train = PreProcess().MissingTreatment(data_train[:,:p], data_train[:,p:])
    x_test, y_test = PreProcess().MissingTreatment(data_test[:,:p], data_test[:,p:])

    A = AffinityMetrics(x_train.T).Boolean()
    L = Laplacian(A)
    s = len(n_feas)
    best_values = np.zeros((5,s))
    for k in range(s):
        Loss = np.zeros((len(alphas), len(betas), 5))
        for i, j in itertools.product(range(len(alphas)), range(len(betas))):
            Loss[i,j,:], obj = main(x_train, y_train, x_test, y_test, L,
                                    n_feas[k], alphas[i], betas[j])

        # best_values = np.zeros((5,))
        for i in range(4):
            best_values[i,k] = np.min(Loss[:,:,i])
        best_values[4,k] = np.max(Loss[:,:,4])
    print(best_values)
