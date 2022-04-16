# -*- coding: utf-8 -*-
import numpy as np
from PreProcess import PreProcess
from Evaluation import Metrics
from AffinityMetrics import AffinityMetrics


def RandomWalk(A, step=100):
    n = A.shape[0]
    C = np.zeros_like(A)
    nodes = np.arange(n)
    for i in range(n):
        if (A[i,:]==0).all():
            A[i,:] = np.random.uniform(0, 1, (1,n))

    for i in range(n):
        prob = (A[i,:]/np.sum(A[i,:])).reshape(n,)
        for _ in range(step):
            index = np.random.choice(nodes, 1, p=prob)
            C[i,index] = C[i,index] + 1
            A_ind = A[index,:]
            A_ind[0,index] = 0
            prob = (A_ind/np.sum(A_ind)).reshape(n,)
    return (C + C.T)/2

def Laplacian(A):
    n = A.shape[0]
    A = (A + A.T)/2
    P = np.zeros_like(A)
    for i in range(n):
        P[i,i] = np.sum(A[i,:])
    return P - A

def Solve(X, Y, L, D, V, H, alpha, beta, gamma):
    n, p = X.shape
    En = np.ones((n,1))
    temp = X.T @ H @ X + alpha*X.T @ L @ X + beta*np.eye(p) + gamma*D
    W = np.linalg.inv(temp) @ X.T @ H @ Y
    b = (En.T @ V @ Y - En.T @ V @ X @ W)/n
    
    return W, b, V

def Update(W):
    p = W.shape[0]
    D = np.eye(p)
    eps = 1e-64
    for i in range(p):
        ele = np.sqrt(np.sum(W[i,:]**2) + eps)
        D[i,i] = 1/(2*ele)
    return D

def Obj(X, Y, W, b, V, H, L, D, alpha, beta, gamma):
    n, p = X.shape
    En = np.ones((n,1))
    item1 = np.trace(W.T @ X.T @ V @ X @ W)
    item2 = np.trace(b.T @ En.T @ V @ X @ W)
    item3 = np.trace(Y.T @ V @ X @ W)
    item4 = np.trace(b.T @ En.T @ V @ En @ b)
    item5 = np.trace(Y.T @ V @ En @ b)
    item6 = np.trace(Y.T @ V @ Y)
    item7 = np.trace(W.T @ X.T @ L @ X @ W)
    item8 = np.trace(W.T @ W)
    item9 = np.trace(W.T @ D @ W)
    return item1 / 2 + item2 - item3 / 2 + item4 / 2 - item5 + item6 / 2 + alpha * item7 / 2 + beta * item8 / 2 + gamma * item9 / 2

def FeatureSelection(W, n_fea):
    weights = np.sqrt(np.sum(W**2, axis=1))
    index = np.argsort(weights)[-n_fea:]
    W = W[index,:]
    return W, index

def Predict(X, W, b, index):
    n = X.shape[0]
    En = np.ones((n,1))
    X = X[:,index]
    return X @ W + En @ b

def Evaluate(truth, pred):
    hl = Metrics(truth, pred).hamming_loss()
    rl = Metrics(truth, pred).ranking_loss()
    oe = Metrics(truth, pred).one_error()
    cov = Metrics(truth, pred).coverage()
    ap = Metrics(truth, pred).average_precision()
    return [hl, rl, oe, cov, ap]

def Iteration(X, Y, L, D, V, H, alpha, beta, gamma):
    it = 30
    obj = np.zeros((it,1))
    for i in range(it):
        W, b, V = Solve(X, Y, L, D, V, H, alpha, beta, gamma)
        obj[i] = Obj(X, Y, W, b, V, H, L, D, alpha, beta, gamma)
        D = Update(W)
    return W, b, V, obj

def main(x_train, y_train, x_test, y_test, V, L, n_fea, alpha, beta, gamma):
    np.random.seed(6)
    n, p = x_train.shape
    D = np.diag(np.random.uniform(1, 100, p))
    H = V - (V @ np.ones((n,1)) @ np.ones((n,1)).T @ V)/n
    W, b, V, obj = Iteration(x_train, y_train, L, D, V, H, alpha, beta, gamma)
    W, index = FeatureSelection(W, n_fea)
    Y_pre = Predict(x_test, W, b, index)
    loss = Evaluate(y_test, Y_pre)
    return loss, obj, V, Y_pre

import itertools
if __name__ == '__main__':
    #### 变量区 ####
    data_names = ["Business","Emotions","Health","Mediamill","Scene","Yeast"]
    p = [300,72,300,120,294,103]
    s = 21
    k = 5
    data_name = data_names[k]
    p = p[k]
    n_fea = 80

    alphas = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    # alphas = np.logspace(-6, -2, 20)
    # betas = np.linspace(0, 1, s)
    # beta = betas[5]
    betas = [0.05, 0.2, 0.5, 0.05, 0.1, 0.25]
    beta = betas[k]
    gammas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    # gammas = np.logspace(0, 2, 10)
    # gammas = np.arange(2,8,0.5)
    #### 读取数据文件 ####
    train_path = f"C:\\Users\\dell\\Desktop\\datasets\\{data_name}\\{data_name}-train.csv"

    test_path = f"C:\\Users\\dell\\Desktop\\datasets\\{data_name}\\{data_name}-test.csv"

    data_train = np.genfromtxt(train_path, delimiter=",", skip_header=1)
    data_test = np.genfromtxt(test_path, delimiter=",", skip_header=1)
    x_train, y_train = PreProcess().MissingTreatment(data_train[:,:p], data_train[:,p:])
    x_test, y_test = PreProcess().MissingTreatment(data_test[:,:p], data_test[:,p:])
    m = y_train.shape[1]

    n, p = x_train.shape
    V = np.eye(n)

    step = 50
    A = AffinityMetrics(x_train).Boolean()
    # S = RandomWalk(A, step)
    L = Laplacian(A)
    best_values = np.zeros((5,))
    Loss = np.zeros((len(alphas), len(gammas), 5))
    for i, j in itertools.product(range(len(alphas)), range(len(gammas))):
        Loss[i,j,:], obj, V, Y_pre = main(x_train, y_train, x_test, y_test,
                                          V, L, n_fea, alphas[i],
                                          beta*gammas[j],
                                          (1-beta)*gammas[j])
    for i in range(4):
        best_values[i] = np.min(Loss[:,:,i])
    best_values[4] = np.max(Loss[:,:,4])
    # print(best_values)
