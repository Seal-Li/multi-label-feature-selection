# -*- coding: utf-8 -*-
import scipy
import numpy as np
from sklearn import linear_model
from PreProcess import PreProcess
from Evaluation import Metrics
from AffinityMetrics import AffinityMetrics

def mcfs(X, n_fea, n_clusters=5):
    A = AffinityMetrics(X).Boolean()
    A = (A + A.T)/2
    A_norm = np.diag(np.sqrt(1/np.sum(A, axis=1)))
    AT = (A_norm @ (A @ A_norm)).T
    A[A < AT] = AT[A < AT]
    eigen_value, ul = scipy.linalg.eigh(a=A)
    Y = A_norm @ ul[:, -1*n_clusters-1:-1]
    
    n_sample, n_feature = X.shape
    W = np.zeros((n_feature, n_clusters))
    for i in range(n_clusters):
        clf = linear_model.Lars(n_nonzero_coefs=n_fea)
        clf.fit(X, Y[:, i])
        W[:, i] = clf.coef_
    return W

def feature_ranking(W):
    mcfs_score = W.max(1)
    idx = np.argsort(mcfs_score, 0)
    idx = idx[::-1]
    return idx

def Evaluate(truth, pred):
    hl = Metrics(truth, pred).hamming_loss()
    rl = Metrics(truth, pred).ranking_loss()
    oe = Metrics(truth, pred).one_error()
    cov = Metrics(truth, pred).coverage()
    ap = Metrics(truth, pred).average_precision()
    loss = [hl, rl, oe, cov, ap]
    return loss


if __name__ == '__main__':
    #### 变量区 ####
    data_names = ["Business","Emotions","Health","Mediamill","Scene","Yeast"]
    p = [300,72,300,120,294,103]
    
    k = 5
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
    S = mcfs(x_train, n_fea=p, n_clusters=5)
    index = feature_ranking(S)[:n_fea]
    
    x_train = x_train[:,index]
    x_test = x_test[:,index]
    
    H = np.eye(n) - (np.ones((n,1)) @ np.ones((1,n)))/n
    W = np.linalg.inv(x_train.T @ H @ x_train) @ x_train.T @ H @ y_train
    b = (np.ones((1,n)) @ y_train - np.ones((1,n)) @ x_train @ W)/n
    
    n1 = x_test.shape[0]
    Y_pre = x_test @ W + np.ones((n1,1)) @ b
    loss = Evaluate(y_test, Y_pre)
    print(loss)