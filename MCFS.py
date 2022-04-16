# -*- coding: utf-8 -*-
import scipy
import numpy as np
from sklearn import linear_model
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
