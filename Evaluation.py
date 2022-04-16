# -*- coding: utf-8 -*-
import numpy as np

"""
示例：
data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
X, Y = myfunc1(data)
Y_pre = myfunc2(X)

Hamming_Loss = Metrics(Y, Y_pre).hamming_loss()
Ranking_Loss = Metrics(Y, Y_pre).ranking_loss()
...
"""

class Metrics():
    """
    This class would be used to get various evaluation index in the field of 
    multi-label learning.
    It contains some loss functions as follow:
    1:"hamming loss"
    2:"ranking loss"
    3:"one error"
    4:"coverage"
    5:"jaccard index"
    6:"precision"
    7:"recall"
    8:"precise matching"
    """
    
    def __init__(self, real, pred):
        self.real = real
        self.pred = pred
        self.n = pred.shape[0]
        self.p = pred.shape[1]
    
    def hamming_loss(self, threshold=0.5):
        self.pred = (self.pred>=threshold).astype("int32")
        Hamming_Loss = np.sum(self.pred!=self.real)/(self.n*self.p)
        return Hamming_Loss
    
    def ranking_loss(self):
        Ranking_Loss = 0
        for i in range(self.n):
            label_0 = (self.real[i,:]==0)
            label_1 = (self.real[i,:]==1)
            den = np.sum(label_0)*np.sum(label_1)  # number of base pairs
            max_0 = np.max(self.pred[i,label_0])  # the max posibility of instance with label 0
            ele = np.sum(self.pred[i,label_1]<=max_0)  # the number of ranking error
            Ranking_Loss = Ranking_Loss + (ele/den)/self.n
        return Ranking_Loss
    
    def one_error(self):
        One_Error= 0
        for i in range(self.n):
            max_index = np.argmax(self.pred[i,:])  # index of max posibility
            flag = (self.real[i, max_index]!=1)  # the label in max_index is 1 or not
            if flag:
                One_Error = One_Error + 1/self.n
        return One_Error
    
    def coverage(self):
        Coverage = 0
        for j in range(self.n):
            ind = np.argsort(self.pred[j,:])
            sort_real = self.real[j,ind]
            num = 0
            for i in range(self.p):
                flag = (sort_real[i]==1)
                if flag:
                    break
                else:
                    num = num + 1
            k = self.p - num - 1
            Coverage = Coverage + k/self.n
        return Coverage
    
    def average_precision(self):
        precision = 0
        for i in range(self.n):
            index = np.where(self.real[i]==1)[0]
            score = self.pred[i,:][index]
            score = sorted(score)
            score_all = sorted(self.pred[i])
            precision_tmp = 0
            for item in score:
                tmp1 = score.index(item)
                tmp1 = len(score) - tmp1
                tmp2 = score_all.index(item)
                tmp2 = len(score_all) - tmp2
                precision_tmp += tmp1 / tmp2
            precision += precision_tmp / len(score)
        Average_Precision = precision / self.n
        return Average_Precision
    
    def jaccard_index(self):
        JaccardIndex = 0
        pre_label = (self.pred>=0.5)
        real_1 = np.sum(self.real==1)
        pred_1 = np.sum(pre_label==1)
        intersect_1 = np.sum((self.real+pre_label)==2)
        JaccardIndex = intersect_1/(real_1 + pred_1 - intersect_1)
        return JaccardIndex
    
    def precision(self):
        Precision = 0
        pre_label = (self.pred>=0.5)
        pred_1 = np.sum(pre_label==1)
        inter_1 = np.sum((self.real+pre_label)==2)
        Precision = inter_1/pred_1
        return Precision
    
    def recall(self):
        Recall = 0
        pre_label = (self.pred>=0.5)
        real_1 = np.sum(self.real==1)
        inter_1 = np.sum((self.real+pre_label)==2)
        Recall = inter_1/real_1
        return Recall
    
    def precise_matching(self):
        Precise_Matching = 0
        pred_label = (self.pred>=0.5)
        for i in range(self.n):
            flag = (self.real[i,:]==pred_label[i,:]).all()
            if flag:
                Precise_Matching = Precise_Matching + 1/self.n            
        return Precise_Matching
    
