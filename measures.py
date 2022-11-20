# -*- coding: utf-8 -*-
import numpy as np


class SingleLabelMetrics(object):
    """
    This class deal with binary classification problems only.
    Tor multi-classification problems, 
    you can refer to the class named MultiLabelMetrics.
    Warning:pred is label, not probability!!!
    """
    def __init__(self, truth, pred):
        self.truth = truth
        self.pred = pred
        self.n = truth.shape[0]
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.Accuracy = 0
        self.Precision = 0
        self.Recall = 0
        self.F_Beta_Score = 0
        self.MCC = 0
        self.AUC = 0
        
    def confusion_matrix(self):
        """
        TP:True Positives,表示实际为正例且被分类器判定为正例的样本数
        FP:False Positives,表示实际为负例且被分类器判定为正例的样本数
        FN:False Negatives,表示实际为正例但被分类器判定为负例的样本数
        TN:True Negatives,表示实际为负例且被分类器判定为负例的样本数
        """
        self.TP = np.sum(self.pred[self.truth==1]==1)
        self.FP = np.sum(self.pred[self.truth==0]==1)
        self.FN = np.sum(self.pred[self.truth==1]==0)
        self.TN = np.sum(self.pred[self.truth==0]==0)
        return self.TP, self.FP, self.FN, self.TN
    
    def accuracy(self):
        self.TP, self.FP, self.FN, self.TN = self.confusion_matrix()
        ele = self.TP + self.TN
        den = self.TP + self.TN + self.FP + self.FN
        self.Accuracy = ele / den
        return self.Accuracy
    
    def precision(self):
        eps = 1e-64
        self.TP, self.FP, self.FN, self.TN = self.confusion_matrix()
        self.Precision = self.TP / (self.TP + self.FP + eps)
        return self.Precision
    
    def recall(self):
        self.TP, self.FP, self.FN, self.TN = self.confusion_matrix()
        self.Recall = self.TP / (self.TP + self.FN)
        return self.Recall
    
    def F_beta_score(self, beta=1):
        self.TP, self.FP, self.FN, self.TN = self.confusion_matrix()
        self.Recall = self.recall()
        self.Precision = self.precision()
        ele =(1 + beta ** 2) * self.Precision * self.Recall
        den = (beta ** 2) * self.Precision + self.Recall
        self.F_Beta_Score = ele / den
        return self.F_Beta_Score
    
    def Mat_cor_coef(self):
        '''
        Matthews correlation coefficient:马修斯相关系数.
        the range of MCC is [-1,1], the higher, the better.
        Warning:this metrics will be unavailable 
        when the number of instance is very large!!!
        '''
        self.TP, self.FP, self.FN, self.TN = self.confusion_matrix()
        # print(self.TP, self.FP, self.FN, self.TN)
        ele = (self.TP * self. TN) - (self.TP * self.FN)
        temp1 = (self.TP + self.FP) * (self.TP + self.FN)
        temp2 = (self.TN + self.FP) * (self.TN + self.FN)
        den = np.sqrt(temp1 * temp2)
        self.MCC = ele / den
        return self.MCC
    
    def ROC(self, probability):
        import matplotlib.pyplot as plt
        FPR = []
        TPR = []
        kvDict = {"FPR":FPR, "TPR":TPR}
        negNum = np.sum(self.truth==False)
        posNum = np.sum(self.truth==True)
        threshold = np.sort(probability)[::-1]

        for th in threshold:
            predLabel = (probability>=th)
            fpr = np.sum(self.truth[predLabel] == False) / negNum
            tpr = np.sum(self.truth[predLabel] == True) / posNum
            FPR.append(fpr)
            TPR.append(tpr)

        plt.figure(dpi=900)
        plt.plot(FPR, TPR, "b-", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.show()
        return kvDict
    
    def AUC_(self, probability):
        posInds = np.where(self.truth==True)[0]
        negInds = np.where(self.truth==False)[0]
        for posInd in posInds:
            for negInd in negInds:
                if probability[posInd] > probability[negInd]:
                    self.AUC = self.AUC + 1
                elif probability[posInd] == probability[negInd]:
                    self.AUC = self.AUC + 0.5
        self.AUC = self.AUC / (len(posInds) * len(negInds))
        return self.AUC


class MultiLabelMetrics(object):
    """
    Multi-label or Multi-Class Metrics based on label, but not probability.
    if this class is used to Multi-Class problems, each Class should be transform
    to a label, such as:
    Y = [[1], 
         [2], 
         [2], 
         [3]] 
    should be set as follows:
    Y = [[1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1]]
    """
    def __init__(self, truth, pred):
        self.truth = truth
        self.pred = pred
        self.n = truth.shape[0]
        self.m = truth.shape[1]
        self.weight = (1 / self.m) * np.ones((self.m,))
        self.HammingLoss = 0
        self.JaccardIndex = 0
        self.PreciseMatching = 0
        self.MacroF1 = 0
        self.MicroF1 = 0
        self.WeightF1 = 0
        self.MacroAUC = 0
    
    def hamming_loss(self):
        miss_pairs = np.sum(self.pred != self.truth)
        self.HammingLoss = miss_pairs / (self.n * self.m)
        return self.HammingLoss
    
    def jaccard_index(self):
        truthPositive = np.sum(self.truth==1)
        predPositive = np.sum(self.pred==1)
        TP = np.sum(self.pred[self.truth==1]==1)
        self.JaccardIndex = TP / (truthPositive + predPositive - TP)
        return self.JaccardIndex

    def precise_matching(self):
        """
        Warning:this metrics is not available when belongs to none of labels!!! 
        """
        for i in range(self.n):
            flag = (self.truth[i, :] == self.pred[i, :]).all()
            if flag:
                self.PreciseMatching = self.PreciseMatching + 1 / self.n
        return self.PreciseMatching
    
    def macro_F1(self):
        Pr = np.zeros((self.m,))
        Re = np.zeros((self.m,))
        for i in range(self.m):
            SLM = SingleLabelMetrics(self.truth[:,i], self.pred[:,i])
            Pr[i] = SLM.precision()
            Re[i] = SLM.recall()
        Pr_macro = np.sum(Pr) / self.m
        Re_macro = np.sum(Re) / self.m
        ele = 2 * Pr_macro * Re_macro
        den = Pr_macro + Re_macro
        self.MacroF1 = ele / den
        return self.MacroF1
    
    def micro_F1(self):
        TP = np.zeros((self.m,))
        FP = np.zeros((self.m,))
        FN = np.zeros((self.m,))
        TN = np.zeros((self.m,))
        for i in range(self.m):
            SLM = SingleLabelMetrics(self.truth[:,i], self.pred[:,i])
            TP[i], FP[i], FN[i], TN[i] = SLM.confusion_matrix()
        Pr_micro = np.sum(TP) / (np.sum(TP) + np.sum(FP))
        Re_micro = np.sum(TP) / (np.sum(TP) + np.sum(FN))
        ele = 2 * Pr_micro * Re_micro
        den = Pr_micro + Re_micro
        self.MicroF1 = ele / den
        return self.MicroF1
    
    def weight_F1(self, weight):
        Pr = np.zeros((self.m,))
        Re = np.zeros((self.m,))
        for i in range(self.m):
            SLM = SingleLabelMetrics(self.truth[:,i], self.pred[:,i])
            Pr[i] = SLM.precision()
            Re[i] = SLM.recall()
        Pr_weight = np.sum(Pr * weight)
        Re_weight = np.sum(Re * weight)
        ele = 2 * Pr_weight * Re_weight
        den = Pr_weight + Re_weight
        self.WeightF1 = ele / den
        return self.WeightF1


class RankingMetrics(object):
    """
    Warnings:An instance belongs to all of labels is not permission!!!
    """
    def __init__(self, truth, prob):
        self.truth = truth
        self.prob = prob
        self.n = truth.shape[0]
        self.m = truth.shape[1]
        self.RankingLoss = 0
        self.OneError = 0
        self.Coverage = 0
        self.AveragePrecision = 0
        
    def ranking_loss(self):
        for i in range(self.n):
            if np.sum(self.truth[i, :]) in [self.m, 0]:
                continue
            label_0 = (self.truth[i, :] == 0)
            label_1 = (self.truth[i, :] == 1)
            den = np.sum(label_0) * np.sum(label_1)  # number of base pairs
            max_0 = np.max(self.prob[i, label_0])  # the max posibility of instance with label 0
            ele = np.sum(self.prob[i, label_1] <= max_0)  # the number of ranking error
            # print(ele)
            self.RankingLoss = self.RankingLoss + (ele / den) / self.n
        return self.RankingLoss
    
    def one_error(self):
        for i in range(self.n):
            maxIndex = np.argmax(self.prob[i, :])  # index of max posibility
            flag = (self.truth[i, maxIndex] != 1)  # the label in max_index is 1 or not
            if flag:
                self.OneError = self.OneError + 1 / self.n
        return self.OneError
    
    def coverage(self):
        for j in range(self.n):
            ind = np.argsort(self.prob[j, :])
            sortReal = self.truth[j, ind]
            num = 0
            for i in range(self.m):
                flag = (sortReal[i] == 1)
                if flag:
                    break
                else:
                    num = num + 1
            k = self.m - num - 1
            self.Coverage = self.Coverage + k / self.n
        return self.Coverage
    
    def average_precision(self):
        precision = 0
        for i in range(self.n):
            if np.sum(self.truth[i, :]) in [self.m, 0]:
                self.n = self.n - 1
                continue
            index = np.where(self.truth[i] == 1)[0]
            score = self.prob[i, :][index]
            score = sorted(score)
            scoreAll = sorted(self.prob[i])
            precision_tmp = 0
            for item in score:
                tmp1 = score.index(item)
                tmp1 = len(score) - tmp1
                tmp2 = scoreAll.index(item)
                tmp2 = len(scoreAll) - tmp2
                precision_tmp += tmp1 / tmp2
            precision += precision_tmp / len(score)
        self.Average_Precision = precision / self.n
        return self.Average_Precision


if __name__ == '__main__':
    np.random.seed(666)
    n, m = 1000, 8
    Y = np.random.randint(0, 2, (n,m))
    pred = np.random.randint(0, 2, (n,m))
    prob = np.random.uniform(0, 1, (n,m))
    
    SLM = SingleLabelMetrics(Y, pred)
    Tp, Fp, Fn, Tn = SLM.confusion_matrix()
    Accuracy = SLM.accuracy()
    Precision = SLM.precision()
    Recall = SLM.recall()
    FBetaScore = SLM.F_beta_score(beta=1)
    # Mcc = SLM.Mat_cor_coef()
    SLM1 = SingleLabelMetrics(Y[:,0], pred[:,0])
    kvDict = SLM1.ROC(prob[:,0])
    Auc = SLM1.AUC_(prob[:,0])

    
    MLM = MultiLabelMetrics(Y, pred)
    HammingLoss = MLM.hamming_loss()
    JaccardIndex = MLM.jaccard_index()
    PreciseMatching = MLM.precise_matching()
    MacroF1 = MLM.macro_F1()
    MicroF1 = MLM.micro_F1()
    weight = np.ones((m,)) * (1/m)
    WeightF1 = MLM.weight_F1(weight)
    
    RM = RankingMetrics(Y, prob)
    RankingLoss = RM.ranking_loss()
    OneError = RM.one_error()
    Coverage = RM.coverage()
    AveragePrecsion = RM.average_precision()
    