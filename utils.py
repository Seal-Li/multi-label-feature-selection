import argparse
import measures
import preprocess
import numpy as np
import pandas as pd


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", 
                        type=str, 
                        default="your data folder")
    parser.add_argument("--data_names", 
                        type=list, 
                        default=["Business", "Emotions", "Health", 
                                 "Mediamill", "Scene", "Yeast"])
    parser.add_argument("--data_dict", 
                        type=dict,
                        default={"Business": {"feature": 300, "label": 8},
                                 "Emotions": {"feature": 72, "label": 6},
                                 "Health": {"feature": 300, "label": 10}, 
                                 "Mediamill": {"feature": 120, "label": 6},
                                 "Scene": {"feature": 294, "label": 6}, 
                                 "Yeast": {"feature": 103, "label": 14}})
    return parser.parse_args()


def pre_process(path, data_dict, name):
    processor = preprocess.PreProcess()
    data_train = pd.read_csv(f"{path}/{name}/{name}-train.csv", delimiter=",")
    data_test = pd.read_csv(f"{path}/{name}/{name}-test.csv", delimiter=",")
    x_train = data_train.iloc[:, :data_dict[name]["feature"]].values
    x_test = data_test.iloc[:, :data_dict[name]["feature"]].values
    y_train = data_train.iloc[:, data_dict[name]["feature"]:].values
    y_test = data_test.iloc[:, data_dict[name]["feature"]:].values
    x_train, y_train = processor.nonsense_treat(x_train, y_train)
    x_test, y_test = processor.nonsense_treat(x_test, y_test)
    return x_train, y_train, x_test, y_test


def laplacian(A):
    n = A.shape[0]
    P = np.eye(n)
    A = (A + A.T) / 2
    for i in range(n):
        P[i, i] = np.sum(A[i, :])
    return P - A


def evaluate(y_true, y_pre):
    pre_labels = y_pre >= 0.5
    MLM = measures.MultiLabelMetrics(y_true, pre_labels)
    RM = measures.RankingMetrics(y_true, y_pre)
    HL = round(MLM.hamming_loss(), 4)
    RL = round(RM.ranking_loss(), 4)
    OE = round(RM.one_error(), 4)
    Cov = round(RM.coverage(), 4)
    AP = round(RM.average_precision(), 4)
    return [HL, RL, OE, Cov, AP]