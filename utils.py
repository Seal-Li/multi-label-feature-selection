import argparse
import measures
import preprocess
import numpy as np
import pandas as pd


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
                        default="C:/Users/lihai/Desktop/MLFS/data")
    parser.add_argument("--save_path", type=str, 
                        default="C:/Users/lihai/Desktop/MLFS/num_feature_result")
    parser.add_argument("--data_names", type=list, 
                        default=["emotions", "image", "scene", "yeast"])
    parser.add_argument("--data_dict", type=dict,
                        default={"emotions": {"feature": 72, "label": 6},
                                 "image": {"feature": 294, "label": 5},
                                 "scene": {"feature": 294, "label": 6}, 
                                 "yeast": {"feature": 103, "label": 14}})
    parser.add_argument("--parameter", type=dict,
                        default={
                            "emotions": {
                                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2], 
                                "beta": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2], 
                                "rho": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                "walk_length": 50},
                            "image": {
                                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1], 
                                "beta": [1, 1e1, 1e2], 
                                "rho": [0.0, 0.3],
                                "walk_length": 50},
                            "scene": {
                                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1], 
                                "beta": [1, 1e1, 1e2], 
                                "rho": [0.0, 0.3, 0.7, 0.8],
                                "walk_length": 50},
                            "yeast": {
                                "alpha": [1e-3, 1e-2, 1e-1, 1], 
                                "beta": [1, 1e1, 1e2, 1e3], 
                                "rho": [0.0, 0.6, 0.8],
                                "walk_length": 50}})
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


def feature_selection(W, num_fea):
    feature_score = np.sqrt(np.sum(W ** 2, axis=1))
    return np.argsort(feature_score)[-num_fea:]


def evaluate(y_true, y_pre):
    MLM = measures.MultiLabelMetrics(y_true, y_pre)
    RM = measures.RankingMetrics(y_true, y_pre)
    HL = round(MLM.hamming_loss(), 4)
    RL = round(RM.ranking_loss(), 4)
    OE = round(RM.one_error(), 4)
    Cov = round(RM.coverage(), 4)
    AP = round(RM.average_precision(), 4)
    return [HL, RL, OE, Cov, AP]
