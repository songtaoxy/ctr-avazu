#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
from DeepFM import DeepFM

#gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)

def getweekday(x):
    '''
    :param x: YYMMDD
    :return: weekday
    '''
    sdt = str(x)
    year = int(sdt[0:2])
    month = int(sdt[2:4])
    day = int(sdt[4:6])
    dt = datetime.date(year, month, day)
    weekday = dt.weekday()
    return weekday

def create_feature(data):
    data["size"] = data["C15"].str.cat(data["C16"], sep="_")
    # 将hour列拆分为
    data["hour1"] = data["hour"].map(lambda x: str(x)[6:8])
    data["day"] = data["hour"].map(lambda x: str(x)[4:6])
    data["weekday"] = data["hour"].map(lambda x: getweekday(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    #此处可以考虑将组合特征的源特征删掉，对比效果
    data = data.drop(["site_id","app_id","device_model","hour","C15","C16"], axis=1)
    return data

def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "click": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


# load data
datapath = "/data/barnett007/ctr-data"
trainfile = os.path.join(datapath ,"train.csv")
dfTrain = pd.read_csv(trainfile,dtype={"C15":str,"C16":str})
print("*********create_feature_begin***********" + str(datetime.datetime.now()))
dfTrain = create_feature(dfTrain)
print("*********create_feature_end***********" + str(datetime.datetime.now()))
print("*********train_data_read_end***********" + str(datetime.datetime.now()))

y_train = dfTrain["click"].values
X_train = dfTrain.drop(["click","id"],axis=1).values
# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))
del X_train
del y_train

print("*********folds_end***********" + str(datetime.datetime.now()))

testfile = os.path.join(datapath ,"test.csv")
dfTest = pd.read_csv(trainfile,dtype={"C15":str,"C16":str})

dfTest = create_feature(dfTest)

print("*********test_data_read_end***********" + str(datetime.datetime.now()))


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
print("*********BEGIN_RUN***********" + str(datetime.datetime.now()))
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)
print("*********END_RUN***********" + str(datetime.datetime.now()))

# ------------------ FM Model ------------------
#fm_params = dfm_params.copy()
#fm_params["use_deep"] = False
#y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


# ------------------ DNN Model ------------------
#dnn_params = dfm_params.copy()
#dnn_params["use_fm"] = False
#y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)
