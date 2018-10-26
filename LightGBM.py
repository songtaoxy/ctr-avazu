#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame
import sys,os
import xgboost as xgb
import lightgbm as lgb
import datetime
from  sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
# 计算分类正确率
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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
    data = data.drop(["id", "hour", "C15", "C16"], axis=1)
    return data

datapath = "/data/barnett007/ctr-data"
output = "output"
trainfile = os.path.join(datapath ,"train.csv")

df_reader = pd.read_csv(trainfile,chunksize=10000000,dtype={"C15":str,"C16":str})
gbm = None
params = {
        'task': 'train',
        'application': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': 31,
        'tree_learner': 'serial',
        'min_data_in_leaf': 100,
        'metric': ['l1','l2','binary_logloss'],  # l1:mae, l2:mse
        'max_bin': 255,
        'num_trees': 300
        #'is_unbalance' : True
    }

i=1
for df_train in df_reader:
    df_train = create_feature(df_train)
    y_all = df_train["click"]
    df_train = df_train.drop(["click"],axis=1)
    columns = df_train.columns.tolist()
    #print(columns)
    
    le = preprocessing.LabelEncoder()
    for columnname in columns:
        df_train[columnname]= le.fit_transform(df_train[columnname])  

    
    X_all = df_train.values
    
    print("split dataset")
    x_train, x_val, y_train, y_val = train_test_split(X_all, y_all, test_size = 0.2, random_state = 2018)
    lgb_train = lgb.Dataset(x_train,y_train)
    lgb_eval = lgb.Dataset(x_val,y_val)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=gbm,  # 如果不为空，就是继续上次的训练。
                    feature_name=columns,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True)  # 增量训练
    score_train = dict([(s[1],s[2]) for s in gbm.eval_train()])
    score_valid = dict([(s[1],s[2]) for s in gbm.eval_valid()])
    print('train : mae=%.4f, mse=%.4f, binary_logloss=%.4f' % (score_train['l1'], score_train['l2'], score_train['binary_logloss']))
    print('valid : mae=%.4f, mse=%.4f, binary_logloss=%.4f' % (score_valid['l1'], score_valid['l2'], score_valid['binary_logloss']))
    i+=1
