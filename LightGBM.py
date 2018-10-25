import lightgbm as lgb
import pandas as pd
import datetime

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
    #data["time_period"] = data["hour1"].map(lambda x:time_period(x))
    data["app_site_id"] = data["app_id"] + "_" + data["site_id"]
    data["app_site_id_model"] = data["app_site_id"] + "_" + data["device_model"]
    #此处可以考虑将组合特征的源特征删掉，对比效果
    data = data.drop(["id", "hour"], axis=1)
    return data


gbm=None

# 第二步，流式讀取數據(每次100萬)
i=1
file = "/data/barnett007/ctr-data/train.csv"
for sub_data in pd.read_csv(file, chunksize=1000000,dtype={"C15":str,"C16":str}):
    sub_data = create_feature(sub_data)
    x_cols = [c for c in sub_data.columns if c not in ["click"]]
    sub_data = sub_data.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(0.1 * sub_data.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]

    # 區分特徵x和結果Y
    x_data = df_train[x_cols]
    y_data = df_train["click"]

    # 創建lgb的數據集
    lgb_train = lgb.Dataset(x_data, y_data.values)
    lgb_eval = lgb.Dataset(df_test[x_cols], df_test["click"].values, reference=lgb_train)

    # 第三步：增量訓練模型
    # 重點來了，通過 init_model 和 keep_training_booster 兩個參數實現增量訓練
    params = {
        'task': 'train',
        'application': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'tree_learner': 'serial',
        'min_data_in_leaf': 100,
        'metric': ['l1','l2','binary_logloss'],  # l1:mae, l2:mse
        'max_bin': 255,
        'num_trees': 300,
        'categorical_feature' : x_cols
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=gbm,             # 如果gbm不爲None，那麼就是在上次的基礎上接着訓練
                    feature_name=x_cols,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True) # 增量訓練 

    # 輸出模型評估分數
    score_train = dict([(s[1], s[2]) for s in gbm.eval_train()])
    score_valid = dict([(s[1], s[2]) for s in gbm.eval_valid()])
    print('當前模型在訓練集的得分是：mae=%.4f, mse=%.4f, binary_logloss=%.4f'%(score_train['l1'], score_train['l2'], score_train['binary_logloss']))
    print('當前模型在測試集的得分是：mae=%.4f, mse=%.4f, binary_logloss=%.4f' % (score_valid['l1'], score_valid['l2'], score_valid['binary_logloss']))
    i += 1
