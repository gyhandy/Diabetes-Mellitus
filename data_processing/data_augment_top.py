#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 1.augment the k_fold top traindata n times
# 2.argment the whole data n1 times
# 若出现数据报错注意编码问题,保存的时候encoding 要和后面读取的时候encoding保持一致


import pandas as pd
import numpy as np
import random
import datetime


def augment_train_data_top(data, k, n):
    """放大血糖k折top的数据n倍
    """
    data = data.sort_values(by=[u"血糖"], ascending=False)
    data_top = data.iloc[0: (data.shape[0] // k), :] # 切片，找到血糖K折后top的数据
    arg_data = pd.DataFrame()
    for i in range(n):
        arg_data = pd.concat([arg_data, data_top], axis=0, ignore_index=True)
    return pd.concat([arg_data, data], axis=0, ignore_index=True)

data = pd.read_csv("./original_data/d_train_20180102.csv", index_col=False, encoding='GBK')
data = data.sort_values(by=[u"血糖"])
data = augment_train_data_top(data, 10, 2)
data = data.sample(frac=1, random_state=618).reset_index(drop=True)  # 随机打乱数据


predictors = [f for f in data.columns if f not in ['血糖', 'id', '性别', "年龄", "体检日期", "乙肝表面抗原",
                                                    "乙肝表面抗体", "乙肝e抗原", "乙肝e抗体", "乙肝核心抗体"]]
# “乙肝...核心” 这五维特征缺失比较严重，在训练模型中均删除，故不考虑处理

new_data = pd.DataFrame(data=None, columns=data.columns)
for i in range(2):  # 5代表放大倍数, argment the whole data n1 times
    print("第{}次加噪处理...".format(i))
    index_list = random.sample(range(data.shape[0]), data.shape[0])
    for j, index in enumerate(index_list):
        print("处理第{}个数...".format(j))
        create_data = data.loc[index]

        cur_random = random.randint(0, 9)  # 产生随机数，用于选择特征处理方法,比例可以调节要修改的特征模式
        cur_len = random.randint(1, 12)  # 选择特征的个数

        select_feature_list = random.sample(range(len(predictors)), cur_len)

        # 随机选择剔除特征或修改特征值
        if cur_random <= 4:
            for feature_index in select_feature_list:
                feature = predictors[feature_index]
                create_data[feature] = np.nan
        elif cur_random <= 8:
            for feature_index in select_feature_list:
                feature = predictors[feature_index]
                if create_data[feature] == np.nan:
                    create_data[feature] = random.random(data[feature].mean() - data[feature].std(),
                                                         data[feature].mean() + data[feature].std())
                else:
                    create_data[feature] *= random.randint(93, 107) * 0.01
        else:
            print("保留原值")
        new_data = new_data.append(create_data)

new_data = new_data.sample(frac=1, random_state=618).reset_index(drop=True)  # 随机打乱数据
# 保存的时候encoding 要和后面读取的时候保持一致
# new_data.to_csv(r'd_top_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, encoding='utf-8')
new_data.to_csv(r'./aug_data/d_top_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, encoding='utf-8')