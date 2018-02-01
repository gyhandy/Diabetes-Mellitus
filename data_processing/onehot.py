#coding=utf-8
# preprocessing for the augmented data and test data(coding 体检日期&性别，消除nan数据)
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool
from dateutil.parser import parse

# os.mkdir('./data/feature')

############     draft表 类别特征编码   ##############
######################################################
train = pd.read_csv('./aug_data/d_top.csv', encoding='utf-8')# process the augmented data
test = pd.read_csv('./original_data/d_test_A_20180102.csv', encoding='gbk')
test['血糖'] = -999
draft_train_test = pd.concat([train, test], axis=0)
# category_var = list(train.columns)
category_var = ['性别']# 只对性别进行one hot 编码

# one-hot编码
draft_train_test['性别'] = draft_train_test['性别'].map({'男': 1, '女': 0})# 先做替换
draft_train_test['体检日期'] = (pd.to_datetime(draft_train_test['体检日期']) - parse('2017-10-09')).dt.days
draft_train_test.fillna(draft_train_test.median(axis=0), inplace=True)

# add the dimension of kinds variables
for var in category_var:
    var_dummies = pd.get_dummies(draft_train_test[var])
    var_dummies.columns = [var+'_'+str(i) for i in range(var_dummies.shape[1])]
    if var in ['性别']:# delete some orign variables
        draft_train_test.drop(var, axis=1, inplace=True)
    draft_train_test = pd.concat([draft_train_test,var_dummies],axis=1)

draft_train = draft_train_test[draft_train_test['血糖']!=-999]
draft_test = draft_train_test[draft_train_test['血糖']==-999]
draft_test.drop('血糖', axis=1, inplace=True)

# choose the 95 and 30 percentile in train data
# keyindex95 = np.percentile(draft_train['血糖'],95)
# keyindex30 = np.percentile(draft_train['血糖'],30)
draft_train.to_csv('./final_onehot/arg_top_.csv', index=None)
draft_test.to_csv('./final_onehot/all_test_feat.csv', index=None)