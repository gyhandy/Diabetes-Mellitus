# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
#loading feature file and merge
origin_train = pd.read_csv('../data_processing/final_onehot/all_train_feat.csv')
draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],95)
# highValue = np.percentile(train['血糖'],80)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)


# 线下
origin_test = pd.read_csv('../data_processing/final_onehot/all_test_feat_A_withlabel.csv')
test_ori_value = origin_test['血糖']# 保存一下， 统一处理完后恢复
origin_test['血糖'] = -999
train_test = pd.concat([train,origin_test],axis=0)
train_test.dropna(axis='columns',how='all',inplace=True)
train = train_test[train_test['血糖']!=-999]
test = train_test[train_test['血糖']==-999]
test['血糖'] = test_ori_value
# generate label for each product_no
train['血糖'] = train['血糖'].apply(lambda x: 1 if x>=highValue else 0)
train_y = train['血糖']
train_x = train.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'], axis=1)# drop some high vacancy param
test_id = test.id
test['血糖'] = test['血糖'].apply(lambda x: 1 if x>=highValue else 0)
test_y = test['血糖']
test_x = test.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'], axis=1)



# # 线上
# origin_test = pd.read_csv('../data_processing/final_onehot/all_test_feat_B.csv')
# # test_ori_value = origin_test['血糖']# 保存一下， 统一处理完后恢复
# origin_test['血糖'] = -999
# train_test = pd.concat([train,origin_test],axis=0)
# train_test.dropna(axis='columns',how='all',inplace=True)
# train = train_test[train_test['血糖']!=-999]
# test = train_test[train_test['血糖']==-999]
# # test['血糖'] = test_ori_value
# #generate label for each product_no
# train['血糖'] = train['血糖'].apply(lambda x: 1 if x>=highValue else 0)
# train_y = train['血糖']
# train_x = train.drop(['血糖','id','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)
# # x_columns = train_x.columns
# # X = train[x_columns]
# test_id = test.id
# # test['血糖'] = test['血糖'].apply(lambda x: 1 if x>=highValue else 0)
# # test_y = test['血糖']
# test_x = test.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)



# Gaussian RBF
Svm_rbf = Pipeline((
    ("scaler", StandardScaler()),
    ("rbf_ovr", SVC(kernel="rbf", C=1, gamma=0.02,probability=True)),
))
Svm_rbf.fit(train_x, train_y)
Svm_rbf_hp = Svm_rbf.predict_proba(test_x)

Svm_rbf_hp = Svm_rbf_hp[:, 1]
test_result = pd.DataFrame(test_id, columns=["id"])

# 线下
test_result["blood suger"] = test_ori_value
test_result["label"] = test_y
test_result["prob"] = Svm_rbf_hp

final_prob = test_result.sort_values(by='prob', axis=0, ascending=False)
top10_prob = final_prob[0:11]
print(top10_prob)

final_prob.to_csv("./output/svm_offline.csv", index=None, encoding='utf-8')
# xgb_blood = test_result.sort_values(by='blood suger', axis=0, ascending=False)
# xgb_blood.to_csv("xgb_blood.csv", index=None, encoding='utf-8')

























