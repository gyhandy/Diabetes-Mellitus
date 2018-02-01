# -*- coding:utf-8 -*-
# random forest for classification
# 1 可以调整分类的 highvalue 置信值来调整label
# 2 可以选择最佳参数best 1 2 3，通过GridSearchCV搜索得到，建议小数据集搜索，大数据集验证，提升效率。

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import h5py
from sklearn import preprocessing

# random forest不需要normalize
# from normalize import normalize

# load processed data from final_onehot folder
# origin_train = pd.read_csv('../data_processing/final_onehot/arg_top_.csv')
origin_train = pd.read_csv('../data_processing/final_onehot/all_train_feat.csv')
draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

# print classification information
print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],98)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)

# 线下 test 数据有血糖标签，可以打印检验分类topK结果
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
train_x = train.drop(['血糖','id','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'], axis=1)
test_id = test.id
test['血糖'] = test['血糖'].apply(lambda x: 1 if x>=highValue else 0)
test_y = test['血糖']
test_x = test.drop(['id','血糖','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'], axis=1)


# # 线上 数据无血糖标签，只能看到置信值，无法检验结果
# # origin_test = pd.read_csv('../data/feature/all_test_feat_B.csv')
# origin_test = pd.read_csv('../data/feature/all_test_feat_B.csv')
# # test_ori_value = origin_test['血糖']# 保存一下， 统一处理完后恢复
# origin_test['血糖'] = -999
# train_test = pd.concat([train,origin_test],axis=0)
# train_test.dropna(axis='columns',how='all',inplace=True)
# train = train_test[train_test['血糖']!=-999]
# test = train_test[train_test['血糖']==-999]
# # test['血糖'] = test_ori_value
# # generate label for each product_no
# train['血糖'] = train['血糖'].apply(lambda x: 1 if x>=highValue else 0)
# train_y = train['血糖']
# train_x = train.drop(['血糖','id','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)
# # x_columns = train_x.columns
# # X = train[x_columns]
# test_id = test.id
# # test['血糖'] = test['血糖'].apply(lambda x: 1 if x>=highValue else 0)
# # test_y = test['血糖']
# test_x = test.drop(['id','血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)

X = train_x
y = train_y


# 用非增强的5000训练数据调参得到3种最佳参数：
rf_best1 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=35, max_features='sqrt', oob_score=True, random_state=10)
rf_best1.fit(X, y)
# rf_best2= RandomForestClassifier(n_estimators=78, max_depth=15, min_samples_split=10,
#                                  min_samples_leaf=35, max_features=23, oob_score=True, random_state=10)
# rf_best2.fit(X, y)
# rf_best3= RandomForestClassifier(n_estimators=78, max_depth=15, min_samples_split=10,
#                                  min_samples_leaf=35, max_features=23, oob_score=True, random_state=10)
# rf_best3.fit(X, y)

y_pres_rf = rf_best1.predict_proba(test_x)
y_pres_rf = y_pres_rf[:, 1]
test_result = pd.DataFrame(test_id, columns=["id"])
test_result["blood suger"] = test_ori_value
test_result["label"] = test_y
test_result["prob"] = y_pres_rf
rf_prob = test_result.sort_values(by='prob', axis=0, ascending=False)
top10_prob = rf_prob[0:11]
print(top10_prob)

# 保存结果

rf_prob.to_csv("./output/rf_best1_98_offline.csv", index=None, encoding='utf-8')


# # 参数调优
# #首先对n_estimators进行网格搜索
# param_test1= {'n_estimators':range(10,71,10)}
# gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
#                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
#                        param_grid =param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X,y)
# gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_
# #这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
# param_test2= {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
# gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
#                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
#    param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(X,y)
# gsearch2.grid_scores_,gsearch2.best_params_, gsearch2.best_score_
# #再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
# param_test3= {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
# gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13,
#                                  max_features='sqrt' ,oob_score=True, random_state=10),
#    param_grid = param_test3,scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,y)
# gsearch3.grid_scores_,gsearch2.best_params_, gsearch2.best_score_
# #最后我们再对最大特征数max_features做调参:
# param_test4= {'max_features':range(3,11,2)}
# gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13, min_samples_split=120,
#                                  min_samples_leaf=20 ,oob_score=True, random_state=10),
#    param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,y)
# gsearch4.grid_scores_,gsearch4.best_params_, gsearch4.best_score_
# rf2= RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
#                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
# rf2.fit(X,y)
# printrf2.oob_score_


