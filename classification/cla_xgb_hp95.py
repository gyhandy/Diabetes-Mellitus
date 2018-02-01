# coding=utf-8

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# load processed data from final_onehot folder
origin_train = pd.read_csv('../data_processing/final_onehot/all_train_feat.csv')
draft_param_feature = list(origin_train.columns)
draft_param_feature.remove('血糖')
draft_param_feature.remove('id')

print("all",origin_train.shape)
train = origin_train[origin_train['血糖']<=19]
highValue = np.percentile(train['血糖'],59)
# highValue = np.percentile(train['血糖'],80)
print('<'+str(19),train.shape)
print('<'+str(highValue),train[train['血糖']>=highValue].shape)


#线下 test 数据有血糖标签，可以打印检验分类topK结果
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
train_x = train.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)# drop some high vacancy param
test_id = test.id
test['血糖'] = test['血糖'].apply(lambda x: 1 if x>=highValue else 0)
test_y = test['血糖']
test_x = test.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)



# # 线上 数据无血糖标签，只能看到置信值，无法检验结果
# origin_test = pd.read_csv('../data_processing/final_onehot/all_test_feat_B.csv')
# origin_test['血糖'] = -999
# train_test = pd.concat([train,origin_test],axis=0)
# train_test.dropna(axis='columns',how='all',inplace=True)
# train = train_test[train_test['血糖']!=-999]
# test = train_test[train_test['血糖']==-999]
# #generate label for each product_no
# train['血糖'] = train['血糖'].apply(lambda x: 1 if x>=highValue else 0)
# train_y = train['血糖']
# train_x = train.drop(['血糖','id','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)
# test_id = test.id
# test_x = test.drop(['id', '血糖', '乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1)

xgb_params={
    'booster':'gbtree',
	# 'n_estimators': 247,
	'objective': 'binary:logistic',
	'eval_metric': 'auc',
	'max_depth':4,
	'reg_lambda':0.2,
	'subsample':0.7,
	'colsample_bytree':0.7,
	'learning_rate': 0.1,
	'seed':1	,
	'nthread': 16
}# change seed to get different top 10
cla_xgb_ori = XGBClassifier(**xgb_params)
cla_xgb_ori.fit(train_x, train_y)
xgb_hp95 = cla_xgb_ori.predict_proba(test_x)

xgb_hp95 = xgb_hp95[:, 1]
test_result = pd.DataFrame(test_id, columns=["id"])
# 线下
test_result["blood suger"] = test_ori_value
test_result["label"] = test_y
test_result["prob"] = xgb_hp95


xgb_hp95_prob = test_result.sort_values(by='prob', axis=0, ascending=False)
top10_prob = xgb_hp95_prob[0:11]
print(top10_prob)

xgb_hp95_prob.to_csv("./output/xgb_offline.csv", index=None, encoding='utf-8')
print('high value:  '+str(highValue))
# adjust param same as rf






















