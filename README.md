# (TianChi)Precision medical competition-Artificial Intelligence Aided genetic risk prediction of diabetes
天池精准医疗大赛——人工智能辅助糖尿病遗传风险预测

A competation in Tianchi platform about Precision medical treatment[kink](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231638)。

## Code description

### data_processing

Before classification and regression, process the raw data 

- only augment top train data，`data_augment_top.py`
- augment train data，`data_augment_all.py`
- onehot coding，`onehot.py`
- make offline dataset，`offline_data_extract.py`

#### original_data

- upload sample，`d_sample_20180102.csv`
- testdata_A，`d_test_A_20180102`
- testdata_B，`d_test_B_20180128.csv`

#### aug_data

- augmented data，`d_top_augment_5times_2.csv`
- augmented data，`d_top_20180201_130642.csv`
- augmented data，`d_top.csv`

#### final_onehot

After one hot coding waiting for being used

- train data，`all_train_feat.csv`
- test data A without label，`all_test_feat_A.csv`
- test data A with label，`all_test_feat_A_withlabel.csv`
- test data B without label，`all_test_feat_B.csv`
- augmented train data，`arg_top_.csv`
- train data - offline test，`offline_train_feat.csv`
- offline data to imitate A/B test，`offline_test_feat.csv`

### regression

Elastic linear regression

- Elastic linear regression，`elasticnet_regression.py`
- normalize function，`normalize.py`

#### output

- output of regression，`pre_results.csv`
- saved regression model，`linear_regression_modelNor.h5`


### classification

- `cla_rf.py`，training random forest classification (threhold of blood suger is top n% Median)
- `cla_xgb_hp95.py`，training XGboost classification (threhold of blood suger is top 95% Median)
- `cla_xgb_lp30.py`,training XGboost classification (threhold of blood suger is low 30% Median)
- `cla_svm.py`,training SVM classification (threhold of blood suger is top n% Median)
- `voting.py`,soft vote to get classification top-k result by combining different classifications
- `hard_voting.py`,hard vote to get classification top-k result (not good)
- `W.py`,training weight to combine different classification (hard to train)
- `W_simple.py`,training weight to combine different classification

#### output

output of single classification

##### deep_w

output of single classification to train W_simple

#### final_top_cla

##### offline
final result of combining classification to get the highest bloos suger with label (offline)

##### online
final result of combining classification to get the highest bloos suger without label (online)


## Solution

[updating...]
