# -*- coding:utf-8 -*-
# 导入需要的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import h5py
from tensorflow.python.framework import ops
from sklearn import preprocessing
from normalize import normalize

# 导入数据
all_ori_data = np.genfromtxt("d_train_20180102.csv", dtype=np.float, delimiter=",")
# feature = range[-18:-1]
all_data = all_ori_data[1:, -18:-1]
# all_data = all_ori_data[1:, 12:15]
all_label = all_ori_data[1:, -1]

# 换标签
clas_label = []
threhold = 9
for label in all_label:
    clas = 0 if label <= threhold else 1
    clas_label.append(clas)
all_label = np.array(clas_label)

# 数据的类别分布
num_1 = list(all_label).count(1)
print("There are %d positive in total %d." %(num_1, len(all_label)))

## preprocessing
all_data_df = pd.DataFrame(all_data)
dele_index = all_data_df[all_data_df.isnull().values==True].index.tolist()  # detect the index of nan
dele_index = list(set(dele_index))
all_data = np.delete(all_data, dele_index, 0)
all_label = np.delete(all_label, dele_index, 0)
# Standardization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
all_data = scaler.fit_transform(all_data)
# all_data = preprocessing.scale(all_data)
all_data, mean, variance = normalize(all_data)

train_indices = np.random.choice(len(all_data),
                                 round(len(all_data)*0.8),
                                 replace=False)
test_indices = np.array(list(set(range(len(all_data))) - set(train_indices)))
traindata = all_data[train_indices]
trainlabel = all_label[train_indices]
testdata = all_data[test_indices]
testlabel = all_label[test_indices]


# 接着选择好样本特征和类别输出，样本特征为除去ID和输出类别的列

X = traindata
y = trainlabel

# 不管任何参数，都用默认的，拟合下数据看看
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X, y)
y_pres_rf = rf0.predict_proba(testdata)


# 调节参数进行分类
class_threhold = 0.4
final_class = []
for output in y_pres_rf:
    pre_class = 1 if output[1]>=0.4 else 0
    final_class.append(pre_class)
# 找到所在位置及编号

aim_class=1
num_aim=final_class.count(aim_class)  #aim_class在final_class中的个数
aim_position = [i for i in range(len(final_class)) if final_class[i] == aim_class]  #aim_class在final_class中的下标
# 统计正确率
posi_rig_num = 0
for j in aim_position:
    if final_class[j] == testlabel[j]:
        posi_rig_num += 1
aim_acc= float(posi_rig_num)/len(aim_position)
print(aim_acc)

right_num = 0
for i in range(len(final_class)):
    if final_class[i] == testlabel[i]:
        right_num += 1
    # else: print(i)
acc= float(right_num)/len(final_class)
print(acc)



print rf0.oob_score_
y_predprob = rf0.predict_proba(X)[:, 1]
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)




# 输出如下：0.98005  AUC Score (Train): 0.999833
# 可见袋外分数已经很高（理解为袋外数据作为验证集时的准确率，也就是模型的泛化能力），而且AUC分数也很高
# （AUC是指从一堆样本中随机抽一个，抽到正样本的概率比抽到负样本的概率 大的可能性）
# 。相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。

# 首先对n_estimators进行网格搜索
param_test1 = {'n_estimators': range(10, 71, 10)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                         min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                         random_state=10),
                        param_grid=param_test1, scoring='roc_auc', cv=5)
gsearch1.fit(X, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

#这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2= {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
   param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.grid_scores_,gsearch2.best_params_, gsearch2.best_score_
#输出如下：

# 已经取了三个最优参数，看看现在模型的袋外分数：
rf1 = RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=110,
                             min_samples_leaf=20, max_features='sqrt', oob_score=True, random_state=10)
rf1.fit(X, y)
print rf1.oob_score_
# 输出结果为：0.984
# 可见此时我们的袋外分数有一定的提高。也就是时候模型的泛化能力增强了。对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。

# 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60, max_depth=13,
                                                         max_features='sqrt', oob_score=True, random_state=10),
                        param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
gsearch3.fit(X, y)
gsearch3.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
# 输出如下：

#最后我们再对最大特征数max_features做调参:
param_test4= {'max_features':range(3,11,2)}
gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13, min_samples_split=120,
                                 min_samples_leaf=20 ,oob_score=True, random_state=10),
   param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_,gsearch4.best_params_, gsearch4.best_score_
#输出如下：

#用我们搜索到的最佳参数，我们再看看最终的模型拟合：
rf2= RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                 min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
rf2.fit(X, y)
print rf2.oob_score_
#此时的输出为：0.984
#可见此时模型的袋外分数基本没有提高，主要原因是0.984已经是一个很高的袋外分数了，如果想进一步需要提高模型的泛化能力，我们需要更多的数据。