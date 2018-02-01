# -*- coding:utf-8 -*-
# 1 depend on the top-k of each single classification, use tensorflow GradientDescentOptimizer adjust the best weighr W
# 2 loss function is the accuracy of top-K by Cross Entropy
# 简单理解，先根据每个分类器暴力融合的结果找到top20，并把top20的id找到，以top-20 id为模板找到他们在每个分类器里面的置信值，
# 组成 20×n_cla 的训练数据，通过优化这些置信值的组合W，使得这top20 的交叉熵最低
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from sklearn import preprocessing

xgb_1028 = pd.read_csv('./output/deep_w/xgb_1028.csv')
xgb_1010 = pd.read_csv('./output/deep_w/xgb_1010.csv')
xgb_1025 = pd.read_csv('./output/deep_w/xgb_1025.csv')
xgb_1025_500 = pd.read_csv('./output/deep_w/xgb_1025_500.csv')
xgb_1025_200 = pd.read_csv('./output/deep_w/xgb_1025_200.csv')
rf_best1 = pd.read_csv('./output/deep_w/rf_best1.csv')
rf_best2 = pd.read_csv('./output/deep_w/rf_best2.csv')
rf_best3 = pd.read_csv('./output/deep_w/rf_best3.csv')

# select the top 20 and boosting
cla_list = [xgb_1028, xgb_1010, xgb_1025, xgb_1025_500, xgb_1025_200, rf_best1, rf_best2, rf_best3]
top_20 = list()
pre_data = pd.DataFrame()
pre_id = pd.DataFrame()
# top_20 = pd.DataFrame()
for cla in cla_list:
    cla_top_20_id = cla['id'][0:20]
    cla_top_20_val = cla['prob'][0:20]
    pre_id = pd.concat([pre_id, cla_top_20_id], axis=1, join_axes=[cla_top_20_id.index])
    pre_data = pd.concat([pre_data, cla_top_20_val], axis=1, join_axes=[cla_top_20_val.index])
    cla_top_20 = pd.concat([cla_top_20_id, cla_top_20_val], axis=1)
    top_20.append(cla_top_20)
predata = np.array(pre_data)
preid = np.array(pre_id)

# preprocessing Minmax
min_max_scaler = preprocessing.MinMaxScaler()
predata_nor = min_max_scaler.fit_transform(predata)
# predata_nor = predata_nor/len(cla_list)




# loss

# find the uniqe id of top20 and fix the top20
out_data = np.transpose(predata_nor).reshape(-1, 1)
out_id = np.transpose(preid).reshape(-1, 1)
only_id = list(set(out_id.flatten().tolist()))
# only_id = set(list(chain.from_iterable(out_id.tolist())))
rank_id = []
rank_sum = []
for i in range(len(only_id)):
    index = np.argwhere(out_id == only_id[i])[:, 0]# find all index in all classification model for each onlyid
    sum = out_data[index].sum()
    rank_id.append(only_id[i])
    rank_sum.append(sum)
rank_data = pd.DataFrame(rank_id, columns=["id"])
rank_data["rank_sum"] = rank_sum
rank_data = rank_data.sort_values(by='rank_sum', axis=0, ascending=False)
top20 = rank_data[0:20]

# find traindata & trainlabel, from each classification get the fixed top20(even some of id are low in the rank of single cla)
train_data = []
for cla in cla_list:
    cla_id = np.array(cla["id"])
    cla_prob = np.array(cla["prob"])
    cla_label = np.array(cla["label"])
    top20_id = list(top20["id"])
    top20_prob = []
    top20_label = []
    for i in range(len(top20_id)):
        index = np.argwhere(cla_id == top20_id[i])
        prob = float(cla_prob[index])
        label = int(cla_label[index])
        top20_prob.append(prob)
        top20_label.append(label)
    train_data.append(top20_prob)

train_data = np.transpose(np.array(train_data))
train_label = np.array(top20_label).reshape(-1, 1)
train_data = min_max_scaler.fit_transform(train_data)

ops.reset_default_graph()
sess = tf.Session()
# set variable and Forward
learning_rate = tf.placeholder(dtype=tf.float32)
W1 = tf.Variable(tf.random_normal(shape=[1, 8]))
W = tf.div(W1, tf.reduce_sum(W1))
x_data = tf.placeholder(shape=[20, 8], dtype=tf.float32)
y_target = tf.placeholder(shape=[20, 1], dtype=tf.float32)
model_output = tf.multiply(x_data, W)
y_ = tf.reshape(tf.reduce_sum(model_output,axis=1), [-1, 1], name=None)
# loss = cross_entropy
loss = -tf.reduce_sum( tf.add(y_target*tf.log(y_), (1-y_target)*tf.log(1-y_)), reduction_indices= 0)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)


iterations = 5000
init = tf.global_variables_initializer()
sess.run(init)


loss_vec = []
train_loss_vec = []
test_loss_vec = []
for i in range(iterations):
    # Learning rate control
    rate = 0.01
    if i >= 2000: rate = 0.005
    if i >= 4000: rate = 0.001
    sess.run(train_step, feed_dict={x_data: train_data, y_target: train_label, learning_rate: rate})
    if (i+1)%100 == 0:
        temp_loss = sess.run(loss, feed_dict={x_data: train_data, y_target: train_label})
        train_loss_vec.append(temp_loss)
        print('Step #' + str(i+1) + ': Loss = ' + str(temp_loss))
        # test_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
        # test_loss_vec.append(test_loss)
        # print('Step #' + str(i + 1) + ': Test Loss = ' + str(test_loss))

# Get and save the model parameter
file = h5py.File('modelW.h5', 'w')
file.create_dataset('weight', data=sess.run(W))
file.close()
plt.plot(train_loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss_vec, 'r-', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
print(sess.run(W))
