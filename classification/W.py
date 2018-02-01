# -*- coding:utf-8 -*-
# This is a bad attempt, To adjust the W, we must refresh the top-k rank, but the select method can not be derivative
# We adjust the method into W_simple.py
# 原始思路：（1） 根据每个分类器得到每个分类器的Top-K （2） 根据每个分类器的top-k × Wi（每个分类器相应的权值）投出总的Top-K 榜
# （3）根据Top-k榜 的交叉熵来更新每个分类器的权值Wi，因为更新Wi的时候要看投出总的Top-K 榜中的id，每一个id可能在不同的分类器中出现，所以每更新一次
# 总榜都会变，所以交叉熵的标签就会变，所以中间的排序过程无法用梯度下降法自动学习。
# 简化为W-simple，固定总榜Top-20
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

ops.reset_default_graph()
sess = tf.Session()


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
    pre_id = pd.concat([pre_id, cla_top_20_id], axis=1, join_axes=[cla_top_20_val.index])
    pre_data = pd.concat([pre_data, cla_top_20_val], axis=1, join_axes=[cla_top_20_val.index])
    cla_top_20 = pd.concat([cla_top_20_id, cla_top_20_val], axis=1)
    top_20.append(cla_top_20)
predata = np.array(pre_data)
preid = np.array(pre_id)
# set variable and Forward
W1 = tf.Variable(tf.random_normal(shape=[1, 8]))
W = tf.div(W1, tf.reduce_sum(W1))
x_data = tf.placeholder(shape=[20, 8], dtype=tf.float32)
y_target = tf.placeholder(shape=[10, 1], dtype=tf.float32)
model_output = tf.multiply(x_data, W)

learning_rate = tf.placeholder(dtype=tf.float32)
iterations = 5000
# loss = cross_entropy
# y_ = tf.placeholder(shape=[10, 1], dtype=tf.float32)
y = tf.placeholder(shape=[10, 1], dtype=tf.float32)

loss = -tf.reduce_sum( tf.add(y_*tf.log(y), (1-y_)*tf.log(1-y)), reduction_indices= 1)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)


loss_vec = []
train_loss_vec = []
test_loss_vec = []
for i in range(iterations):
    sess.run(model_output, feed_dict={x_data: predata, learning_rate: rate})
    # loss
    out_data = np.transpose(model_output).reshape(-1, 1)
    out_id = np.transpose(preid).reshape(-1, 1)
    only_id = list(set(out_id.flatten().tolist()))
    # only_id = set(list(chain.from_iterable(out_id.tolist())))
    rank_id = []
    rank_sum = []
    for i in range(len(only_id)):
        index = np.argwhere(out_id == only_id[i])[:, 0]
        sum = out_data[index].sum()
        rank_id.append(only_id[i])
        rank_sum.append(sum)
    rank_data = pd.DataFrame(rank_id, columns=["id"])
    rank_data["rank_sum"] = rank_sum
    rank_data = rank_data.sort_values(by='rank_sum', axis=0, ascending=False)
    top10 = rank_data[0:10]

    # find label
    all_id = np.array(xgb_1028["id"])
    all_label = np.array(xgb_1028["label"])
    top10_id = list(top10["id"])
    top10_label = []
    for i in range(len(top10_id)):
        index = np.argwhere(all_id == top10_id[i])
        label = int(all_label[index])
        top10_label.append(label)

    y_ = list(top10["rank_sum"])
    y = top10_label

    # Learning rate control
    rate = 0.01

    sess.run(train_step, feed_dict={x_data: predata, y_target: y, learning_rate: rate})
    if (i+1)%100 == 0:
        temp_loss = sess.run(loss, feed_dict={x_data: predata, y_target: y})
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
