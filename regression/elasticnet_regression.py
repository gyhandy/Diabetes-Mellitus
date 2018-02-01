# Elastic Net Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve elastic net regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width
# 在当时处理特异值是将有缺失值的部分删除，在预测的时候要特别取出，放在最后面。后面在分类中的做法就比较好，直接以中位数补全
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from sklearn import preprocessing
from normalize import normalize
###
# Set up for TensorFlow
###

ops.reset_default_graph()

# Create graph
sess = tf.Session()

###
# Obtain data
###

# Load the data

all_ori_data = np.genfromtxt("d_train_20180102.csv", dtype=np.float, delimiter=",")
# feature = range[-18:-1]
all_data = all_ori_data[1:, -18:-1]
# all_data = all_ori_data[1:, 12:15]
all_label = all_ori_data[1:, -1]


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

###
# Setup model
###

# Declare batch size and interations
# batch_size = 50
batch_size = 120
iterations = 30000
# Initialize placeholders
x_data = tf.placeholder(shape=[None, all_data.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
learning_rate = tf.placeholder(dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[all_data.shape[1],1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare the elastic net loss function
elastic_param1 = tf.constant(1.0)
elastic_param2 = tf.constant(1.0)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
loss_tianchi_test = tf.div( tf.reduce_mean(tf.square(y_target - model_output)), 2.0)
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

###
# Train model
###
# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)
# Training loop
testlabel = np.transpose([testlabel[:]])
loss_vec = []
loss_test_vec = []
for i in range(iterations):
    rand_index = np.random.choice(len(traindata), size=batch_size)
    rand_x = traindata[rand_index]
    # rand_y = trainlabel[rand_index]
    rand_y = np.transpose([trainlabel[rand_index]])
    # Learning rate control
    rate = 0.001
    if i > 20000 and i < 25000:
        rate = rate * 0.3
    elif i >= 25000:
        rate = rate * 0.1
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y, learning_rate: rate})
    if (i+1)%500==0:
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        tianchi_test_loss = sess.run(loss_tianchi_test, feed_dict={x_data: testdata, y_target: testlabel})
        loss_vec.append(temp_loss[0])
        loss_test_vec.append(np.array(tianchi_test_loss))
        # print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Step #' + str(i+1) + ': Loss = ' + str(temp_loss))
        print('Tianchi_test_Loss = ' + str(tianchi_test_loss))

###
# Extract model results
###

# Get the optimal coefficients
A_predict = sess.run(A)
b_predict = sess.run(b)

# Load the prediction data

pre_oridata = np.genfromtxt("d_test_A_20180102.csv", dtype=np.float, delimiter=",")
all_id = np.array(pre_oridata[1:, 0])
pre_data = pre_oridata[1:, -17:]
# pre_data = pre_data[1:, 12:16]

## processing get the abnormal one which may be stack by mean
pre_data_df = pd.DataFrame(pre_data)
abnornal_index = pre_data_df[pre_data_df.isnull().values==True].index.tolist()  # detect the index of nan
abnornal_index = list(set(abnornal_index))
nornal_index = np.array(list(set(range(pre_data_df.shape[0])) - set(abnornal_index)))
pre_data = np.delete(pre_data, abnornal_index, 0)

# ret_list = np.array(list(set(all_id)^set(abnornal_index)))
abnormal_id = np.array(sorted(all_id[abnornal_index]))
normal_id = np.array(sorted(all_id[nornal_index]))
# Standardization
normalized_pre_data = preprocessing.scale(pre_data)
# final_output = tf.add(tf.matmul(x_data, A_predict), b_predict)
normal_output = np.round(np.dot(normalized_pre_data, A_predict) + b_predict, 3)
nomal_results = np.c_[normal_id, normal_output]
abnormal_results = np.c_[abnormal_id, np.round(np.mean(normal_output), 3)*np.ones((abnormal_id.shape[0], 1))]
test_results = np.vstack((nomal_results, abnormal_results))
final_output = pd.DataFrame(test_results)
final_output.columns={'id', 'xuetang'}
final_output.to_csv('./output/pre_results.csv',index=False)
# final_output.to_csv("./output/output.csv",index=False,sep=',')

# d_test_label_predict = np.round(np.dot(normalize_dataMat, A) + bias, 3)
# pre_results = np.c_[normal_id, d_test_label_predict]
#
# abnormal_results = np.c_[abnormal_id, np.round(np.mean(d_test_label_predict), 3)*np.ones((abnormal_id.shape[0], 1))]
#
# test_results = np.vstack((pre_results, abnormal_results))

# df = pd.DataFrame(test_results)
# df.columns={'id', 'xuetang'}
# df.to_csv('pre_results.csv',index=False)
print(abnormal_id)

# Get and save the model parameter
file = h5py.File('./output/linear_regression_modelNor.h5', 'w')
file.create_dataset('weight', data=A_predict)
file.create_dataset('bias', data=b_predict)

file.create_dataset('scaler_data_max', data=scaler.data_max_)
file.create_dataset('scaler_data_min', data=scaler.data_min_)
file.create_dataset('scaler_data_scaler', data=scaler.scale_)

file.create_dataset('mean', data=mean)
file.create_dataset('variance', data=variance)
file.close()
###
# Plot results
###
# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')

plt.plot(loss_test_vec, 'r-')
plt.title('test_Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()