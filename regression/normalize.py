import numpy as np
import matplotlib.pyplot as plt

def normalize(dataMat, axis=0):
    meanArray = np.mean(dataMat, axis=axis)
    centerArray = dataMat - meanArray
    variance = centerArray.var(axis=axis)
    normalize_dataMat = centerArray/variance
    return normalize_dataMat,meanArray,variance

# a = np.array([[0.5,1.0,1.0,1.5],[1.05,1.1,0.91,0.95]]).T
# d_train = np.genfromtxt("../data/d_train_20180102.csv", dtype=np.float, delimiter=",")[1:][:]
#
# # split data and label
# data_col = [2,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41] # add gender later
# d_data_origin = d_train[:, data_col]
# not_nan_col = np.logical_not(np.isnan(d_data_origin[:,10]))
# d_data = d_data_origin[not_nan_col,0:-1] # clear nan data
# d_label = d_data_origin[not_nan_col, -1]
#
# # Standardize data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# d_data = scaler.fit_transform(d_data)
#
# a = d_data

# plt.figure()
# plt.subplot(121)
# plt.scatter(a[:,-2],a[:,-1])
# ax = plt.gca()
# ax.set_aspect(1)
# normalize_dataMat = normalize(a)
# plt.subplot(122)
# plt.scatter(normalize_dataMat[:,-2],normalize_dataMat[:,-1])
# ax = plt.gca()
# ax.set_aspect(1)
# plt.show()

