# 实现一个可以识别手写数字的神经网络。
# 权重已经预先训练好，你的目标是在现有权重基础上，实现前馈神经网络。

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report
# ========================== 加载数据 ===============================
weights = loadmat('./data_sets/ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
print('theta1.shape = {}, theta2.shape = {}'.format(theta1.shape, theta2.shape))

data = loadmat('./data_sets/ex3data1.mat')
x_data, y_data = data['X'], data['y']
x = np.c_[np.ones(len(x_data)), x_data]
y = np.array(y_data)
print('x.shape = {}, y.shape = {}'.format(x.shape, y.shape))

# ========================== 转矩阵 ===============================
x = np.matrix(x)
theta1 = np.matrix(theta1)
theta2 = np.matrix(theta2)

# ========================== 前向传播 ===============================
# x--->z--->o
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
z = sigmoid(x * theta1.T)
print('z.shape = {}'.format(z.shape))
z = np.c_[np.ones(z.shape[0]), z]
o = sigmoid(z * theta2.T)
print(o)

y_pred = np.argmax(o, axis=1) + 1
print(classification_report(y, y_pred))