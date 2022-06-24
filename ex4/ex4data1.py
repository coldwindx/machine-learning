# 5000张20*20像素的手写数字数据集，以及对应的数字（1-9，0对应10）
# 使用反向传播的前馈神经网络，自动学习神经网络的参数。

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report

data = loadmat('./data_sets/ex4data1.mat')
x_data, y_data = data['X'], data['y']
print('x.shape = {}, y.shape = {}'.format(x_data.shape, y_data.shape))

weights = loadmat('./data_sets/ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
print('theta1.shape = {}, theta2.shape = {}'.format(theta1.shape, theta2.shape))

# ===================== 函数 ==========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_g1(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward(x, theta1, theta2):
    (m, n) = x.shape
    # a1 ---> z2 --sigmoid--> a2 ---> z3 --sigmoid--> h
    a1 = np.insert(x, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def backprop(params, input_size, hidden_size, labels, x, y, lam):
    (m, n) = x.shape
    x = np.matrix(x)
    y = np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (labels, hidden_size + 1)))

    a1, z2, a2, z3, h = forward(x, theta1, theta2)
        
    J = 0
    for i in range(m):
        first = -np.multiply(y[i,:], np.log(h[i,:]))
        second = -np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first + second)
    J /= m
    # 加入正则化
    J += lam / (2 * m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for t in range(m):
        a1t = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        z3t = z3[t,:]
        ht = h[t,:]
        yt = y[t,:]

        d3t = ht - yt
        delta2 = delta2 + d3t.T * a2t

        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_g1(z2t))
        delta1 = delta1 + d2t[:,1:].T * a1t
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * lam) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * lam) / m
    
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

def cost(theta1, theta2 , input_size, hidden_size, labels, x, y, lam):
    (m, n) = x.shape
    x = np.matrix(x)
    y = np.matrix(y)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)

    # 前向传播
    a1, z2, a2, z3, h = forward(x, theta1, theta2)

    J = 0
    for i in range(m):
        first = -np.multiply(y[i,:], np.log(h[i,:]))
        second = -np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first + second)
    J /= m
    # 加入正则化
    J += lam / (2 * m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    return J

# 标签转为独热码
y_onehot = OneHotEncoder(sparse=False).fit_transform(y_data)
# 初始化设置
input_size = 400
hidden_size = 25
labels = 10
lam = 1
print(cost(theta1, theta2, input_size, hidden_size, labels, x_data, y_onehot, lam))
# 随机初始值
params = (np.random.random(hidden_size * (input_size + 1) + (hidden_size + 1) * labels) - 0.5) * 0.24

fmin = minimize(fun=backprop, x0=(params),
                args=(input_size, hidden_size, labels, x_data, y_onehot, lam), 
                method='TNC', jac=True, options={'maxiter': 250})
x = np.matrix(x_data)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward(x, thetafinal1, thetafinal2 )
y_pred = np.array(np.argmax(h, axis=1) + 1)
print(classification_report(y_data, y_pred))