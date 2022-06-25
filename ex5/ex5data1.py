# 对一个水库的流出水量以及水库水位进行正则化线性归回
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

data = loadmat('./data_sets/ex5data1.mat')
x_data, y_data, x_val, y_val, x_test, y_test = data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']


x = np.c_[np.ones(x_data.shape[0]), x_data]
y = np.array(y_data)
theta = np.ones(x.shape[1])
# print(theta)

def cost(theta, x, y, lam = 1):
    (m, n) = x.shape
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)

    h = x * theta.T - y
    J = np.sum(h.T * h) / (2 * m)

    reg = lam / (2 * m) * np.sum(np.power(theta[:,1:], 2))
    return J + reg
# print(cost(theta, x, y))

def gradient(theta, x, y, lam = 1):
    (m, n) = x.shape
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)

    h = x * theta.T - y
    grad = (h.T * x) / m

    reg = (lam / m) * theta
    reg[:,0] = 0
    return grad + reg
# print(gradient(theta, x, y))

fmin = minimize(fun=cost, x0 = theta, args = (x, y, 0),
                method='TNC', jac=gradient, options={'disp': True})
print(fmin.x)

plt.xlabel('water level')
plt.ylabel('flow')
plt.scatter(x_data, y_data, c='b', marker='o', label = 'training data')
plt.plot(x_data, x_data * fmin.x[1] + fmin.x[0], c='r', label = 'prediction')
plt.legend()
plt.show()

# 画出学习曲线来判断方差和偏差的问题。
(m, n) = x.shape
x_val = np.c_[np.ones(len(x_val)), x_val]
costs_train, costs_val = [], []
for i in range(1, m + 1):
    theta = np.ones(n)
    ret = minimize(fun=cost, x0 = theta, args = (x[:i, :], y[:i, :], 0),
                method='TNC', jac=gradient, options={'disp': True})

    J_train = cost(ret.x, x[:i,:], y[:i,:])
    costs_train.append(J_train)

    J_val = cost(ret.x, x_val, y_val)
    costs_val.append(J_val)


plt.xlabel('data sets size')
plt.plot(np.arange(1, m + 1), costs_train, label='J_train')
plt.plot(np.arange(1, m + 1), costs_val, label='J_validation')
plt.legend()
plt.show()