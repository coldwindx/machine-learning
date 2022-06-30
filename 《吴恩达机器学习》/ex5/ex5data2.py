# 对一个水库的流出水量以及水库水位进行正则化线性归回

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

data = loadmat('./data_sets/ex5data1.mat')
x_data, y_data, x_val, y_val, x_test, y_test = data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']

# 扩展特征数量
def poly_features(x, degree):
    ans = x.copy()
    for i in range(2, degree + 1):
        ans = np.append(ans, np.power(x, i), axis=1)
    return ans

# 处理输入数据
def mean_normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return np.divide(np.subtract(x, mean), std)
def prepare_poly_data(*args,degree):
    def prepare(x):
        x = poly_features(x, degree)
        x = mean_normalize(x)
        x = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
        return x
    return [prepare(x) for x in args]

x, x_val, x_test = prepare_poly_data(x_data, x_val, x_test, degree=8)
theta = np.ones(x.shape[1])
# print(x)

def cost(theta, x, y, lam = 1):
    (m, n) = x.shape
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)

    h = x * theta.T - y
    J = np.sum(h.T * h) / (2 * m)

    reg = lam / (2 * m) * np.sum(np.power(theta[:,1:], 2))
    return J + reg

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

fmin = minimize(fun=cost, x0 = theta, args = (x, y_data, 1),
                method='TNC', jac=gradient, options={'disp': True})
# print(fmin.x)

fitx = np.linspace(-50, 50, 100)
fitx = np.reshape(fitx, (100, 1))
fity = np.dot(prepare_poly_data(fitx, degree=8), fmin.x.T).T
plt.plot(fitx, fity, c='r', label='fitcurve')
plt.scatter(x_data, y_data, c='b', label='initial_Xy')
plt.xlabel('water level')
plt.ylabel('flow')
plt.legend()
plt.show()

# 画出学习曲线来判断方差和偏差的问题。
(m, n) = x.shape
costs_train, costs_val = [], []
for i in range(1, m + 1):
    theta = np.ones(n)
    ret = minimize(fun=cost, x0 = theta, args = (x[:i, :], y_data[:i, :], 0),
                method='TNC', jac=gradient, options={'disp': False})

    J_train = cost(ret.x, x[:i,:], y_data[:i,:], 0)
    costs_train.append(J_train)

    J_val = cost(ret.x, x_val, y_val, 0)
    costs_val.append(J_val)


plt.xlabel('data sets size')
plt.plot(np.arange(1, m + 1), costs_train, label='J_train')
plt.plot(np.arange(1, m + 1), costs_val, label='J_validation')
plt.legend()
plt.show()
        
#  查找最优lambda
lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
(m, n) = x.shape
costs_train, costs_val = [], []
for l in lams:
    theta = np.ones(n)
    ret = minimize(fun=cost, x0 = theta, args = (x, y_data, l),
                method='TNC', jac=gradient, options={'disp': False})

    J_train = cost(ret.x, x, y_data, l)
    costs_train.append(J_train)

    J_val = cost(ret.x, x_val, y_val, l)
    costs_val.append(J_val)
plt.xlabel('lambda')
plt.plot(lams, costs_train, label='J_train')
plt.plot(lams, costs_val, label='J_validation')
plt.legend()
plt.show()

for l in lams:
    theta = np.ones(n)
    ret = minimize(fun=cost, x0 = theta, args = (x, y_data, l),
                method='TNC', jac=gradient, options={'disp': False})
    print('test cost(l={}) = {}'.format(l, cost(theta, x_test, y_test, l)))