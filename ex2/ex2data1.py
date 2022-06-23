# 在训练的初始阶段，我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。
# 设想你是大学相关部分的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。
# 现在你拥有之前申请学生的可以用于训练逻辑回归的训练样本集。
# 对于每一个训练样本，你有他们两次测试的评分和最后是被录取的结果。

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

# 定义函数
def mean_normalization(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return np.divide(np.subtract(x, mean), std)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = -np.multiply(y, np.log(sigmoid(x * theta.T)))
    second = -np.multiply(1 - y, np.log(1 - sigmoid(x * theta.T)))
    return np.sum(first + second) / len(x)

def gradient(theta, x, y):
    (m, n) = x.shape
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    paramters = int(theta.ravel().shape[1])
    grad = np.zeros(paramters)
    cost = sigmoid(x * theta.T) - y

    for i in range(paramters):
        term = np.multiply(cost, x[:, i])
        grad[i] = np.sum(term) / m
    return grad

def predict(theta, x):
    probability = sigmoid(np.dot(x, theta))
    return probability

# 加载数据
path = './data_sets/ex2data1.txt'
data = np.loadtxt(path, delimiter=',')
data = np.array(data)
# x = mean_normalization(data[:,:2])
x = data[:,:2]
x = np.c_[np.ones(x.shape[0]), x]
y = data[:, -1:]
theta = np.zeros(x.shape[1])
print('x.shape={}, y.shape={}, theta.shape={}'.format(x.shape, y.shape, theta.shape))
# 梯度下降
print('the cost when begin: {}'.format(cost(theta, x, y)))
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
print(result)
print('the cost when end: {}'.format(cost(result[0], x, y)))
# 绘制图象
positive = [[d[0], d[1]] for d in data if d[2] == 1]
negative = [[d[0], d[1]] for d in data if d[2] == 0]
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]

plt.title('Admitting and Score graph')
plt.xlabel('exam1 score')
plt.ylabel('exam2 score')
plt.scatter(np.array(positive)[:, :1],np.array(positive)[:, -1:], marker='.', c='blue')
plt.scatter(np.array(negative)[:, :1],np.array(negative)[:, -1:], marker='x', c='red')
plt.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
plt.show()

# ===================  评价逻辑回归模型  =========================
def hfunc(theta, x):
    return sigmoid(np.dot(theta.T, x))
print('hfunc1(result[0],[1,45,85])={}'.format(hfunc(result[0],[1,45,85])))
