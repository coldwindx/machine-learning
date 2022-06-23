# 设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果，测试结果决定是否芯片要被接受或抛弃。
# 你有一些历史数据，帮助你构建一个逻辑回归模型。


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize as opt


# 定义函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta, x, y, lam):
    m = x.shape[0]
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    first = -np.multiply(y, np.log(sigmoid(x * theta.T)))
    second = -np.multiply(1 - y, 1 - np.log(sigmoid(x * theta.T)))
    reg = lam / (2 * m) * np.sum(np.power(theta[:, 1:], 2))
    return np.sum(first + second) / m + reg

def gradient(theta, x, y, lam):
    (m, n) = x.shape
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    paramters = int(theta.ravel().shape[1])
    grads = np.zeros(paramters)

    cost = sigmoid(x * theta.T) - y
    for i in range(paramters):
        term = np.multiply(cost, x[:, i])
        grads[i] = np.sum(term) / m
        if i != 0:
            grads[i] += (lam / m) * theta[:, i]
    return grads

def predict(theta, x):
    p = sigmoid(x * theta.T)
    return [ 1 if k > 0.5 else 0 for k in p]
# 加载数据
path = './data_sets/ex2data2.txt'
data = np.loadtxt(path, delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1:]
# 特征映射 --> 6次幂
degree = 6
x = []
for d in data:
    x.append([
        np.power(d[0], i - j) * np.power(d[1], j)
        for i in range(1, degree + 1) for j in range(0, i + 1)
    ])
x = np.c_[np.ones(len(x)), x]
y = np.array(y_data)
# 梯度下降
lam = 1
# lam = 0 # 过拟合
# lam = 100 # 欠拟合
theta = np.zeros(x.shape[1])
print('初始代价: {}'.format(cost(theta, x, y, lam)) )
result = opt.fmin_tnc(func=cost, fprime=gradient, x0=theta, args=(x, y, lam))
print(result)
# 查看训练数据准确率
predictions = predict(np.matrix(result[0]), x)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
# ===================== 绘制图象 ===================== 
def hfunc2(theta, x1, x2):
    temp = theta[0]
    place = 0
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp+= np.power(x1, i-j) * np.power(x2, j) * theta[place+1]
            place+=1
    return temp
def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])
    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1, decision.x2

accepted = [[x[0] for x in data if x[2] == 1], [x[1] for x in data if x[2] == 1]]
rejected = [[x[0] for x in data if x[2] == 0], [x[1] for x in data if x[2] == 0]]
plt.xlabel('Test1 score')
plt.ylabel('Test2 score')
plt.scatter(accepted[0], accepted[1], c='blue', marker='o')
plt.scatter(rejected[0], rejected[1], c='red', marker='x')
x, y = find_decision_boundary(result[0])
plt.scatter(x, y, c='y', s=10, label='Prediction')
plt.show()