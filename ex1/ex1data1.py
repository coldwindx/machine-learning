# 在本部分的练习中，您将使用一个变量实现线性回归，以预测食品卡车的利润。
# 假设你是一家餐馆的首席执行官，正在考虑不同的城市开设一个新的分店。
# 该连锁店已经在各个城市拥有卡车，而且你有来自城市的利润和人口数据。
# 您希望使用这些数据来帮助您选择将哪个城市扩展到下一个城市。
# x表示人口数据，y表示利润，一共97行数据。

import numpy as np
import matplotlib.pyplot as plot

path = './data_sets/ex1data1.txt'
data = np.loadtxt(path, delimiter=',', usecols=(0, 1))

# 损失函数
def compute_cost(x, y, theta):
    hypthesis = np.dot(x, np.transpose(theta))
    cost = np.dot(np.transpose(hypthesis - y), hypthesis - y) / 2 / x.shape[0]
    return cost
# 梯度下降
def gradient_descant(x, y, alpha, epoch):
    m = y.shape[0]
    theta = np.zeros(x.shape[1])
    # 用于纪录
    costs = np.zeros(epoch)
    # 迭代
    for i in range(epoch):
        costs[i] = compute_cost(x, y, theta)
        hypthesis = np.dot(x, np.transpose(theta))
        theta = theta - alpha * np.dot(np.transpose(hypthesis - y), x) / m
    return theta, costs
# 梯度下降法进行拟合过程
x_data = data[:, 0]
y_data = data[:, 1]
    # 这里给x添加一列，用于对应系数theta0
x = np.c_[np.ones(x_data.shape[0]), x_data]
y = np.array(y_data)

alpha = 0.01
epoch = 1500
theta, costs = gradient_descant(x, y, alpha, epoch)
# 绘制图象
plot.subplot(1, 2, 1)
plot.title('Cost and Epoch Curve')
plot.xlabel('epoch')
plot.ylabel('cost')
plot.plot(costs)

plot.subplot(1, 2, 2)
plot.title('Population and Profit Curve')
plot.scatter(x_data, y_data, c='blue', marker='.', s=20)
plot.plot(x_data, np.dot(x, np.transpose(theta)), 'r', label='Prediction')
plot.xlabel('population')
plot.ylabel('profit')
plot.show()


