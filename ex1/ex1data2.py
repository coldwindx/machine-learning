# # 在本部分的练习中，需要预测房价，输入变量有两个特征，
# # 一是房子的面积，
# # 二是房子卧室的数量；
# # 输出变量是房子的价格

from matplotlib import pyplot as plt
import numpy as np

# 定义损失函数和梯度下降算法
def compute_cost(x, y, theta):
    m = x.shape[0]
    hypthesis = np.dot(x, np.transpose(theta))
    cost = np.dot(np.transpose(hypthesis - y), hypthesis - y) / (2 * m)
    return cost[0][0]

def gradient_descent(x, y, alpha, epoch):
    m = y.shape[0]
    theta = np.zeros((1, x.shape[1]))
    costs = np.zeros(epoch)
    # 迭代
    for i in range(epoch):
        hypthesis = np.dot(x, np.transpose(theta))
        theta = theta - alpha * np.dot(np.transpose(hypthesis - y), x) / m
        costs[i] = compute_cost(x, y, theta)
    return theta, costs
# 读取数据
path = './data_sets/ex1data2.txt'
data = np.loadtxt(path, delimiter=',')
x_data = data[:,0:2]
y_data = data[:,-1:]
x = np.c_[np.ones(x_data.shape[0]), x_data]
y = np.array(y_data)

# 开始拟合
alpha = 0.05
epoch = 300
theta, costs = gradient_descent(x, y, alpha, epoch)
print(costs)
# 绘制图象
plt.title('Cost and Epoch Curve')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(np.arange(1, epoch + 1), costs)
plt.show()
