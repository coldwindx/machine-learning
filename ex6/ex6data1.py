
import scipy as sci
import numpy as np
import pandas as pd

from scipy import io
from matplotlib import pyplot as plt
from sklearn import svm


data = sci.io.loadmat('./data_sets/ex6data1.mat')
x_data, y_data = data['X'], data['y']
print('x_data.shape = {}, y.shape = {}'.format(x_data.shape, y_data.shape))

# 显示数据
def plot_data(x, y):
    posindex = np.where(y == 1)
    negindex = np.where(y == 0)

    positive = x[posindex[0], :]
    negative = x[negindex[0], :]

    plt.scatter(positive[:,0], positive[:,1],c='b',marker='x',label='positive')
    plt.scatter(negative[:,0], negative[:,1],c='r',marker='o',label='negative')

# plot_data(x_data, y_data)
# plt.legend()
# plt.show()

# 支持向量机
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
print(svc)

svc.fit(x_data, y_data)
score = svc.score(x_data, y_data)
print('score = {}'.format(score))

# 可视化分类边界
def plot_decision_boundary(x, y, svc, diff):

    x1min, x1max, x2min, x2max =  0, 4, 1.5, 5
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)
    cordinates = [(m, n) for m in x1 for n in x2]
    x_val, y_val = zip(*cordinates)
    decision = svc.decision_function(cordinates)
    index = np.where(np.abs(decision) < diff)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x1_decision = x_val[index]
    x2_decision = y_val[index]
    plt.scatter(x1_decision, x2_decision, label='decision_boundary')

plot_data(x_data, y_data)
plot_decision_boundary(x_data, y_data,svc, 2 * 10**-3)
plt.legend()
plt.show()