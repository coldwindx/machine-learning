# 这个部分需要你实现手写数字（0到9）的识别。
# 你需要扩展之前的逻辑回归，并将其应用于一对多的分类。
import matplotlib
import numpy as np
from scipy.io import loadmat
from scipy import optimize as opt
from matplotlib import pyplot as plt
from sklearn import metrics
# 定义函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, x, y, lam):
    (m, n) = x.shape
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    first = - np.multiply(y, np.log(sigmoid(x * theta.T)))
    second = - np.multiply(1 - y, np.log(1 - sigmoid(x * theta.T)))
    reg = lam / (2 * m) * np.sum(np.power(theta[:,1:], 2))
    return np.sum(first + second) / m + reg

def gradient(theta, x, y, lam):
    (m, n) = x.shape
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    # 向量化计算
    cost = sigmoid(x * theta.T) - y
    grads = ((x.T * cost).T + lam * theta) / m
    grads[0, 0] = np.sum(np.multiply(cost, x[:, 0])) / m
    return np.array(grads).ravel()

def one_vs_all(x, y, labels, lam):
    (rows, features) = x.shape
    thetas = np.zeros((labels, features + 1))
    x = np.insert(x, 0, values=np.ones(rows), axis=1)
    for i in range(1, labels + 1):
        theta = np.zeros(features + 1)
        y_i = np.array([1 if v == i else 0 for v in y])
        y_i = np.reshape(y_i, (rows, 1))
        fmin = opt.minimize(fun=cost, x0 = theta, args=(x, y_i, lam), method='TNC', jac=gradient)
        thetas[i-1:] = fmin.x
    return thetas
# =================== 加载数据 =======================
path = './data_sets/ex3data1.mat'
data = loadmat(path)
print('x.shape = {}, y.shape = {}'.format(data['X'].shape, data['y'].shape))
thetas = one_vs_all(data['X'], data['y'], 10, 1)
print(thetas)
# =================== 数据可视化 =======================
# sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
# sample_x = data['X'][sample_idx, :]
# # print(sample_x)
# fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(12, 12))
# for r in range(10):
#     for c in range(10):
#         img = sample_x[r * 10 + c].reshape(20, 20).T
#         axes[r, c].matshow(img, cmap=matplotlib.cm.binary)
# plt.xticks(np.array([]))
# plt.yticks(np.array([]))
# plt.show()
# =================== 预测 =======================
def predict(thetas, x):
    (rows, features) = x.shape
    x = np.insert(x, 0, values=np.ones(rows), axis=1)
    thetas = np.matrix(thetas)
    h = sigmoid(x * thetas.T)
    h_argmax = np.argmax(h, axis=1) + 1
    return h_argmax

y_pred = predict(thetas, data['X'])
print(metrics.classification_report(data['y'], y_pred))