
import numpy as np
import scipy as sci

from scipy import io
from scipy import stats
from matplotlib import pyplot as plt

class AnomalyDetecter:

    # 计算高斯分布
    def norm(self, x):
        (m, n) = x.shape
        self.u = np.mean(x, axis=0)
        self.sigma2 = np.var(x, axis=0)
        return self.u, self.sigma2

    # 计算概率
    def pdf(self, x):
        (m, n) = x.shape
        p = np.zeros((m, n))
        for i in range(n):
            p[:,i] = sci.stats.norm(self.u[i], self.sigma2[i]).pdf(x[:,i])
        return p

    # 阈值选择
    def threshold(self, p, y):
        best_epsilon = 0
        best_f1 = 0
        step = (p.max() - p.min()) / 1000

        for epsilon in np.arange(p.min(), p.max(), step):
            tp = np.sum(np.logical_and(p < epsilon, y == 1)).astype(float)
            fp = np.sum(np.logical_and(p < epsilon, y == 0)).astype(float)
            fn = np.sum(np.logical_and(p >= epsilon, y == 1)).astype(float)
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = epsilon
        return best_epsilon, best_f1

    def print(self):
        print('u={}, sigma2={}'.format(self.u, self.sigma2))

    def compute(self, shape, *args):
        y = np.zeros(shape)
        for i, x in enumerate(args):
            y -= np.power(x - self.u[i], 2) / (2 * self.sigma2[i])
        return np.exp(y)

if __name__ == '__main__':
    data = sci.io.loadmat('./data_sets/ex8data1.mat')
    x_data = np.array(data['X'])
    # 可视化
    plt.scatter(x_data[:,0], x_data[:,1], c='blue')
    plt.show()

    # 计算高斯分布
    detecter = AnomalyDetecter()
    u, sigma2 = detecter.norm(x_data)
    print('u={}, sigma2={}'.format(u, sigma2))

    # 绘制概率轮廓分布
    x = np.linspace(0, 25, 100)
    y = np.linspace(0, 25, 100)
    x, y = np.meshgrid(x, y)
    z = detecter.compute(x.shape, x, y)

    plt.scatter(x_data[:,0], x_data[:,1], c='blue')
    plt.contour(x, y, z, [10**-11, 10**-7, 10**-5, 10**-3, 0.1], c='k')
    plt.show()

    #选择阈值ε
    x_val, y_val = data['Xval'], data['yval']
    p_val = detecter.pdf(x_val)
    epsilon, f1 = detecter.threshold(p_val, y_val)
    print('epsilon, f1 = {}, {}'.format(epsilon, f1))

    # 可视化检测结果
    p = detecter.pdf(x_data)
    index = np.where(p < epsilon)
    plt.scatter(x_data[:,0], x_data[:,1], c='blue')
    plt.scatter(x_data[index[0],0], x_data[index[0],1], c='red')
    plt.show()
