import scipy as sci 
import numpy as np

from scipy import io
from matplotlib import pyplot as plt

class PCA:
    def __init__(self, x) -> None:
        (m, n) = x.shape
        x = (x - x.mean()) / x.std()
        x = np.matrix(x)
        sigma = (x.T * x) / m
        # 奇异值分解
        U, S, V = np.linalg.svd(sigma)
        self.U = U
    def pca(self, x, k):
        return x * self.U[:, :k]
    def recover(self, z, k):
        return z * self.U[:, :k].T

if __name__ == '__main__':
    data = sci.io.loadmat('./data_sets/ex7data1.mat')
    x_data = np.array(data['X'])
    # 可视化输入
    plt.scatter(x_data[:,0], x_data[:,1], c='blue')
    plt.show()


    pca = PCA(x_data)
    z = pca.pca(x_data, 1)
    # print(z)
    x_recover = pca.recover(z, 1)
    # print(x_recover)

    # 可视化输出
    x_recover = np.array(x_recover)
    plt.scatter(x_recover[:,0], x_recover[:,1], c='blue')
    plt.show()
