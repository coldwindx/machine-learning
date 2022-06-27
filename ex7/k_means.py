
import numpy as np
import scipy as sci

from matplotlib import pyplot as plt
from scipy import io

def k_means(x, k, epochs = 10):
    (m, n) = x.shape
    u = x[np.random.choice(m, size=k),:]
    c = np.zeros(m, dtype=np.int64)

    for i in range(epochs):
        u_temp = np.zeros((k, n))
        u_num = np.zeros(k)
        for j in range(m):
            dist = np.sum(np.power(np.abs(u - x[j]), 2), axis=1)
            c[j] = np.argmin(dist)
            u_temp[c[j]] = u_temp[c[j]] + x[j]
            u_num[c[j]] += 1
        u = [u_temp[i] / u_num[i] for i in range(k)]
    return u, c

if __name__ == '__main__':
    data = sci.io.loadmat('./data_sets/ex7data2.mat')
    x_data = np.array(data['X'])
    # 可视化输入
    # plt.scatter(x_data[:,0], x_data[:,1], c='blue')
    # plt.show()
    K = 3
    u, c = k_means(x_data, K)

    cluster0 = x_data[np.where(c == 0)[0],:]
    cluster1 = x_data[np.where(c == 1)[0],:]
    cluster2 = x_data[np.where(c == 2)[0],:]

    plt.scatter(cluster0[:,0], cluster0[:,1], c='red')
    plt.scatter(cluster1[:,0], cluster1[:,1], c='green')
    plt.scatter(cluster2[:,0], cluster2[:,1], c='blue')

    plt.show()
    