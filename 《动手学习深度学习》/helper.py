from cProfile import label
from time import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision


def corr2d(X, K):
    '''
    @desc: 计算二维互相关运算
    @param: X 输入特征 
            K 卷积核
    '''
    h, w = K.shape
    m, n = X.shape
    Y = torch.zeros((m - h + 1, n - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = torch.sum(X[i:i + h, j:j + w] * K)
    return Y

def load_data_fashion_mnist(batch_size, path = ""):
    # 下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root = path, train=True, download=True,
        transform=torchvision.transforms.ToTensor()     # 自动转为torch张量
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = path, train=False, download=True,
        transform=torchvision.transforms.ToTensor()     # 自动转为torch张量
    )
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size,
        shuffle=True,
        num_workers=0       # 开启num_workers个线程
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size,
        shuffle=True,
        num_workers=0       # 开启num_workers个线程
    )
    return train_iter, test_iter

def accuracy(y_hat, y):
    '''计算预测准确率'''
    m, n = y_hat.shape
    if 1 < m and 1 < n:
        y_hat = y_hat.argmax(axis = 1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

class GPU:
    @staticmethod
    def try_gpu(i = 0):
        '''如果存在，则返回gpu(i)，否则返回cpu()'''
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f"cuda:{i}")
        return torch.device('cpu')
    @staticmethod
    def try_all_gpu():
        '''返回所有可用的GPU，如果没有GPU，则返回[cpu(),]'''
        devices = [ torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return devices if devices else [torch.device('cpu')]

class Animator:
    '''简单的曲线绘制封装类'''
    def __init__(self, xlabel = None, ylabel = None, legend = None,
                        xlim = None, ylim = None, nrows = 1, ncols = 1) -> None:
        _, self.axes = plt.subplots(nrows, ncols)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        if legend is None:
            legend = []
        plt.legend(legend)
    def add(self, x, y):
        self.axes.plot(x, y)
    def show(self):
        plt.show()

class Accumulator():
    '''实用程序类Accumulator，用于对多个变量进行累加。'''
    def __init__(self, n) -> None:
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    '''记录运行时间'''
    def __init__(self) -> None:
        self.times = []
        self.tik = time()
    def start(self):
        self.tik = time()
    def stop(self):
        self.times.append(time() - self.tik)
        return self.times[-1]
    def sum(self):
        return sum(self.times)
    def avg(self):
        return sum(self.times) / len(self.times)
    def cunsum(self):
        return np.array(self.times).cumsum().tolist()


if __name__ =='__main__':
    load_data_fashion_mnist(256)