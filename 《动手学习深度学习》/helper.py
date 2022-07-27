import os
import hashlib
import requests
import numpy as np

import torch
import torchvision

from time import time
from urllib import request
from matplotlib import pyplot as plt
from torch import all
from IPython import display
from torchvision import transforms

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

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

class Loader:
    @staticmethod
    def load_data_fashion_mnist(batch_size, path = "", resize = None):
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        # 下载数据集
        mnist_train = torchvision.datasets.FashionMNIST(
            root = path, train=True, download=True,
            transform=trans     # 自动转为torch张量
        )
        mnist_test = torchvision.datasets.FashionMNIST(
            root = path, train=False, download=True,
            transform=trans     # 自动转为torch张量
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

    @staticmethod
    def load_array(data_arrays, batch_size, shuffle = True):
        datasets = torch.utils.data.TensorDataset(*data_arrays)
        return torch.utils.data.DataLoader(datasets, batch_size, shuffle=shuffle)
    
    @staticmethod
    def download(url, folder = '../data', sha1_hash = None):
        if not url.startswith('http'):
            url, sha1_hash = DATA_HUB[url]
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, url.split('/')[-1])
        # check if hit cache
        if os.path.exists(fname) and sha1_hash:
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:
                while True:
                    data = f.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return fname
        # download from url
        print(f'Downloading {fname} from {url} ...')
        r = requests.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return fname

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
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        
        # display.display(self.fig)
        # display.clear_output(wait=True)

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

def evaluate_accuracy_gpu(net, data_iter, device = None):
    '''使用GPU计算操作模型在数据集上的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval() # 切换评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    '''用GPU训练模型'''
    def init_weight(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    print('train on ', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss = torch.nn.CrossEntropyLoss()
    animator = Animator(xlabel='epochs', xlim = [1, num_epochs],
                                legend=['train_loss', 'train_acc', 'test_acc'])
    timer, num_batchs = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():       # 不更新梯度
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batchs // 5) == 0 or i == num_batchs - 1:
                animator.add(epoch + (i + 1) / num_batchs,
                                            (train_l, train_acc, None))
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        torch.save(net.state_dict(), 'net.pt')
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    plt.savefig('accuracy.png')
    plt.show()

class Evaluater:
    @staticmethod
    def loss(net, data_iter, loss):
        metric = Accumulator(2) 
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
        return metric[0] / metric[1]

if __name__ =='__main__':
    a = Animator(xlabel='epochs', ylabel='metrics', metrics=['m'])
    # x = [1, 2, 3, 4, 5]
    # y = [4, 5, 6, 7, 8]
    a.add(0.1, 0.2, 'm')
    a.add(2, 3, 'm')
    # a.show()
    a.show()