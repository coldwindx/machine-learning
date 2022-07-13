import torch


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