import torch
from abc import ABCMeta  

class Module(torch.nn.Module):
    # 指定Module为抽象类
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        super(Module, self).__init__()
    
    def compile(self, optimizers, loss):
        self.optimizers = optimizers
        self.loss = loss

    def fit(self, data, batch_size, epochs, validation_data):
        train_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
        test_iter = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)
        for epoch in range(epochs):
            train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
            for x, y in train_iter:
                y_ = self.forward(x)
                l = self.loss(y_, y)
                for opt in self.optimizers:
                    opt.zero_grad()
                l.backward()
                for opt in self.optimizers:
                    opt.step()
      
                train_loss_sum += l.sum().item()
                train_acc_sum += (y_.argmax(dim = 1) == y).sum().item()
                n += y.shape[0]

            acc_sum = 0.0
            for x, y in test_iter:
                acc_sum += (self.forward(x).argmax(dim = 1) == y).float().sum().item()

            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
                    (epoch+1, train_loss_sum/n, train_acc_sum/n, acc_sum / len(validation_data)))
