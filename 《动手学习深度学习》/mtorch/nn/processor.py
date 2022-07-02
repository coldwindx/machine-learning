
import torch
import torch.utils.data

class Processor:
    def __init__(self, module) -> None:
        self.net = module

    def __net__(self):
        return self.net

    def compile(self, optimizer, loss, metrics = ''):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
    
    def fit(self, data, batch_size, epochs,
             validation_data, validation_split = '', validation_freq = ''):
        train_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
        test_iter = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)
        for epoch in range(epochs):
            train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
            for x, y in train_iter:
                y_ = self.net(x)
                l = self.loss(y_, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                
                train_loss_sum += l.sum().item()
                train_acc_sum += (y_.argmax(dim = 1) == y).sum().item()
                n += y.shape[0]

            acc_sum = 0.0
            for x, y in test_iter:
                acc_sum += (self.net(x).argmax(dim = 1) == y).float().sum().item()

            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
                    (epoch+1, train_loss_sum/n, train_acc_sum/n, acc_sum / len(validation_data)))
