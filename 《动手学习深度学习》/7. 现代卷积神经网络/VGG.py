import torch
import torch.nn

## 构造VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        in_channels = out_channels
    layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    return torch.nn.Sequential(*layers)

## 构造VGG网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return torch.nn.Sequential(
        *conv_blks,
        torch.nn.Flatten(),
        torch.nn.Linear(out_channels * 7 * 7, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 10)
    )

import os
import helper
if __name__ == '__main__':
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    if os.path.exists('net.pk'):
        net.load_state_dict(torch.load('net.pk'))
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = helper.load_data_fashion_mnist(batch_size, path='../data', resize=224)
    helper.train_ch6(net, train_iter, test_iter, num_epochs, lr, helper.GPU.try_gpu())
    

