import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_5D(nn.Module):
    '''
    The input Image size is batchsize*1*32*15
    '''
    def __init__(self):
        super(CNN_5D, self).__init__()
        self.cnn_block1 = CNN_Block(in_channels=1, out_channels=64, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block2 = CNN_Block(in_channels=64, out_channels=128, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.linear = nn.Linear(
            in_features=15360, out_features=2)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNN_20D(nn.Module):
    '''
    The input Image size is batchsize*1*64*60
    '''
    def __init__(self):
        super(CNN_20D, self).__init__()
        self.cnn_block1 = CNN_Block(in_channels=1, out_channels=64, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block2 = CNN_Block(in_channels=64, out_channels=128, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block3 = CNN_Block(in_channels=128, out_channels=256, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.linear = nn.Linear(
            in_features=122880, out_features=2)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNN_60D(nn.Module):
    '''
    The input Image size is batchsize*1*96*180
    '''
    def __init__(self):
        super(CNN_20D, self).__init__()
        self.cnn_block1 = CNN_Block(in_channels=1, out_channels=64, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block2 = CNN_Block(in_channels=64, out_channels=128, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block3 = CNN_Block(in_channels=128, out_channels=256, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block4 = CNN_Block(in_channels=256, out_channels=512, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.linear = nn.Linear(
            in_features=552960, out_features=2)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.cnn_block4
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[5, 3], pooling_size=[2, 1], negative_slope=0.01):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=1, padding=[2, 1])
        self.max_pooling = nn.MaxPool2d(kernel_size=pooling_size)
        self.LReLU = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.LReLU(x)
        x = self.max_pooling(x)
        return x



