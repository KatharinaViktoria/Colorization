from __future__ import print_function
import argparse
import os
import math
import numpy as np
import numpy.random as npr
import scipy.misc
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg') # switch backend
import matplotlib.pyplot as plt 


from load_data import load_cifar10
from preprocessing import *


HORSE_CATEGORY = 7


######################################################################
# MODELS
######################################################################

class MyConv2d(nn.Module):
    """
    Our simplified implemented of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)

class MyDilatedConv2d(MyConv2d):
    """
    Dilated Convolution 2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(MyDilatedConv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size)
        self.dilation = dilation

    def forward(self, input):
        ############### YOUR CODE GOES HERE ############### 
        weight = torch.Tensor(self.weight.shape[0],
                              self.weight.shape[1],
                              self.weight.shape[2] + (self.weight.shape[2] - 1) * self.dilation,
                              self.weight.shape[3] + (self.weight.shape[3] - 1) * self.dilation)
        weight[:, :, ::(self.dilation + 1), ::(self.dilation + 1)] = self.weight.data
        return F.conv2d(input, nn.parameter.Parameter(weight), self.bias, padding=self.padding + self.padding * self.dilation)
        ###################################################

class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colours):
        super(CNN, self).__init__()
        padding = kernel // 2

        ############### YOUR CODE GOES HERE ############### 
        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            MyConv2d(num_filters * 2, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            MyConv2d(num_filters, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU())

        self.finalconv = nn.Sequential(
            MyConv2d(num_colours, num_colours, kernel_size=kernel),
            nn.Tanh())
        ###################################################

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        self.out_final = self.finalconv(self.out5)
        return self.out_final

class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours):
        super(UNet, self).__init__()

        ############### YOUR CODE GOES HERE ############### 
        padding = kernel // 2
        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            MyConv2d(num_filters * 2, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters * 2 * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            MyConv2d(num_filters * 2, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU())

        self.finalconv = MyConv2d(num_colours + 1, num_colours, kernel_size=kernel)
        ###################################################

    def forward(self, x):
        ############### YOUR CODE GOES HERE ############### 
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(torch.cat([self.out3, self.out2], dim=1))
        self.out5 = self.upconv2(torch.cat([self.out4, self.out1], dim=1))
        self.out_final = self.finalconv(torch.cat([self.out5, x], dim=1))
        return self.out_final
        ###################################################
        pass

class DilatedUNet(UNet):
    def __init__(self, kernel, num_filters, num_colours):
        super(DilatedUNet, self).__init__(kernel, num_filters, num_colours)
        # replace the intermediate dilations
        self.rfconv = nn.Sequential(
            MyDilatedConv2d(num_filters*2, num_filters*2, kernel_size=kernel, dilation=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())
