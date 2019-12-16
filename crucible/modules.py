import torch
import torch.nn as nn
from .normalization import GroupNorm

class fcdr(nn.Module):
    def __init__(self, in_features, out_features, p=0.5, activation=True):
        super(fcdr, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(p=p),
        )
        if activation: 
            self.seq.add_module("activatoin", nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)

class fcbr(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super(fcbr, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        if activation: 
            self.seq.add_module("activatoin", nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)

class fcgr(nn.Module):
    def __init__(self, in_features, out_features, num_groups=32, activation=True):
        super(fcgr, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GroupNorm(num_groups, out_features),
        )
        if activation: 
            self.seq.add_module("activatoin", nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)

class conv1dbr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv1dbr, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        if activation: 
            self.seq.add_module("activation", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x) 


class conv2dbr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv2dbr, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        if activation: self.seq.add_module("activation", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)


class conv2dgr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv2dgr, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            GroupNorm(out_channels),
        )
        if activation: self.seq.add_module("activation", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)

"""
Experiment shows that BN after ReLU is better.
ref: https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
"""

class conv1drb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv1drb, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv", nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding))
        if activation: self.seq.add_module("activation", nn.ReLU(inplace=True))
        self.seq.add_module("bn", nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return self.seq(x) 

class conv2drb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(conv2drb, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding))
        if activation: self.seq.add_module("activation", nn.ReLU(inplace=True))
        self.seq.add_module("bn", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.seq(x) 

class fcrb(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super(fcrb, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("fc", nn.Linear(in_features, out_features))
        if activation: self.seq.add_module("activatoin", nn.ReLU(inplace=True))
        self.seq.add_module("bn", nn.BatchNorm1d(out_features))

    def forward(self, x):
        return self.seq(x)
