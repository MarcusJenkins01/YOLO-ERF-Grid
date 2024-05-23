import torch
import torch.nn as nn
from common import *
from erfnet import X_Block
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt


# This simply processes the 40 x 40 output from ERF_PAN to have one output channel, which is the number output
# for each grid cell ranging from 0 to 1, denoting the confidence that an object lies within it.
class ERF_Head(nn.Module):
    def __init__(self, in_channels, init_biases):
        super().__init__()
        self.conv_1 = Conv2d_BN_LeakyReLU(in_channels, 256, 1, 1)
        self.obj_pred = nn.Sequential(
            X_Block(256, 256),
            X_Block(256, 256),
            nn.Conv2d(256, 1, 1, stride=1)
        )
        self.sigmoid_out = nn.Sigmoid()

        if init_biases:
            print("-- INITIALIZING BIASES --")
            b = self.obj_pred.bias.view(1, -1)
            b.data.fill_(-math.log((1 - 1e-2) / 1e-2))
            self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.obj_pred(out)
        out = out.view(-1, 30, 40, 1)
        out = self.sigmoid_out(out)
        return out
