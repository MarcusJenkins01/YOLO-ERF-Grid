import torch
import torch.nn as nn

# A common block of a convolutional layer followed by batch normalisation then some activation function (I have chosen Leaky ReLU)
class Conv2d_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bn=True):
        super().__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        
        if self.bn:
            out = self.bn(out)
            
        out = self.leaky_relu(out)
        return out



