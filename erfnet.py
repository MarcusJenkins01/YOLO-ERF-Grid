import torch
import torch.nn as nn
from common import *
import torch.nn.functional as F


# [1] http://assets.researchsquare.com/files/rs-2656656/v1/fcbd6ca2e2d80a34706e5f00.pdf
# [2] http://arxiv.org/pdf/2111.09957.pdf
# [3] http://arxiv.org/pdf/1709.01507.pdf
# [4] https://github.com/ultralytics/yolov5/blob/master/models/common.py#L236

# B: the batch size
# H: the height dimension of the tensor, which is also the number of rows
# W: the width dimension of the tensor, which is also the number of columns
# C: the number of channels, this is 3 for an image (R, G and B) or the number of filters for a convolutional layer


# Squeeze-and-Excitation block [3]
# Essentially takes the output of a CNN of dimensions B x H x W x C and applies global average pooling to every pixel in each channel
# to produce a tensor of size B x 1 x 1 x C. This is then fed into a linear neural network of C number of input neurons and C//ratio
# number of output neurons, and a non-linear activation layer is applied, e.g. ReLU or Leaky ReLU. It is then passed through another
# linear network with C number of outputs. After this a sigmoid activation is applied to normalise the outputs to between 0 and 1.
# The resulting array of length C is then multiplied by the original feature map of dimensions B x H x W x C, so that every pixel in
# each channel is scaled by the corresponding value in the list according to the corresponding channel. This allows the network to
# essentially learn which filters to pay the most attention to, by allowing it to scale the intensity of activation of each pixel in
# a filter by a learned factor of between 0 and 1.
class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        cr = in_channels // reduction_ratio  # Since linear networks are expensive we can reduce the size of the inner layer(s)
        self.fc_1 = nn.Linear(in_channels, cr)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(cr, in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        out = out.permute(0, 2, 3, 1)
        #print(out.shape)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.sigmoid(out)
        out = out.permute(0, 3, 1, 2)
        out = out * x
        return out


# X block [1]
# Derived from the D block [2]
# This works by first having a convolutional layer of kernel size 1x1. The output is then split equally into two halves. The first
# half is fed into a grouped convolutional layer. This means that rather than having a filter for each input channel,
# we have g number of filters that is applied over each group of channels at a time of size g channels. This is essentially convolving
# but over channels instead. Dilation is also used for both grouped convolutional layers. The output of both is then concatenated
# channel-wise, and fed into the SE block as implemented above. The idea of this X block is to improve field of view (also referred
# to as receptive field) of the CNN whilst keeping the number of parameters relatively unchanged.
class X_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=4, d1=1, d2=1, p1=1, p2=1):
        super().__init__()
        self.conv_1_out_ch = 256
        self.d1_d2_in_ch = self.conv_1_out_ch // 2  # We split the output of the 1x1 conv in half
        self.d1_d2_out_ch = 64
        self.se_in_ch = self.d1_d2_out_ch * 2  # d1 and d2 are concatenated channel-wise before going into the SE block
        
        self.conv_1 = Conv2d_BN_LeakyReLU(in_channels, self.conv_1_out_ch, 1, 1)
        self.conv_d1 = Conv2d_BN_LeakyReLU(self.d1_d2_in_ch, self.d1_d2_out_ch, 3, stride, padding=p1, dilation=d1, groups=g)
        self.conv_d2 = Conv2d_BN_LeakyReLU(self.d1_d2_in_ch, self.d1_d2_out_ch, 3, stride, padding=p2, dilation=d2, groups=g)
        self.bn = nn.BatchNorm2d(self.se_in_ch)
        self.se_block = SE_Block(self.se_in_ch)
        self.conv_final = Conv2d_BN_LeakyReLU(self.se_in_ch, out_channels, 1, 1)  # Output depth of the SE block is the same as for the input

    def forward(self, x):
        out = self.conv_1(x)
        halves = torch.split(out, self.d1_d2_in_ch, dim=1)
        d1_out = self.conv_d1(halves[0])
        d2_out = self.conv_d2(halves[1])
        out = torch.cat((d1_out, d2_out), dim=1)
        out = self.bn(out)  # Perform batch norm again on new set
        out = self.se_block(out)
        out = self.conv_final(out)
        return out
        

# ERF block when stride is 1 [1]
# This splits the input channels in half and passes one half through a single 1x1 convolutional layer, and the other half through
# a 1x1 convolutional layer followed by a sequence of X blocks. The output of the single conv layer from the first half is
# concatenated with the output of the 1x1 conv layer of the other half and the output of every X block to give one output of
# the same number of channels / depth of the original input. This is then passed through a single final 1x1 convolutional layer.
class ERF_Block_S1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c_div_2 = in_channels // 2

        self.conv_left = Conv2d_BN_LeakyReLU(self.c_div_2, self.c_div_2, 1, 1)
        self.conv_right = Conv2d_BN_LeakyReLU(self.c_div_2, self.c_div_2, 1, 1)
        self.x_block_1 = X_Block(self.c_div_2, self.c_div_2)
        self.x_block_2 = X_Block(self.c_div_2, self.c_div_2, d2=2, p2=2)
        self.x_block_3 = X_Block(self.c_div_2, self.c_div_2, d2=5, p2=5)
        self.conv_final = Conv2d_BN_LeakyReLU(self.c_div_2 * 5, out_channels, 1, 1)

    def forward(self, x):
        split = torch.split(x, self.c_div_2, dim=1)
        left = split[0]
        right = split[1]
        cl_out = self.conv_left(left)
        cr_out = self.conv_right(right)
        xb_1_out = self.x_block_1(cr_out)
        xb_2_out = self.x_block_2(xb_1_out)
        xb_3_out = self.x_block_3(xb_2_out)
        out = torch.cat((cl_out, cr_out, xb_1_out, xb_2_out, xb_3_out), dim=1)
        out = self.conv_final(out)
        return out


# ERF block when stride is 2 [1]
# The purpose of this is to incorporate an X Block for learning features as well as a max pooling layer to downsample the input
class ERF_Block_S2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.split_channels = in_channels // 2
        self.x_block = X_Block(self.split_channels, self.split_channels, stride=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv_final = Conv2d_BN_LeakyReLU(in_channels, out_channels, 1, 1)

    def forward(self, x):
        split = torch.split(x, self.split_channels, dim=1)
        left = split[0]
        right = split[1]
        x_block_out = self.x_block(left)
        max_pool_out = self.max_pool(right)
        out = torch.cat((x_block_out, max_pool_out), dim=1)
        out = self.conv_final(out)
        return out


# An ERF module consists of a stride 2 ERF block followed by a stride 1 ERF block [1]
class ERF_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.erf_stride_2 = ERF_Block_S2(in_channels, out_channels)
        self.erf_stride_1 = ERF_Block_S1(out_channels, out_channels)

    def forward(self, x):
        out = self.erf_stride_2(x)
        out = self.erf_stride_1(out)
        return out


# The Focus module as implemented in YOLOv5 [4]
# This module splits the image up into 4 separate images by taking each pixel at every even row and even column, then every odd row and
# even column, then even row and odd column, and finally every odd row and odd column. These 4 separate images are then concantenated
# together as separate channels. The resultant tensor is therefore of size B x (3x4) x H/2 x W/2. This is then passed through a final
# convolutional layer. The reasoning behind this module is to reduce the width and height of the image to reduce the number of
# parameters without losing any information [1], since the information is simply rearranged into channel space (c-space) [4].
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = Conv2d_BN_LeakyReLU(in_channels * 4, out_channels, kernel_size, stride, padding=padding, groups=groups)
        
    def forward(self, x):
        even_r_even_c = x[..., ::2, ::2]  # This is taking the value at every even row (height) and even column (width) of the image
        odd_r_even_c = x[..., 1::2, ::2]  # The value at every odd row and even column of the image
        even_r_odd_c = x[..., ::2, 1::2]  # The value at every even row and odd column of the image
        odd_r_odd_c = x[..., 1::2, 1::2]  # The value at every odd row and odd column of the image
        out = torch.cat((even_r_even_c, odd_r_even_c, even_r_odd_c, odd_r_odd_c), dim=1)  # Concatenate along channel dimension
        out = self.conv(out)
        return out


# The backbone for YOLO-ERF-T [1]
# This consists of a Focus module that takes in the input image of size B x 3 x H x W, followed by a number of ERF modules.
# The output of this backbone are the three inputs to ERF-PAN [1].
class ERF_Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.focus = Focus(3, 32)
        self.erf_module_1 = ERF_Module(32, 64)
        self.erf_module_2 = ERF_Module(64, 128)
        self.erf_module_3 = ERF_Module(128, 256)
        self.erf_module_4 = ERF_Module(256, 512)

    def forward(self, x):
        out = self.focus(x)
        erf_1_out = self.erf_module_1(out)
        erf_2_out = self.erf_module_2(erf_1_out)
        erf_3_out = self.erf_module_3(erf_2_out)
        erf_4_out = self.erf_module_4(erf_3_out)
        return erf_2_out, erf_3_out, erf_4_out  # C3, C4 and C5 [1]
        

##backbone = ERF_Backbone(640)
##backbone.eval()
##
##x = torch.rand((1, 3, 640, 640), dtype=torch.float32)
##c3, c4, c5 = backbone(x)
##pytorch_total_params = sum(p.numel() for p in backbone.parameters())
##
##print("Input shape:", x.shape)
##print()
##print("C3 shape:", c3.shape)
##print("C4 shape:", c4.shape)
##print("C5 shape:", c5.shape)
##print()
##print("Total parameters:", pytorch_total_params)
