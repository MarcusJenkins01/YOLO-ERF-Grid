import torch
import torch.nn as nn
from common import *
from erfnet import ERF_Block_S1, ERF_Block_S2

# [1] http://assets.researchsquare.com/files/rs-2656656/v1/fcbd6ca2e2d80a34706e5f00.pdf
# See fpn_new.png in the root directory for a diagram of this updated FPN.
# P3 is the 80 x 80 feature map, which is downsized to P4's size (40 x 40), and P5 is 20 x 20 and so is up-sampled to
# P4's size. All three, P3, P4 and P5, are then concatenated together to form the output grid of 40 x 40.
# The intermediate steps link the multiple scales of P3, P4 and P5 together to relate objects of multiple scales (what
# a feature pyramid network is designed to do).
class ERF_PAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_c5 = Conv2d_BN_LeakyReLU(512, 256, 1, 1)
        self.upsample_c5 = nn.Upsample(scale_factor=2)
        self.erf_s1_c4 = ERF_Block_S1(512, 256)
        self.conv_c4 = Conv2d_BN_LeakyReLU(256, 128, 1, 1)
        self.upsample_c4 = nn.Upsample(scale_factor=2)
        self.erf_s1_p3 = ERF_Block_S1(256, 128)
        self.erf_s2_p3 = ERF_Block_S2(128, 128)
        self.erf_s1_p4 = ERF_Block_S1(256, 128)
        self.erf_s2_p4 = ERF_Block_S2(128, 128)
        self.erf_s1_p5 = ERF_Block_S1(256+128, 128)

        self.p3_erf_s1_out = ERF_Block_S1(128, 128)
        self.p3_erf_s2_out = ERF_Block_S2(128, 128)
        self.p5_conv_out = Conv2d_BN_LeakyReLU(128, 128, 1, 1)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p5_upsample = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        c3, c4, c5 = x[0], x[1], x[2]
        c5 = self.conv_c5(c5)
        c5_up = self.upsample_c5(c5)
        c4 = torch.cat((c4, c5_up), dim=1)
        c4 = self.erf_s1_c4(c4)
        c4 = self.conv_c4(c4)
        c4_up = self.upsample_c4(c4)
        p3 = torch.cat((c3, c4_up), dim=1)
        p3 = self.erf_s1_p3(p3)
        p3_erf_s2 = self.erf_s2_p3(p3)
        p4 = torch.cat((c4, p3_erf_s2), dim=1)
        p4 = self.erf_s1_p4(p4)
        p4_erf_s2 = self.erf_s2_p4(p4)
        p5 = torch.cat((c5, p4_erf_s2), dim=1)
        p5 = self.erf_s1_p5(p5)

        p3 = self.p3_erf_s1_out(p3)
        p3 = self.p3_erf_s2_out(p3)
        p5 = self.p5_conv_out(p5)
        p5 = self.p5_upsample(p5)

        return torch.cat((p3, p4, p5), dim=1)
