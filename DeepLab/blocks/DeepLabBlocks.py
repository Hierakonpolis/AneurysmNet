import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.nn import Conv3d, BatchNorm3d, ReLU, Conv2d, BatchNorm2d
from torch.nn import AdaptiveAvgPool2d, AdaptiveAvgPool3d
import numpy as np
from DeepLab.blocks.BasicBlocks import Interpolate, Cat

class DeepLabv3_Head(nn.Module):
    """
    """
    def __init__(self, in_filters, n_classes, dims, last_layer=True):
        """If last_layer=True then it includes the final
           in_filters -> n_classes 1x1 convolution to produce the logits.
           If false, it won't, which is used by DeepLabv3Plus.
        """
        super(DeepLabv3_Head, self).__init__()

        if dims == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dims == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.project = nn.Sequential(
                Conv(in_filters*5, in_filters, 1),
                BN(in_filters),
                ReLU(),
                )

        self.last = None
        if last_layer:
            self.last = Conv(in_filters, n_classes, 1)

    def forward(self, x):
        x = self.project(x)
        if self.last != None:
            return self.last(x)
        return x

class DeepLabv3_ASPP_ImPooling(nn.Module):
    def __init__(self, in_filters, out_filters, dim):
        super(DeepLabv3_ASPP_ImPooling, self).__init__()

        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
            AvgPool = AdaptiveAvgPool2d
            self.interpol_mode = "bilinear"
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d
            AvgPool = AdaptiveAvgPool3d
            self.interpol_mode = "trilinear"

        self.seq = nn.Sequential(
                AvgPool(1), # (Output = # of filters, avg across the HxW(xD)
                Conv(in_filters, out_filters, 1),
                #BN(out_filters), # This BN will throw an error if batch = 1
                ReLU(),
                )
        self.interpolate = Interpolate()

    def forward(self, x):
        out = self.seq(x)
        out = self.interpolate(out, x.shape[2:], mode=self.interpol_mode,
                align_corners=False)
        return out

class DeepLabv3_ASPP(nn.Module):
    """Head of DeepLabv3.
       This head combines: (a) 1x1, 3x3, 3x3, 3x3 and (b) global avg pooling.

    """

    def __init__(self, in_filters, out_filters, dilation_rates, dim):
        super(DeepLabv3_ASPP, self).__init__()

        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.aspp = nn.ModuleList()
        self.aspp.append(nn.Sequential(
                Conv(in_filters, out_filters, 1),
                BN(out_filters),
                ReLU(),
                ))

        for rate in dilation_rates:
            self.aspp.append(nn.Sequential(
                Conv(in_filters, out_filters, 3, dilation=rate, padding=rate),
                BN(out_filters),
                ReLU(),
                ))

        self.pooling = DeepLabv3_ASPP_ImPooling(in_filters, out_filters, dim)

        self.cat = Cat()

    def forward(self, x):
        
        out = []
        for i in range(len(self.aspp)):
            out.append(self.aspp[i](x))
        out.append(self.pooling(x))

        out = self.cat(out, dim=1)
        return out

