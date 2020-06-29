from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU, Conv2d, BatchNorm2d
from torch.nn import MaxPool2d, MaxPool3d
import numpy as np
import torch
from DeepLab.blocks.BasicBlocks import Sum, Interpolate, Softmax, Cat

class Sep_Conv_DeepLabv3Plus(nn.Module):
    def __init__(self, filters, stride, dilation, skip_con, dim,
            return_skip=False):
        """filters: list of 4 filters for the 3 convolutions
           stride: stride of the last conv.
           dilation: dilation rate
           skip_con: type of skip connection (sum, none, conv)
           dim: dimensions
           return_skip: whether to return the second seq for the skip conenction
        """
        super(Sep_Conv_DeepLabv3Plus, self).__init__()
        self.return_skip = return_skip

        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d

        self.seq1 = nn.Sequential(
                Depthwise_Sep_Conv(filters[0], filters[1],
                    Conv, dilation=dilation,
                    stride=1),
                BN(filters[1]),
                ReLU()
                )

        self.seq2 = nn.Sequential(
                Depthwise_Sep_Conv(filters[1], filters[2],
                    Conv, dilation=dilation,
                    stride=1),
                BN(filters[2]),
                ReLU()
                )

        self.seq3 = nn.Sequential(
                Depthwise_Sep_Conv(filters[2], filters[3],
                    Conv, dilation=dilation,
                    stride=stride),
                BN(filters[3]),
                ReLU()
                )

        # Skip connection (if any)
        skip_layers = []
        if skip_con == "sum":
            self.skip = nn.Sequential() # Identity
        elif skip_con == "conv":
            self.skip = nn.Sequential(
                    Conv(filters[0], filters[-1], 1, stride=stride),
                    BN(filters[-1])
                    )
        else:
            self.skip = None

        self.sum = Sum()

    def forward(self, x):
        x1 = self.seq1(x)
        x1 = self.seq2(x1) # This may be returned for the skip connection
        out = self.seq3(x1)

        if self.skip != None:
            out = self.sum([self.skip(x), out])

        if self.return_skip:
            return out, x1
        else:
            return out

class Depthwise_Sep_Conv(nn.Module):
    """Generic 'Sep Conv' used by Xception_DeepLabv3Plus (Fig. 4)
    """
    def __init__(self, in_filters, out_filters, Conv, dilation=1, stride=1):
        super(Depthwise_Sep_Conv, self).__init__()

        self.conv = Conv(in_filters, in_filters, 3, padding=dilation,
                groups=in_filters, dilation=dilation, stride=stride)
        self.pointwise = Conv(in_filters, out_filters, 1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x

