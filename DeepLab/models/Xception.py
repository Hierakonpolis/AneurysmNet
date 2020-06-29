from torch import nn
from torch.nn import Conv3d, Conv2d, MaxPool2d, MaxPool3d, BatchNorm3d, Linear
from torch.nn import BatchNorm2d, ReLU, AdaptiveAvgPool2d, AdaptiveAvgPool3d
from DeepLab.blocks.XceptionBlocks import Sep_Conv_DeepLabv3Plus

class Xception_DeepLabv3Plus(nn.Module):
    """ResNet version used by DeepLab.
       It differs from ResNet in the fully-connected part.
    """
    params = ["modalities", "first_filters", "n_classes", "dim"]
    def __init__(self, modalities, first_filters=32, n_classes=20, dim="2D"):
        super(Xception_DeepLabv3Plus, self).__init__()

        # Originally, first_filters = 32
        if dim == "2D":
            Conv = Conv2d
            BN = BatchNorm2d
            MaxPool = MaxPool2d
            AvgPool = AdaptiveAvgPool2d
        elif dim == "3D":
            Conv = Conv3d
            BN = BatchNorm3d
            MaxPool = MaxPool3d
            AvgPool = AdaptiveAvgPool3d

        nfi = first_filters
        nf2 = int(22.75*nfi) # Originall: 728
        layers1 = []

        #skip_con=sum|conv|none, filters=[1,2,3], strides=1|2, rate=1|2

        ### Entry Flow
        layers1.append(
                Conv(modalities, nfi, kernel_size=3, stride=2, padding=1))
        layers1.append(BN(nfi))
        layers1.append(ReLU())
        layers1.append(
                Conv(nfi, nfi*2, kernel_size=3, stride=1, padding=1))
        layers1.append(BN(nfi*2))
        layers1.append(ReLU())

        # Module 1
        layers1.append(Sep_Conv_DeepLabv3Plus(
            filters=[nfi*2, nfi*4, nfi*4, nfi*4], stride=2, dilation=1,
            skip_con="conv", dim=dim))
        # Module 2
        layers1.append(Sep_Conv_DeepLabv3Plus(
            filters=[nfi*4, nfi*8, nfi*8, nfi*8], stride=2, dilation=1,
            skip_con="conv", dim=dim, return_skip=True))

        self.xception_part1 = nn.Sequential(*layers1)

        layers2 = []
        # Module 3
        layers2.append(Sep_Conv_DeepLabv3Plus(
            filters=[nfi*8, nf2, nf2, nf2], stride=2, dilation=1,
            skip_con="conv", dim=dim))

        for i in range(16):
            layers2.append(Sep_Conv_DeepLabv3Plus(
                filters=[nf2, nf2, nf2, nf2], stride=1, dilation=1,
                skip_con="sum", dim=dim))

        # Exit flow
        layers2.append(Sep_Conv_DeepLabv3Plus(
            filters=[nf2, nf2, nfi*32, nfi*32], stride=1, dilation=1,
            skip_con="conv", dim=dim))
        
        layers2.append(Sep_Conv_DeepLabv3Plus(
            filters=[nfi*32, nfi*48, nfi*48, nfi*64], stride=1, dilation=2,
            skip_con="none", dim=dim))

        self.xception_part2 = nn.Sequential(*layers2)

    def forward(self, x):
        x1, skip_connection = self.xception_part1(x)
        x2 = self.xception_part2(x1)
        return x2, skip_connection

