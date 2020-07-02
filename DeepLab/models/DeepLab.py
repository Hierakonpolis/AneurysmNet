import torch
from torch.nn import Conv2d, Conv3d, ConvTranspose3d, ConvTranspose2d
from DeepLab.models.Xception import Xception_DeepLabv3Plus
from DeepLab.blocks.DeepLabBlocks import DeepLabv3_ASPP, DeepLabv3_Head
from DeepLab.blocks.BasicBlocks import Interpolate, Softmax, Cat

# first_filters=32
# dim="2D"


class DeepLabv3Plus(torch.nn.Module):
    params = ["modalities", "n_classes", "first_filters", "dim"]
    def __init__(self, PARS):
        super(DeepLabv3Plus, self).__init__()
        modalities=PARS['modalities']
        n_classes=PARS['n_classes']
        first_filters=PARS['first_filters']
        
        dim=PARS['dim']
        self.interpol_mode = "bilinear" if dim == "2D" else "trilinear"
        # Xception
        self.xception = Xception_DeepLabv3Plus(modalities, first_filters,
                n_classes, dim)

        self.aspp_L = DeepLabv3_ASPP(first_filters*64,
                first_filters*8, [6, 12, 18], dim)

        self.project = DeepLabv3_Head(first_filters*8, n_classes,
                dim, last_layer=False)

        if dim == "2D":
            Conv = Conv2d
            CT=ConvTranspose2d
        elif dim == "3D":
            Conv = Conv3d
            CT=ConvTranspose3d
        
        self.interp1=CT(first_filters*16, first_filters*16, kernel_size=4,stride=2,padding=1)
        self.interp2=CT(first_filters*16, first_filters*16, kernel_size=4,stride=2,padding=1)
        
        self.last_conv = Conv(first_filters*16, n_classes, 3, padding=1)

        self.cat = Cat()
        self.interpolate = Interpolate()
        
        
        
        self.softmax = Softmax()

    def forward(self, x, x1):
        x=torch.cat((x,x1),dim=1).cuda()
        out, skip = self.xception(x)
        out = self.aspp_L(out)
        out = self.project(out)
        out = self.interpolate(out, size=list(skip.size()[2:]),
                mode=self.interpol_mode, align_corners=False)
        out = self.cat([out, skip], dim=1)
        
        out = self.interp1(out,output_size=torch.tensor(x.shape[2:])/2)
        out = self.interp2(out,output_size=x.shape[2:])
        
        out = self.last_conv(out)
        
        out = self.softmax(out, dim=1)
        return [], out 

