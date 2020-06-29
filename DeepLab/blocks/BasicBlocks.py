import torch
from torch import nn
from torch.nn.functional import interpolate, softmax

class Interpolate(nn.Module):
    def __init__(self):
        """
        """
        super(Interpolate, self).__init__()

    def forward(self, input, size=None, scale_factor=None, mode="nearest", align_corners=None):
        return interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

class Softmax(nn.Module):
    def __init__(self):
        """
        """
        super(Softmax, self).__init__()

    def forward(self, input, dim=None, _stacklevel=3, dtype=None):
        return softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

class Cat(nn.Module):
    def __init__(self):
        """Concatenation
        """
        super(Cat, self).__init__()

    def forward(self, tensors, dim=0, out=None):
        return torch.cat(tensors, dim=dim, out=out)

class Sum(nn.Module):
    def __init__(self):
        """
        """
        super(Sum, self).__init__()

    def forward(self, input, dtype=None):
        out = input[0] + input[1]
        for i in range(2, len(input)):
            out += input[i]
        return out
