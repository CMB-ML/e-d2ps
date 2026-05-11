import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNDouble(nn.Module):
    def __init__(self,
                 mid_channels,
                 out_channels,
                 out_power=1
                 ):
        super().__init__()
        if out_power not in [1,2,1.0,2.0]:
            raise NotImplementedError("Out pow must be either 1 or 2.")
        self.lin1 = nn.LazyLinear(out_features=mid_channels)
        self.act1 = nn.ReLU()
        self.lin2 = nn.LazyLinear(out_features=out_channels)
        self.out_pow = out_power
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        if self.out_pow == 2:
            x = x.square()
        return x


class FCNSingle(nn.Module):
    def __init__(self,
                 out_channels,
                 out_power=1
                 ):
        super().__init__()
        if out_power not in [1,2,1.0,2.0]:
            raise NotImplementedError("Out pow must be either 1 or 2.")
        self.lin1 = nn.LazyLinear(out_features=out_channels)
        self.out_pow = out_power
    
    def forward(self, x):
        x = self.lin1(x)
        if self.out_pow == 2:
            x = x.square()
        return x
