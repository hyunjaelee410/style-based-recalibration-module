import torch.nn as nn
import functools
import math
import torch
from torch.nn.parameter import Parameter

class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, -1)
        channel_std = x.view(N, C, -1).std(dim=2).view(N, C, -1)

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t 
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()

        self.reduction = reduction

        self.fc = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // self.reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_y = self.avgpool(x).view(b, c)

        gate = self.fc(avg_y).view(b, c, 1, 1)
        gate = self.activation(gate)

        return x * gate 

class GELayer(nn.Module):
    def __init__(self, channel, layer_idx):
        super(GELayer, self).__init__()

        # Kernel size w.r.t each layer for global depth-wise convolution
        kernel_size = [-1, 56, 28, 14, 7][layer_idx]

        self.conv = nn.Sequential(
                        nn.Conv2d(channel, channel, kernel_size=kernel_size, groups=channel), 
                        nn.BatchNorm2d(channel),
                    )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        gate = self.conv(x)
        gate = self.activation(gate)

        return x * gate 
