"""
Written using ChatGPT with instructions to produce the simplest implementation of a UNet model,
with parameterized depth (for simplicity when debugging: a very shallow UNet can be produced).

Some modifications were made.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_factories import (
    get_activation,
    get_dropout,
    get_norm,
    get_downsample_pooling,
    get_bottleneck_pooling
)


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_kind,
                 dropout_rate,
                 activation):
        super().__init__()

        norm_kind = 'none' if norm_kind is None else norm_kind.lower()
        use_bias  = norm_kind == 'none'

        self.conv1  = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=3, padding=1, bias=use_bias)
        self.norm1  = get_norm(norm_kind, out_channels)
        self.act_1  = get_activation(activation)
        self.drop1  = get_dropout(dropout_rate)
        self.conv2  = nn.Conv2d(out_channels, out_channels, 
                          kernel_size=3, padding=1, bias=use_bias)
        self.norm2  = get_norm(norm_kind, out_channels)
        self.act_2  = get_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act_1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act_2(x)
        return x


class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_kind,
                 dropout_rate,
                 activation,
                 pool_kind):
        super().__init__()
        out_channels = in_channels * 2

        self.pool = get_downsample_pooling(pool_kind)
        self.dconv = DoubleConv(in_channels, out_channels, norm_kind, dropout_rate, activation)

    def forward(self, x):
        x = self.pool(x)
        x = self.dconv(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 n_in_channels,
                 n_init_features,
                 n_levels,
                 norm_kind,
                 dropout_rate,
                 activation,
                 downsample_pooling_kind):
        super().__init__()
        self.in_c = DoubleConv(in_channels=n_in_channels,
                               out_channels=n_init_features,
                               norm_kind=norm_kind,
                               dropout_rate=dropout_rate,
                               activation=activation)

        downs = []
        curr_features = n_init_features
        for _ in range(n_levels):
            downs.append(Down(curr_features, 
                              norm_kind=norm_kind,
                              dropout_rate=dropout_rate,
                              activation=activation,
                              pool_kind=downsample_pooling_kind))
            curr_features *= 2
        self.dns = nn.ModuleList(downs)
        self.bottleneck_features = curr_features

    def forward(self, x):
        x = self.in_c(x)
        for down in self.dns:
            x = down(x)
        return x


class FCNDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 out_channels, 
                 dropout_rate,
                 activation,
                 bottleneck_pooling,
                 final_activation):
        super().__init__()

        # norm_kind = 'none' if norm_kind is None else norm_kind.lower()
        # use_bias  = norm_kind == 'none'

        self.pool = get_bottleneck_pooling(bottleneck_pooling)
        self.flat = nn.Flatten()
        self.mlp1 = nn.Linear(in_channels, mid_channels, bias=True)
        self.act1 = get_activation(activation)
        self.drop = get_dropout(dropout_rate)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.act2 = get_activation(final_activation)

    def forward(self, x):
        x = self.pool(x)
        x = self.flat(x)
        x = self.mlp1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.mlp2(x)
        x = self.act2(x)
        return x

class SimpleEncDec(nn.Module):
    def __init__(
                 self,
                 nside,          # Unused, for compatibility with TLP models
                 n_ells,
                 n_in_channels,  # Should be the number of detectors
                 n_init_features=16,
                 n_levels=5,
                 norm_kind=None,
                 dropout_rate=0.0,
                 downsample_pooling='avg',
                 bottleneck_pooling='avg',
                 activation='relu',
                 final_activation=None,
                 note='',
                 ):
        super().__init__()
        self.nside = nside
        self.example_in_shape = (n_in_channels, 4*nside, 3*nside)

        self.note = note

        self.enc = Encoder(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            norm_kind=norm_kind,
            dropout_rate=dropout_rate,
            activation=activation,
            n_levels=n_levels,
            downsample_pooling_kind=downsample_pooling
        )

        self.dec = FCNDecoder(
            in_channels=self.enc.bottleneck_features,
            mid_channels=self.enc.bottleneck_features // 2,
            out_channels=n_ells,
            dropout_rate=dropout_rate,
            activation=activation,
            bottleneck_pooling=bottleneck_pooling,
            final_activation=final_activation
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
