"""
Written using ChatGPT with instructions to produce the simplest implementation of a UNet model,
with parameterized depth (for simplicity when debugging: a very shallow UNet can be produced).

Modifications made.
"""
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from d2ps_nn.models.layer_factories import (
    get_activation,
    get_dropout,
    get_norm,
    get_downsample_pooling,
    get_spatial_reduction
)
from d2ps_nn.models.down_reduction_spec import expand_down_reduction_spec
from d2ps_fcn.models.generic_fcn_decoder import make_default_decoder_factory, DecoderFactory


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
                 down_reduction):
        super().__init__()
        out_channels = in_channels * 2

        self.pool = get_downsample_pooling(down_reduction)
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
                 down_reduction_spec):
        super().__init__()
        self.in_c = DoubleConv(in_channels=n_in_channels,
                               out_channels=n_init_features,
                               norm_kind=norm_kind,
                               dropout_rate=dropout_rate,
                               activation=activation)

        curr_features = n_init_features
        down_reductions = expand_down_reduction_spec(down_reduction_spec,
                                                     n_levels)
        downs = []
        for dr in down_reductions:
            downs.append(Down(curr_features, 
                              norm_kind=norm_kind,
                              dropout_rate=dropout_rate,
                              activation=activation,
                              down_reduction=dr))
            curr_features *= 2
        self.dns = nn.ModuleList(downs)
        self.num_enc_out_features = curr_features

    def forward(self, x):
        x = self.in_c(x)
        for down in self.dns:
            x = down(x)
        return x


class Aggregator(nn.Module):
    def __init__(self, 
                 n_features,
                 spatial_reduction
                 ):
        super().__init__()

        self.pool = get_spatial_reduction(spatial_reduction)
        self.flat = nn.Flatten()
        self.num_agg_out_features = n_features

    def forward(self, x):
        x = self.pool(x)
        x = self.flat(x)
        return x


class SimpleEncDec(nn.Module):
    def __init__(
                 self,
                 nside,          # Unused, for compatibility with TLP models
                 n_outputs,
                 n_in_channels,  # Should be the number of detectors
                 n_init_features=16,
                 n_levels=7,
                 norm_kind=None,
                 dropout_rate=0.0,
                 down_reductions='avg',
                 spatial_reduction='avg',
                 activation='relu',
                #  final_activation=None,
                 decoder_factory: Optional[DecoderFactory] = None,
                 note=''
                 ):
        super().__init__()
        self.nside = nside

        self.note = note

        self.enc = Encoder(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            n_levels=n_levels,
            norm_kind=norm_kind,
            dropout_rate=dropout_rate,
            activation=activation,
            down_reduction_spec=down_reductions
        )

        self.agg = Aggregator(
            n_features=self.enc.num_enc_out_features,
            spatial_reduction=spatial_reduction
        )

        if decoder_factory is None:
            decoder_factory = make_default_decoder_factory(
                dropout_rate=dropout_rate,
                activation=activation,
                # final_activation=final_activation,
            )
        self.dec = decoder_factory(self.agg.num_agg_out_features, n_outputs)
        
        self.example_in_shape = (n_in_channels, 4*nside, 3*nside)
        self.example_bn_shape = (self.agg.num_agg_out_features, )
        self.example_out_shape = (n_outputs, )

    def forward(self, x):
        x = self.enc(x)
        x = self.agg(x)
        x = self.dec(x)
        return x
