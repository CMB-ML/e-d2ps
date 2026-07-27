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
    get_dropout_1d,
    get_dropout_2d,
    get_norm,
    get_downsample_pooling,
    get_spatial_reduction
)
from .down_reduction_spec import expand_down_reduction_spec


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_kind,
                 activation):
        super().__init__()

        norm_kind = 'none' if norm_kind is None else norm_kind.lower()
        use_bias  = norm_kind == 'none'

        self.conv1  = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=3, padding=1, bias=use_bias)
        self.norm1  = get_norm(norm_kind, out_channels)
        self.act_1  = get_activation(activation)
        self.conv2  = nn.Conv2d(out_channels, out_channels, 
                          kernel_size=3, padding=1, bias=use_bias)
        self.norm2  = get_norm(norm_kind, out_channels)
        self.act_2  = get_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act_1(x)
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
                 down_reduction,
                 dropout_position='post_pool',
                 drop_type='drop2d'):
        super().__init__()

        if dropout_position not in ('pre_pool', 'post_pool'):
            raise ValueError(
                "dropout_position must be 'pre_pool' or 'post_pool', "
                f"got {dropout_position!r}"
            )
        if drop_type not in ('drop2d', 'drop1d'):
            raise ValueError(
                "drop_type must be 'drop2d' or 'drop1d', "
                f"got {drop_type!r}"
            )

        out_channels = in_channels * 2
        self.pool = get_downsample_pooling(down_reduction)
        self.dconv = DoubleConv(in_channels, out_channels, norm_kind, activation)
        if drop_type == 'drop2d':
            self.drop = get_dropout_2d(dropout_rate)
        else:
            # To be considered for the case of pre_pool
            self.drop = get_dropout_1d(dropout_rate)
        self.dropout_position = dropout_position

    def forward(self, x):
        if self.dropout_position == 'pre_pool':
            x = self.drop(x)
            x = self.pool(x)
        else:  # post_pool
            x = self.pool(x)
            x = self.drop(x)
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
                 down_reduction_spec,
                 dropout_position='post_pool',
                 drop_type='drop2d'):
        super().__init__()
        down_reductions = expand_down_reduction_spec(
            down_reduction_spec,
            n_levels
        )

        if isinstance(dropout_rate, list):
            expected_len = len(down_reductions)

            if len(dropout_rate) != expected_len:
                raise ValueError(
                    f"Encoder dropout_rate list must have length {expected_len}: "
                    f"{len(down_reductions)} down blocks. "
                    f"Got {len(dropout_rate)}."
                )

            down_dropouts = dropout_rate
        else:
            down_dropouts = [dropout_rate] * len(down_reductions)

        self.in_c = DoubleConv(
            in_channels=n_in_channels,
            out_channels=n_init_features,
            norm_kind=norm_kind,
            activation=activation
        )
        
        curr_features = n_init_features
        downs = []

        for dr, drop_rate in zip(down_reductions, down_dropouts):
            downs.append(
                Down(
                    curr_features,
                    norm_kind=norm_kind,
                    dropout_rate=drop_rate,
                    activation=activation,
                    down_reduction=dr,
                    dropout_position=dropout_position,
                    drop_type=drop_type
                )
            )
            curr_features *= 2

        self.dns = nn.ModuleList(downs)
        self.num_enc_out_features = curr_features

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
                 spatial_reduction,
                 final_activation):
        super().__init__()
        if len(dropout_rate) != 2:
            raise ValueError(
                f"FCNDecoder dropout_rate must contain 2 rates, got {len(dropout_rate)}."
            )
        
        self.pool = get_spatial_reduction(spatial_reduction)
        self.feature_drop = get_dropout_1d(dropout_rate[0])
        self.flat = nn.Flatten()
        self.mlp1 = nn.Linear(in_channels, mid_channels, bias=True)
        self.act1 = get_activation(activation)
        self.hidden_drop = get_dropout_1d(dropout_rate[1])
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.act2 = get_activation(final_activation)

    def forward(self, x):
        x = self.pool(x)
        x = self.flat(x)
        x = self.feature_drop(x)
        x = self.mlp1(x)
        x = self.act1(x)
        x = self.hidden_drop(x)
        x = self.mlp2(x)
        x = self.act2(x)
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
                 final_activation=None,
                 enc_dropout_pos=None,
                 enc_drop_type=None,
                 note=''
                 ):
        super().__init__()
        self.nside = nside
        self.example_in_shape = (n_in_channels, 4*nside, 3*nside)

        self.note = note

        if enc_dropout_pos is None:
            raise ValueError(f"enc_dropout_pos should be either 'post_pool' or 'pre_pool', got {enc_dropout_pos}")
        if enc_drop_type is None:
            raise ValueError(f"enc_drop_type should be either 'drop2d' or 'drop1d', got {enc_drop_type}")

        if isinstance(dropout_rate, list):
            expected_len = n_levels + 2

            if len(dropout_rate) != expected_len:
                raise ValueError(
                    f"dropout_rate list must have length {expected_len} "
                    f"for n_levels={n_levels}: "
                    f"{n_levels} encoder rates + 2 decoder rate. "
                    f"Got {len(dropout_rate)}."
                )

            enc_dropout = dropout_rate[:-2]
            dec_dropout = dropout_rate[-2:]
        else:
            enc_dropout = dropout_rate
            dec_dropout = dropout_rate[dropout_rate, dropout_rate]

        self.enc = Encoder(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            n_levels=n_levels,
            norm_kind=norm_kind,
            dropout_rate=enc_dropout,
            activation=activation,
            down_reduction_spec=down_reductions,
            dropout_position=enc_dropout_pos,
            drop_type=enc_drop_type,
        )

        self.dec = FCNDecoder(
            in_channels=self.enc.num_enc_out_features,
            mid_channels=self.enc.num_enc_out_features // 2,
            out_channels=n_outputs,
            dropout_rate=dec_dropout,
            activation=activation,
            spatial_reduction=spatial_reduction,
            final_activation=final_activation
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
