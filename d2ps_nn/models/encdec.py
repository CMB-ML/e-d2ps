"""
Written using ChatGPT with instructions to produce the simplest implementation of a UNet model,
with parameterized depth (for simplicity when debugging: a very shallow UNet can be produced).

Some modifications were made.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    """Return activation module based on string name."""
    if activation is None:
        raise ValueError("Activation cannot be None. Please specify 'relu' or 'gelu' in config.")
    
    activation = activation.lower()
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {activation}. Choose from: relu, gelu.")

def get_pooling(pooling):
    """Return pooling module based on string name"""
    if pooling is None:
        raise ValueError("Please specify 'avg' or 'max' pooling.")
    pooling = pooling.lower()
    if pooling == 'avg':
        return nn.AdaptiveAvgPool2d((1,1))
    elif pooling == 'max':
        return nn.AdaptiveMaxPool2d((1,1))
    else:
        raise ValueError(f"Unknowing pooling: {pooling}. Choose from: avg, max.")

class DoubleConv(nn.Module):
    def __init__(self, in_features, out_channels, use_batch_norm, dropout_rate=0.0, activation='relu'):
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_features, out_channels, 
                      kernel_size=3, padding=1, bias=False))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        layers.append(nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=1, bias=False))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(activation))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, use_batch_norm, dropout_rate=0.0, activation='relu'):
        super().__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, use_batch_norm, dropout_rate, activation)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 n_in_channels,
                 n_init_features,
                 use_batch_norm_enc,
                 dropout_rate,
                 activation,
                 n_levels):
        super().__init__()
        self.in_c = DoubleConv(in_features=n_in_channels,
                               out_channels=n_init_features,
                               use_batch_norm=use_batch_norm_enc,
                               dropout_rate=dropout_rate,
                               activation=activation)

        downs = []
        curr_features = n_init_features
        for _ in range(n_levels):
            downs.append(Down(curr_features, 
                              use_batch_norm=use_batch_norm_enc,
                              dropout_rate=dropout_rate,
                              activation=activation))
            curr_features *= 2
        self.dns = nn.ModuleList(downs)
        self.bottleneck_features = curr_features

    def forward(self, x):
        x = self.in_c(x)
        for down in self.dns:
            x = down(x)
        return x


class FCNDecoder(nn.Module):
    def __init__(self, in_channels, 
                 mid_channels, 
                 out_channels, 
                 use_batch_norm,
                 dropout_rate=0.0,
                 activation='relu',
                 pooling='max', 
                 final_activation=None
                 ):
        super().__init__()
        layers = []
        layers.append(get_pooling(pooling))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, mid_channels, bias=not use_batch_norm))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(mid_channels))

        layers.append(get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(mid_channels, out_channels))

        if final_activation is not None:
            layers.append(final_activation)
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleEncDec(nn.Module):
    def __init__(
                 self,
                 nside,          # Unused, for compatibility with TLP models
                 n_ells,
                 n_in_channels,  # Should be the number of detectors
                 final_activation=None,
                 n_init_features=16,
                 n_levels=5,
                 use_batch_norm_enc=True,
                 use_batch_norm_dec=False,
                 dropout_rate=0.0,
                 activation='relu',
                 pooling='avg',
                 note='',
                 ):
        super().__init__()
        self.in_shape = (n_in_channels, 4*nside, 3*nside)
        self.note = note

        self.enc = Encoder(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            use_batch_norm_enc=use_batch_norm_enc,
            dropout_rate=dropout_rate,
            activation=activation,
            n_levels=n_levels
        )

        self.dec = FCNDecoder(
            in_channels=self.enc.bottleneck_features,
            mid_channels=self.enc.bottleneck_features // 2,
            out_channels=n_ells,
            use_batch_norm=use_batch_norm_dec,
            dropout_rate=dropout_rate,
            activation=activation,
            pooling=pooling,
            final_activation=final_activation
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
