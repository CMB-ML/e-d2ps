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


class FaceEncoder(nn.Module):
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


class Aggregator(nn.Module):
    def forward(self, x):
        # This assumes the Encoder has fully compressed each face (H,W==1)
        # x: (B, F, D, 1, 1)
        x = x.squeeze(-1).squeeze(-1)   # (B, F, D)
        x = x.mean(dim=1)               # (B, D)
        return x


class FCNDecoder(nn.Module):
    def __init__(self, in_channels, 
                 mid_channels, 
                 out_channels, 
                 use_batch_norm,
                 dropout_rate=0.0,
                 activation='relu',
                 final_activation=None
                 ):
        super().__init__()
        layers = []
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


class TLPEncAggDec(nn.Module):
    def __init__(
                 self,
                 nside,
                 n_outputs,
                 n_in_channels,  # Should be the number of detectors
                 final_activation=None,
                 n_init_features=16,
                 n_levels=7,
                 use_batch_norm_enc=True,
                 use_batch_norm_dec=False,
                 dropout_rate=0.0,
                 activation='relu',
                 note=''
                 ):
        super().__init__()
        self.nside = nside
        self.example_in_shape = (12, n_in_channels, nside, nside)
        self.note = note

        if nside != 2**n_levels:
            raise ValueError("This model assumes full compression to 1x1 px after Encoder")

        self.enc = FaceEncoder(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            use_batch_norm_enc=use_batch_norm_enc,
            dropout_rate=dropout_rate,
            activation=activation,
            n_levels=n_levels
        )

        self.agg = Aggregator()

        self.dec = FCNDecoder(
            in_channels=self.enc.bottleneck_features,
            mid_channels=self.enc.bottleneck_features // 2,
            out_channels=n_outputs,
            use_batch_norm=use_batch_norm_dec,
            dropout_rate=dropout_rate,
            activation=activation,
            final_activation=final_activation
        )

    def forward(self, x):
        check_main_input(x)
        face_embeddings = []
        for f in range(12):
            z_f = self.enc(x[:, f])              # x[:, f] -> (B, C, H, W)
            face_embeddings.append(z_f)
        z = torch.stack(face_embeddings, dim=1)  # (B, F, D, 1, 1)
        check_features(z)
        z = self.agg(z)                          # (B, D)
        y = self.dec(z)                          # (B, lmax)
        return y


def check_main_input(x):
    if x.ndim != 5:
        raise ValueError(
            f"Expected input shape (B, 12, C, H, W); got {tuple(x.shape)}"
        )
    B, F, C, H, W = x.shape
    if F != 12:
        raise ValueError(
            f"Expected 12 faces; got {x.shape[1]}"
        )


def check_features(z):
    if z.shape[-2:] != (1, 1):
        raise RuntimeError(
            f"Expected encoder output spatial shape (1,1), got {z.shape[-2:]}"
        )


class TLPEnc2AggDec(nn.Module):
    def __init__(
                 self,
                 nside,
                 n_outputs,
                 n_in_channels,  # Should be the number of detectors
                 final_activation=None,
                 n_init_features=16,
                 n_levels=7,
                 use_batch_norm_enc=True,
                 use_batch_norm_dec=False,
                 dropout_rate=0.0,
                 activation='relu',
                 note=''
                 ):
        super().__init__()
        self.nside = nside
        self.example_in_shape = (12, n_in_channels, nside, nside)

        self.note = note

        if nside != 2**n_levels:
            raise ValueError("This model assumes full compression to 1x1 px after Encoder")

        # Use separate encoders for equatorial and polar TLPs, as those classes have different disortion
        face_encoder_params = dict(
            n_in_channels=n_in_channels,
            n_init_features=n_init_features,
            use_batch_norm_enc=use_batch_norm_enc,
            dropout_rate=dropout_rate,
            activation=activation,
            n_levels=n_levels
        )
        self.enc_polar = FaceEncoder(**face_encoder_params)
        self.enc_equatorial = FaceEncoder(**face_encoder_params)

        self.agg = Aggregator()

        self.dec = FCNDecoder(
            in_channels=self.enc_polar.bottleneck_features,
            mid_channels=self.enc_polar.bottleneck_features // 2,
            out_channels=n_outputs,
            use_batch_norm=use_batch_norm_dec,
            dropout_rate=dropout_rate,
            activation=activation,
            final_activation=final_activation
        )

    def forward(self, x):
        check_main_input(x)
        face_embeddings = []
        for f in range(12):
            z_f = self.get_face_enc(f)(x[:, f])  # x[:, f] -> (B, C, H, W)
            face_embeddings.append(z_f)
        z = torch.stack(face_embeddings, dim=1)  # (B, F, D, 1, 1)
        check_features(z)
        z = self.agg(z)                          # (B, D)
        y = self.dec(z)                          # (B, lmax)
        return y

    def get_face_enc(self, f):
        if f in [4,5,6,7]:
            return self.enc_equatorial
        else:
            return self.enc_polar
