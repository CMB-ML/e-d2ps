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

def make_norm2d(n_channels, kind: str):
    if kind == "bn":
        return nn.BatchNorm2d((n_channels))
    elif kind == "gn":
        for n_groups in (32, 16, 8, 4, 2, 1):
            if n_channels % n_groups == 0 and n_channels // n_groups > 1:
                return nn.GroupNorm(n_groups, n_channels)
        raise ValueError(f"No prescribed group count for {n_channels} channels")
    raise ValueError(f"kind {kind} not recognized when setting group or batch norm.")

class DoubleConv(nn.Module):
    def __init__(self, in_features, out_channels, norm_kind, dropout_rate=0.0, activation='relu'):
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_features, out_channels, 
                      kernel_size=3, padding=1, bias=False))
        if norm_kind is not None:
            layers.append(make_norm2d(out_channels, norm_kind))
        layers.append(get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        layers.append(nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=1, bias=False))
        if norm_kind is not None:
            layers.append(make_norm2d(out_channels, norm_kind))
        layers.append(get_activation(activation))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, norm_kind="gn", dropout_rate=0.0, activation='relu'):
        super().__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, norm_kind, dropout_rate, activation)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class FCNDecoder(nn.Module):
    def __init__(self, in_channels, 
                 mid_channels, 
                 out_channels, 
                 norm_kind,
                 dropout_rate=0.0,
                 activation='relu',
                 pooling='max', 
                 final_activation=None
                 ):
        super().__init__()
        layers = []
        layers.append(get_pooling(pooling))
        layers.append(nn.Flatten())
        bias = not norm_kind is None
        layers.append(nn.Linear(in_channels, mid_channels, bias=bias))

        if norm_kind is not None:
            raise NotImplementedError("Whoops")
            layers.append(nn.BatchNorm1d(mid_channels))

        layers.append(get_activation("gelu"))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(mid_channels, out_channels))

        if final_activation is not None:
            layers.append(final_activation)
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _batch_debug(self, tag, t, layer_repr):
        with torch.no_grad():
            B = t.shape[0]
            flat = t.flatten(1)  # [B, D]
            d01 = (flat[0] - flat[1]).abs().max().item() if B >= 2 else float("nan")
            same01 = torch.equal(flat[0], flat[1]) if B >= 2 else False
            dmin = torch.pdist(flat).min().item() if B >= 2 else float("nan")
            print(f"{tag:>12} {layer_repr} shape={tuple(t.shape)} mean={t.mean().item():.3e} std={t.std().item():.3e} "
                f"d01={d01:.3e} same01={same01} dmin={dmin:.3e}")

    # def forward(self, x):
    #     self._batch_debug("in", x, "None")

    #     for i in range(len(self.fc)):
    #         x = self.fc[i](x)   # pooling
    #         self._batch_debug(f"after {i}", x, self.fc[i].__repr__())

    #     # y = self.fc[1](y)   # flatten
    #     # self._batch_debug("after flatten", y)

    #     # y = self.fc[2](y)   # linear1
    #     # self._batch_debug("after linear1", y)

    #     # for layer in self.fc[3:]:
    #     #     y = layer(y)

    #     return x

class SimpleEncDec(nn.Module):
    def __init__(
                 self,
                 n_ells,
                 n_in_channels,  # Should be the number of detectors
                 final_activation=None,
                 n_init_features=16,
                 n_levels=5,
                 norm_kind_enc="gn",
                 norm_kind_dec=None,
                 dropout_rate=0.0,
                 activation='relu',
                 pooling='avg',
                 note='',
                 ):
        super().__init__()
        self.note = note

        self.in_c = DoubleConv(in_features=n_in_channels,
                               out_channels=n_init_features,
                               norm_kind=norm_kind_enc,
                               dropout_rate=dropout_rate,
                               activation=activation)

        downs = []
        curr_features = n_init_features
        for _ in range(n_levels):
            downs.append(Down(curr_features, 
                              norm_kind=norm_kind_enc,
                              dropout_rate=dropout_rate,
                              activation=activation))
            curr_features *= 2
        self.dns = nn.ModuleList(downs)

        self.out = FCNDecoder(
            in_channels=curr_features,
            mid_channels=curr_features // 2,
            out_channels=n_ells,
            norm_kind=norm_kind_dec,
            dropout_rate=dropout_rate,
            activation=activation,
            pooling=pooling,
            final_activation=final_activation
        )

    def forward(self, x):
        # with torch.no_grad():
        #     print("in:", x.shape)        
        x = self.in_c(x)
        # with torch.no_grad():
        #     print("in_c:", x.shape)
        #     i=0
        for down in self.dns:
            x = down(x)
            # with torch.no_grad():
            #     i += 1
            #     print(f"down{i}:", x.shape)
        # with torch.no_grad():
        #     encoder_out_shape = x.shape
        #     encoder_delta = (x[0]-x[1]).abs().max().item()
        #     encoder_std = x.std().item()
        x = self.out(x)
        # with torch.no_grad():
        #     print("Out", x.shape)
        # with torch.no_grad():
        #     decoder_out_shape = x.shape
        #     decoder_delta = (x[0]-x[1]).abs().max().item()
        #     decoder_std = x.std().item()
        #     print(f"enc shape: {encoder_out_shape} enc_delta: {encoder_delta:.2e} enc_std: {encoder_std:.2e} dec shape: {decoder_out_shape} dec_delta: {decoder_delta:.2e} dec_std: {decoder_std:.2e}")

        return x
