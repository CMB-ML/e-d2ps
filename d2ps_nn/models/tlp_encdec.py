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
                 down_reduction):
        super().__init__()
        out_channels = in_channels * 2

        self.pool = get_downsample_pooling(down_reduction)
        self.dconv = DoubleConv(in_channels, out_channels, norm_kind, dropout_rate, activation)

    def forward(self, x):
        x = self.pool(x)
        x = self.dconv(x)
        return x


def expand_down_reduction_spec(spec: str, n_levels: int) -> list[str]:
    if not isinstance(spec, str):
        raise TypeError(f"feature_extraction spec must be a string; got {type(spec)}")

    spec = spec.strip().lower()

    # Handle full-word cases
    if spec in {"mean", "avg"}:
        return ["avg"] * n_levels
    if spec == "max":
        return ["max"] * n_levels

    # Compact spec using m/x
    if any(ch not in {"m", "x"} for ch in spec):
        raise ValueError(
            f"Invalid feature_extraction spec {spec!r}; "
            "allowed: 'm', 'x', or full strings 'mean', 'avg', 'max'"
        )

    if len(spec) > n_levels:
        raise ValueError(
            f"feature_extraction spec {spec!r} has length {len(spec)} > n_levels={n_levels}"
        )

    # Pad with mean
    spec = spec + "m" * (n_levels - len(spec))

    mapping = {
        "m": "avg",
        "x": "max",
    }

    return [mapping[ch] for ch in spec]


class FaceEncoder(nn.Module):
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


def _spatial_avg(x):
    return x.mean(dim=(-2, -1))


def _spatial_max(x):
    return x.amax(dim=(-2, -1))


class BaseAggregator(nn.Module):
    def __init__(self, n_features: int, spatial_reduction: str = "avg"):
        """
        Base class

        spatial_pool reduces each face spatially: (B, F, D, H, W) -> (B, F, D)
        """
        super().__init__()
        self.spatial_pool = spatial_reduction
        if spatial_reduction == "avg":
            self.reduce_spatial = _spatial_avg
        elif spatial_reduction == "max":
            self.reduce_spatial = _spatial_max
        else:
            raise ValueError(f"Unknown spatial_pool: {spatial_reduction}")
        self.num_agg_out_features = n_features

    def set_trainable(self, trainable: bool = True):
        for p in self.parameters():
            p.requires_grad = trainable


class MeanAggregator(BaseAggregator):
    def forward(self, x):
        x = self.reduce_spatial(x)                      # (B, F, D)
        return x.mean(dim=1)                            # (B, D)


class MaxAggregator(BaseAggregator):
    def forward(self, x):
        x = self.reduce_spatial(x)                      # (B, F, D)
        return x.amax(dim=1)                            # (B, D)


class MeanStdvAggregator(BaseAggregator):
    def __init__(self, n_in_channels: int, spatial_pool: str = "avg"):
        super().__init__(n_features=n_in_channels, spatial_reduction=spatial_pool)
        self.num_agg_out_features = n_in_channels * 2

    def forward(self, x):
        x = self.reduce_spatial(x)                      # (B, F, D)
        mean = x.mean(dim=1)                            # (B, D)
        stdv = x.std(dim=1, unbiased=False)             # (B, D)
        return torch.cat([mean, stdv], dim=-1)          # (B, 2D)


class FaceWeightedMeanAggregator(BaseAggregator):
    """
    Input-independent learnable per-face weighting.

    logits=0 at init -> uniform softmax -> exact mean over faces.
    """
    def __init__(self, n_in_channels: int, spatial_pool: str = "avg"):
        super().__init__(n_features=n_in_channels, spatial_reduction=spatial_pool)
        self.logits = nn.Parameter(torch.empty(12))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.logits)

    def forward(self, x):
        if x.shape[1] != 12:
            raise ValueError(f"Expected 12 faces, got {x.shape[1]}")
        x = self.reduce_spatial(x)                      # (B, F, D)
        weights = torch.softmax(self.logits, dim=0)     # (F,)
        return (x * weights[None, :, None]).sum(dim=1)  # (B, D)


class SoftmaxAggregator(BaseAggregator):
    """
    Feature-wise softmax pooling over faces after spatial reduction.
    AKA a first-order, linear approximation of log-sum-exp pooling,
    which is bounded by max(x) whereas log-sum-exp(x) is not.

    weights = softmax(alpha * x, dim=1)

    alpha = 0    -> exact mean
    alpha -> inf -> max-like over faces
    """
    def __init__(
        self,
        n_in_channels: int,
        spatial_pool: str = "avg",
        init_alpha: float = 0.0,
    ):
        super().__init__(n_features=n_in_channels, spatial_reduction=spatial_pool)

        self.alpha = nn.Parameter(torch.empty(()))
        self.init_alpha = float(init_alpha)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.alpha.fill_(self.init_alpha)

    def forward(self, x):
        x = self.reduce_spatial(x)                      # (B, F, D)
        weights = torch.softmax(self.alpha * x, dim=1)  # (B, F, D)
        return (weights * x).sum(dim=1)                 # (B, D)


def build_aggregator(
    face_reduction: str,
    n_in_channels: int,
    spatial_reduction: str = "avg",
    **kwargs,
) -> BaseAggregator:
    face_reduction = face_reduction.lower()

    if face_reduction == "avg":
        return MeanAggregator(n_features=n_in_channels, 
                              spatial_reduction=spatial_reduction)

    elif face_reduction == "max":
        return MaxAggregator(n_features=n_in_channels, 
                             spatial_reduction=spatial_reduction)

    elif face_reduction == "meanstdv":
        return MeanStdvAggregator(n_in_channels=n_in_channels, 
                                  spatial_pool=spatial_reduction, 
                                  **kwargs)

    elif face_reduction == "weighted":
        raise NotImplementedError("This reduction is incompatible with model loading.")
        #TODO
        return FaceWeightedMeanAggregator(n_in_channels=n_in_channels, 
                                          spatial_pool=spatial_reduction)

    elif face_reduction == "softmax":
        return SoftmaxAggregator(n_in_channels=n_in_channels, 
                                 spatial_pool=spatial_reduction, 
                                 **kwargs)

    else:
        raise ValueError(f"Unknown aggregator: {face_reduction}")


class FCNDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 out_channels, 
                 dropout_rate,
                 activation,
                 final_activation
                 ):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels, bias=True)
        self.act1 = get_activation(activation)
        self.drop = get_dropout(dropout_rate)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.act2 = get_activation(final_activation)


    def forward(self, x):
        x = self.mlp1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.mlp2(x)
        x = self.act2(x)
        return x


def check_main_input(x: torch.Tensor):
    if x.ndim != 5:
        raise ValueError(
            f"Expected input shape (B, 12, C, H, W); got {tuple(x.shape)}"
        )
    B, F, C, H, W = x.shape
    if F != 12:
        raise ValueError(
            f"Expected 12 faces; got {x.shape[1]}"
        )


class TLPEncAggDec(nn.Module):
    def __init__(
                 self,
                 nside,
                 n_ells,
                 n_in_channels,  # Should be the number of detectors
                 final_activation=None,
                 n_init_features=16,
                 n_levels=7,
                 norm_kind='gn',
                 dropout_rate=0.0,
                 activation='relu',
                 down_reductions='avg',
                 spatial_reduction='avg',
                 face_reduction='avg',
                 note=''
                 ):
        super().__init__()
        self.nside = nside
        self.example_in_shape = (12, n_in_channels, nside, nside)
        self.note = note

        # No longer the case
        # if nside != 2**n_levels:
        #     raise ValueError("This model assumes full compression to 1x1 px after Encoder")

        self.enc = FaceEncoder(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            n_levels=n_levels,
            norm_kind=norm_kind,
            dropout_rate=dropout_rate,
            activation=activation,
            down_reduction_spec=down_reductions
        )

        self.agg: BaseAggregator = build_aggregator(
            face_reduction, 
            n_in_channels=self.enc.num_enc_out_features, 
            spatial_reduction=spatial_reduction
            )

        self.dec = FCNDecoder(
            in_channels=self.agg.num_agg_out_features,
            mid_channels=self.agg.num_agg_out_features // 2,
            out_channels=n_ells,
            dropout_rate=dropout_rate,
            activation=activation,
            final_activation=final_activation
        )

    def forward(self, x: torch.Tensor):
        check_main_input(x)
        B, F, C, H, W = x.shape           # (B, F, C, H, W)
        x = x.reshape(B * 12, C, H, W)    # (B * F, C, H, W)
        z = self.enc(x)
        _, D, H_, W_ = z.shape
        z = z.reshape(B, 12, D, H_, W_)   # (B, F, D, H', W')
        z = self.agg(z)                   # (B, D)
        y = self.dec(z)                   # (B, n_ells)
        return y


class TLPEnc2AggDec(nn.Module):
    def __init__(
                 self,
                 nside,
                 n_ells,
                 n_in_channels,  # Should be the number of detectors
                 final_activation=None,
                 n_init_features=16,
                 n_levels=7,
                 norm_kind='gn',
                 dropout_rate=0.0,
                 activation='relu',
                 down_reductions='avg',
                 spatial_reduction='avg',
                 face_reduction='avg',
                 note=''
                 ):
        super().__init__()
        self.nside = nside
        self.example_in_shape = (12, n_in_channels, nside, nside)

        self.note = note

        # No longer the case
        # if nside != 2**n_levels:
        #     raise ValueError("This model assumes full compression to 1x1 px after Encoder")

        # Use separate encoders for equatorial and polar TLPs, as those classes have different disortion
        face_encoder_params = dict(
            n_in_channels = n_in_channels,
            n_init_features=n_init_features,
            n_levels=n_levels,
            norm_kind=norm_kind,
            dropout_rate=dropout_rate,
            activation=activation,
            down_reduction_spec=down_reductions
        )
        self.enc_polar = FaceEncoder(**face_encoder_params)
        self.enc_equatorial = FaceEncoder(**face_encoder_params)

        self.agg: BaseAggregator = build_aggregator(
            face_reduction, 
            n_in_channels=self.enc_polar.num_enc_out_features, 
            spatial_reduction=spatial_reduction
            )

        self.dec = FCNDecoder(
            in_channels=self.agg.num_agg_out_features,
            mid_channels=self.agg.num_agg_out_features // 2,
            out_channels=n_ells,
            dropout_rate=dropout_rate,
            activation=activation,
            final_activation=final_activation
        )
        self.register_buffer(
            "polar_faces",
            torch.tensor([0, 1, 2, 3, 8, 9, 10, 11], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "equatorial_faces",
            torch.tensor([4, 5, 6, 7], dtype=torch.long),
            persistent=False,
        )

    def forward(self, x):
        check_main_input(x)
        B, F, C, H, W = x.shape

        # F = 12 = 8 + 4; (8 polar top-level pixels, 4 equatorial)
        x_polar = x[:, self.polar_faces].reshape(B * 8, C, H, W)
        x_equat = x[:, self.equatorial_faces].reshape(B * 4, C, H, W)

        z_polar = self.enc_polar(x_polar)      # (B*8, D, H', W')
        z_equat = self.enc_equatorial(x_equat) # (B*4, D, H', W')

        if z_polar.shape[1:] != z_equat.shape[1:]:
            raise ValueError(
                f"Polar/equatorial encoder outputs differ: "
                f"{z_polar.shape} vs {z_equat.shape}"
            )

        _, D, H_, W_ = z_polar.shape

        z_polar = z_polar.reshape(B, 8, D, H_, W_)
        z_equat = z_equat.reshape(B, 4, D, H_, W_)

        # The face order will have changed. 
        # The subsequent agg will reduce over face order consistently.
        z = torch.cat([z_polar, z_equat], dim=1)   # (B, F, D, H', W')

        z = self.agg(z)                            # (B, D)
        y = self.dec(z)                            # (B, n_ells)
        return y
