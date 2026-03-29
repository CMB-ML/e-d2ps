from typing import Optional
import torch.nn as nn


def get_activation(kind: Optional[str]) -> nn.Module:
    """Return an activation module from a string spec."""
    if kind is None:
        kind = "none"

    kind = kind.lower()

    if kind == "relu":
        return nn.ReLU(inplace=True)
    elif kind == "gelu":
        return nn.GELU()
    elif kind == "softplus":  # For final activation.
        return nn.Softplus()
    elif kind == "none":
        return nn.Identity()    
    else:
        raise ValueError(f"Unknown activation '{kind}'. Choose from: 'relu', 'gelu', 'softplus' and None.")


def get_downsample_pooling(kind: str) -> nn.Module:
    """Return a global adaptive pooling module from a string spec."""
    if kind is None:
        raise ValueError("Pooling must be specified. Choose from: 'avg', 'max'.")

    kind = kind.lower()

    if kind == "avg":
        return nn.AvgPool2d(kernel_size=2)
    elif kind == "max":
        return nn.MaxPool2d(kernel_size=2)
    else:
        raise ValueError(f"Unknown pooling '{kind}'. Choose from: 'avg', 'max'.")


def get_bottleneck_pooling(kind: str) -> nn.Module:
    """Return a global adaptive pooling module from a string spec."""
    if kind is None:
        raise ValueError("Pooling must be specified. Choose from: 'avg', 'max'.")

    kind = kind.lower()

    if kind == "avg":
        return nn.AdaptiveAvgPool2d((1, 1))
    elif kind == "max":
        return nn.AdaptiveMaxPool2d((1, 1))
    else:
        raise ValueError(f"Unknown pooling '{kind}'. Choose from: 'avg', 'max'.")


def get_norm(kind: Optional[str], num_channels: int) -> nn.Module:
    """Return a normalization layer or Identity if disabled."""
    if kind is None:
        kind = "none"

    kind = kind.lower()

    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    elif kind == "gn":
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)
    elif kind == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm kind '{kind}'. Choose from: 'bn', 'gn', 'none'.")


def get_dropout(p: Optional[float]) -> nn.Module:
    """Return Dropout2d or Identity if disabled."""
    if p is None or p == 0:
        return nn.Identity()

    if not (0.0 <= p < 1.0):
        raise ValueError(f"Dropout probability must satisfy 0 <= p < 1, got {p}.")

    return nn.Dropout2d(p)
