from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from d2ps_nn.models.layer_factories import (
    get_activation,
    get_dropout
)


class FCNDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 out_channels, 
                 dropout_rate,
                 activation,
                #  final_activation
                 ):
        super().__init__()

        # norm_kind = 'none' if norm_kind is None else norm_kind.lower()
        # use_bias  = norm_kind == 'none'

        self.mlp1 = nn.Linear(in_channels, mid_channels, bias=True)
        self.act1 = get_activation(activation)
        self.drop = get_dropout(dropout_rate)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        # self.act2 = get_activation(final_activation)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.mlp2(x)
        # x = self.act2(x)
        return x


DecoderFactory = Callable[[int, int], nn.Module]


def make_default_decoder_factory(
    dropout_rate: float,
    activation: str,
    # final_activation: Optional[str],
) -> DecoderFactory:
    """Original two-layer FCNDecoder (mid = n_in // 2).

    Used when a model is built without an explicit decoder_factory, so prior
    behavior and configs are preserved exactly.
    """
    def factory(n_in: int, n_out: int) -> nn.Module:
        return FCNDecoder(
            in_channels=n_in,
            mid_channels=n_in // 2,
            out_channels=n_out,
            dropout_rate=dropout_rate,
            activation=activation,
            # final_activation=final_activation,
        )
    return factory
