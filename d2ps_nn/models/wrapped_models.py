from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Union

import torch
import torch.nn as nn

from d2ps_nn.models.polynomial_layers import BaseSpectrumLayer
from d2ps_nn.utils.scale_coef import Processor, IdentityProcessor


class WrappedSpectrumModel(nn.Module):
    """
    Thin composition wrapper:

        input -> backbone -> coefs_proc -> coeff_processor.postprocess(...)
              -> coefs_raw -> spectrum_layer -> spectrum

    The backbone is expected to return shape (B, n_coefs).
    The spectrum_layer maps (B, n_coefs) -> (B, n_ell).
    """

    def __init__(
        self,
        backbone_layer: nn.Module,
        spectrum_layer: BaseSpectrumLayer,
        coef_processor: Optional[Processor] = None,
        return_mode: str = "spectrum",
    ) -> None:
        """
        Parameters
        ----------
        backbone
            Model returning coefficients, shape (B, n_coefs).
        spectrum_layer
            Deterministic layer mapping coefs -> spectrum.
        return_mode
            One of:
            - "spectrum"
            - "coefs"
            - "both"
        """
        super().__init__()

        return_mode = return_mode.lower()
        if return_mode not in {"spectrum", "coefs", "both"}:
            raise ValueError(
                f"Unknown return_mode={return_mode!r}. "
                f"Expected 'spectrum', 'coefs', or 'both'."
            )

        self.backbone_layer = backbone_layer
        self.spectrum_layer = spectrum_layer
        self.coef_processor = coef_processor or IdentityProcessor()
        self.return_mode = return_mode

        self.example_in_shape = self.backbone_layer.example_in_shape

    def forward(self, x: torch.Tensor):
        coefs_scl = self.backbone_layer(x)

        if self.return_mode == "coefs":
            return coefs_scl

        coefs_raw = self.coef_processor.postprocess(coefs_scl)
        spectrum = self.spectrum_layer(coefs_raw)

        if self.return_mode == "spectrum":
            return spectrum

        return {
            "coefs_scl": coefs_scl,
            "coefs_raw": coefs_raw,
            "spectrum": spectrum,
        }

    def post(self, coef_scl):
        coef_raw = self.coef_processor.postprocess(coef_scl)
        return coef_raw

    def to_ps(self, coeff_scl):
        coeff_raw = self.post(coeff_scl)
        return self.spectrum_layer(coeff_raw)
