from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Union

import torch
import torch.nn as nn

from d2ps_nn.models.polynomial_layers import BaseSpectrumLayer


class WrappedSpectrumModel(nn.Module):
    """
    Thin composition wrapper:

        input -> backbone -> coeffs -> spectrum_layer -> spectrum

    The backbone is expected to return shape (B, n_coeffs).
    The spectrum_layer maps (B, n_coeffs) -> (B, n_ell).
    """

    def __init__(
        self,
        backbone_layer: nn.Module,
        spectrum_layer: BaseSpectrumLayer,
        return_mode: str = "spectrum",
    ) -> None:
        """
        Parameters
        ----------
        backbone
            Model returning coefficients, shape (B, n_coeffs).
        spectrum_layer
            Deterministic layer mapping coeffs -> spectrum.
        return_mode
            One of:
            - "spectrum"
            - "coeffs"
            - "both"
        """
        super().__init__()

        return_mode = return_mode.lower()
        if return_mode not in {"spectrum", "coeffs", "both"}:
            raise ValueError(
                f"Unknown return_mode={return_mode!r}. "
                f"Expected 'spectrum', 'coeffs', or 'both'."
            )

        self.backbone_layer = backbone_layer
        self.spectrum_layer = spectrum_layer
        self.return_mode = return_mode

        self.example_in_shape = self.backbone_layer.example_in_shape

    def forward(self, x: torch.Tensor):
        coeffs = self.backbone_layer(x)
        spectrum = self.spectrum_layer(coeffs)

        if self.return_mode == "coeffs":
            return coeffs
        if self.return_mode == "spectrum":
            return spectrum
        return {
            "coeffs": coeffs,
            "spectrum": spectrum,
        }
