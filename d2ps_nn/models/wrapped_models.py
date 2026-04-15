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
        backbone: nn.Module,
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

        self.backbone = backbone
        self.spectrum_layer = spectrum_layer
        self.return_mode = return_mode

        self.example_in_shape = self.backbone.example_in_shape

    def forward(self, x: torch.Tensor):
        coeffs = self.backbone(x)
        spectrum = self.spectrum_layer(coeffs)

        if self.return_mode == "coeffs":
            return coeffs
        if self.return_mode == "spectrum":
            return spectrum
        return {
            "coeffs": coeffs,
            "spectrum": spectrum,
        }


def build_from_registry(
    kind: str,
    kwargs: dict[str, Any],
    registry: Mapping[str, Callable[..., nn.Module]],
) -> nn.Module:
    if kind not in registry:
        raise KeyError(
            f"Unknown kind={kind!r}. "
            f"Available: {sorted(registry.keys())}"
        )
    return registry[kind](**kwargs)


def build_wrapped_spectrum_model(
    backbone: Union[nn.Module, tuple[str, dict[str, Any]]],
    spectrum_layer: Union[BaseSpectrumLayer, tuple[str, dict[str, Any]]],
    backbone_registry: Optional[Mapping[str, Callable[..., nn.Module]]] = None,
    spectrum_layer_registry: Optional[Mapping[str, Callable[..., nn.Module]]] = None,
    return_mode: str = "spectrum",
) -> WrappedSpectrumModel:
    """
    Convenience builder that accepts either instances or (kind, kwargs) tuples.

    Example
    -------
    model = build_wrapped_spectrum_model(
        backbone=("SimpleEncDec", {...}),
        spectrum_layer=("PolynomialSpectrumLayer", {...}),
        backbone_registry=BACKBONE_REGISTRY,
        spectrum_layer_registry=SPECTRUM_LAYER_REGISTRY,
        return_mode="both",
    )
    """
    if isinstance(backbone, nn.Module):
        backbone_obj = backbone
    else:
        if backbone_registry is None:
            raise ValueError("backbone_registry is required when backbone is not an instance")
        backbone_kind, backbone_kwargs = backbone
        backbone_obj = build_from_registry(backbone_kind, backbone_kwargs, backbone_registry)

    if isinstance(spectrum_layer, nn.Module):
        spectrum_layer_obj = spectrum_layer
    else:
        if spectrum_layer_registry is None:
            raise ValueError(
                "spectrum_layer_registry is required when spectrum_layer is not an instance"
            )
        layer_kind, layer_kwargs = spectrum_layer
        layer_obj = build_from_registry(layer_kind, layer_kwargs, spectrum_layer_registry)
        if not isinstance(layer_obj, BaseSpectrumLayer):
            raise TypeError(
                f"Built spectrum layer {layer_kind!r}, but it is not a BaseSpectrumLayer"
            )
        spectrum_layer_obj = layer_obj

    return WrappedSpectrumModel(
        backbone=backbone_obj,
        spectrum_layer=spectrum_layer_obj,
        return_mode=return_mode,
    )
