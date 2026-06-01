from typing import Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch

from cmbml.core import ConfigHelper, Namer


class MapShaper(ABC):
    """Reversible-in-principle layout transform for HEALPix maps.

    Operates on (npix,) or (B, npix) and rearranges into whatever the
    downstream model expects. Pure geometry — no channel/field state.
    """
    @abstractmethod
    def forward(self, x): ...
    # Inverse intentionally omitted for now; trivial to add via argsort
    # when an inverse-reshape consumer exists.


class IdentityShaper(MapShaper):
    def forward(self, x):
        return x


# @dataclass
# class LUTShaper(MapShaper):
#     """Apply a precomputed integer index LUT, then add a singleton channel dim.

#     Pure layout: out = x[..., lut], reshaping (..., npix) into (..., *lut.shape).
#     """
#     def __init__(self, lut, add_channel_dim: bool = True):
#         self._lut_np = np.asarray(lut, dtype=np.int64)
#         self.add_channel_dim = add_channel_dim
#         self._lut_torch_cache = {}  # device -> tensor

#     def _get_lut(self, x):
#         """Return the LUT matching x's array type and device."""
#         if not isinstance(x, torch.Tensor):
#             return self._lut_np
#         dev = x.device
#         if dev not in self._lut_torch_cache:
#             self._lut_torch_cache[dev] = torch.as_tensor(self._lut_np, device=dev)
#         return self._lut_torch_cache[dev]

#     def forward(self, x):
#         out = x[..., self._get_lut(x)]
#         if self.add_channel_dim:
#             axis = out.ndim - self._lut_np.ndim
#             if isinstance(out, torch.Tensor):
#                 out = out.unsqueeze(axis)
#             else:
#                 out = np.expand_dims(out, axis)
#         return out

class LUTShaper(MapShaper):
    def __init__(self, lut_ref, add_channel_dim: bool = True):
        # lut_ref: LazyLUT, anything with .get() -> ndarray, or a raw ndarray.
        self._lut_ref = lut_ref
        self.add_channel_dim = add_channel_dim
        self._lut_np = None            # resolved on first forward / first get
        self._lut_torch_cache = {}     # device -> tensor

    def _resolve_lut_np(self):
        if self._lut_np is None:
            ref = self._lut_ref
            src = ref.get() if hasattr(ref, "get") else ref
            self._lut_np = np.asarray(src, dtype=np.int64)
        return self._lut_np

    def _get_lut(self, x):
        lut_np = self._resolve_lut_np()
        if not isinstance(x, torch.Tensor):
            return lut_np
        dev = x.device
        if dev not in self._lut_torch_cache:
            self._lut_torch_cache[dev] = torch.as_tensor(lut_np, device=dev)
        return self._lut_torch_cache[dev]

    def forward(self, x):
        lut = self._get_lut(x)             # also resolves _lut_np
        out = x[..., lut]
        if self.add_channel_dim:
            out = out.unsqueeze(out.ndim - self._lut_np.ndim)
        return out


@dataclass
class LazyLUT:
    """Late-loaded pixel-rearrangement LUT.

    Reads once on first use under the name_tracker 'style' context, then caches.
    Parallels LazyStat: the asset can't be read at __init__ time because the
    read requires a context that isn't set until execute().
    """
    lut_asset: object
    lut_style: str
    _cached: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def get(self) -> np.ndarray:
        if self._cached is None:
            with self.lut_asset.name_tracker.set_context("style", self.lut_style):
                self._cached = np.asarray(self.lut_asset.read(), dtype=np.int64)
        return self._cached


def make_px_shaper(cfg, stage_str="train_otf") -> MapShaper:
    lut_style = cfg.model.patches.get("lut_style", None)
    if lut_style is None or lut_style in ["none", "None", "null"]:
        return IdentityShaper()
    ch = ConfigHelper(cfg, stage_str=stage_str)
    n = Namer(cfg)
    lut_asset = ch.get_assets_in(n, stage_str)["lut"]
    return LUTShaper(LazyLUT(lut_asset=lut_asset, lut_style=lut_style))
