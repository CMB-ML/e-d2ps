from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pysm3.units as u
import numpy as np
import torch
from omegaconf import DictConfig

from cmbml.utils.map_fields_helper import map_field_str2int
from cmbml.core.config_helper import ConfigHelper
from cmbml.core.namers import Namer
from cmbml.utils.planck_instrument import Instrument, make_instrument
from d2ps_nn.utils.select_features import FeatureAsset, FeatureAssetNames


Array = np.ndarray


# Map statistics are saved by convention
STAT_MIN = 0
STAT_MAX = 1
STAT_MEAN = 2
STAT_STDV = 3


class MapProcessor(ABC):
    """
    Reversible preprocessing transform for maps.
    """

    selection_space = "pre"
    report_space = "post"
    pre_name = "processed"
    post_name = "raw"

    @abstractmethod
    def preprocess(self, x, channel=None, field_str="I"):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, x, channel=None, field_str="I"):
        raise NotImplementedError

    @staticmethod
    def _coerce_like(val, x):
        # Quantity in, Quantity out for NumPy/Astropy workflows
        # Quantity in, Tensor out for torch workflows (unit blind)
        if isinstance(x, torch.Tensor):
            if isinstance(val, u.Quantity):
                val = val.to_value()
            return torch.as_tensor(val, device=x.device, dtype=x.dtype)
        return val

@dataclass(frozen=True)
class IdentityMapProcessor(MapProcessor):
    selection_space = "pre"
    report_space = "pre"
    pre_name = "map"
    post_name = "map"

    def preprocess(self, x, channel=None, field_str: str = "I"):
        return x

    def postprocess(self, x, channel=None, field_str: str = "I"):
        return x


class LazyStat(ABC):
    """
    Late-loaded stat reference.
    """
    @abstractmethod
    def get(self, channel, field_str="I") -> np.ndarray:
        raise NotImplementedError


@dataclass
class LazyDatasetChannelStat(LazyStat):
    """
    Late-load one scalar stat at a time from dataset_stats.

    Expected loaded structure is like:
        stats["cmb"]["I"]["vmin"]
        stats["100"]["I"]["vmax"]
    The Config AssetHandler will have already converted these to
    Astropy Quantities.

    The full stats dict is loaded once on first use and cached.
    """
    stats_asset: object
    which: str  # "vmin" or "vmax"

    _cached_full_stats: Optional[dict] = field(default=None, init=False, repr=False)

    def get(self, channel, field_str: str = "I") -> np.ndarray:
        stats = self._get_full_stats()

        value = stats[channel][field_str][self.which]
        return value

    def _get_full_stats(self) -> dict:
        if self._cached_full_stats is None:
            stats = self.stats_asset.read()
            if not isinstance(stats, dict):
                raise TypeError(
                    f"dataset_stats.read() must return a dict, got {type(stats)}"
                )
            self._cached_full_stats = stats
        return self._cached_full_stats


@dataclass
class LazyFeaturePixelStat(LazyStat):
    """
    Late-load one pixelwise stat map at a time from feature stats assets.

    Expected per-channel asset payload is stacked like:
        arr[0] -> min map
        arr[1] -> max map
        arr[2] -> mean map
        arr[3] -> stdv map

    The loaded object may be ndarray or Quantity.

    Cache policy:
      - cache the loaded stacked payload per channel
      - extract requested stat and field on demand
    """
    feature_stats_asset: object
    which_idx: int

    _cached_payloads: dict = field(default_factory=dict, init=False, repr=False)

    def get(self, channel, field_str: str = "I"):
        payload = self._get_payload_for_channel(channel)
        stat_map = payload[self.which_idx]
        return self._select_field(stat_map, field_str)

    def _get_payload_for_channel(self, channel):
        key = str(channel)

        if key not in self._cached_payloads:
            asset = self._get_asset_for_channel(channel)
            with asset.name_tracker.set_context("freq", key):
                self._cached_payloads[key] = asset.read()

        return self._cached_payloads[key]

    def _get_asset_for_channel(self, channel):
        """
        Resolve the correct asset object for the requested channel.

        Assumes `feature_stats_asset` is a FeatureAsset-like object supporting
        `.feature_iter()`, where the yielded channel labels match the runtime
        `channel` argument used by preprocess().
        """
        key = str(channel)

        for ch, asset in self.feature_stats_asset.feature_iter():
            if str(ch) == key:
                return asset

        raise KeyError(f"No feature stats asset found for channel {channel!r}")

    @staticmethod
    def _select_field(stat_map, field_str: str = "I"):
        """
        For now:
          - if stat_map is already one map, return it unchanged
          - if stat_map has a leading map-field axis, select field_str
        """
        try:
            ndim = len(stat_map.shape)
        except AttributeError:
            # Very unusual, but just return as-is.
            return stat_map

        if ndim == 1:
            # already one map: (npix,)
            return stat_map

        # likely (n_fields, npix)
        field_int = map_field_str2int(field_str)
        return stat_map[field_int]

@dataclass(frozen=True)
class MinMaxProcessor(MapProcessor):
    """
    Min-max scaling to [0, 1].

    Works for:
      - global scalar stats
      - local pixelwise stats

    Stats are looked up by (channel, field_str) at runtime.
    """
    selection_space = "pre"
    report_space = "pre"
    pre_name = "minmax_map"
    post_name = "map"

    min_ref: LazyStat
    max_ref: LazyStat
    eps: float = 1e-12

    def preprocess(self, x, channel, field_str: str = "I"):
        mn, mx = self._get_min_max(channel=channel, field_str=field_str, x=x)

        if hasattr(x, "unit") and not hasattr(mn, "unit"):
            mn = u.Quantity(mn, x.unit)
            mx = u.Quantity(mx, x.unit)

        den = mx - mn

        if hasattr(den, "unit"):
            eps = u.Quantity(self.eps, den.unit)
        else:
            eps = self.eps

        return (x - mn) / (den + eps)

    def postprocess(self, x, channel, field_str: str = "I"):
        mn, mx = self._get_min_max(channel=channel, field_str=field_str, x=x)

        if not hasattr(x, "unit") and hasattr(mn, "unit"):
            x = u.Quantity(x, u.dimensionless_unscaled)

        span = mx - mn

        if hasattr(span, "unit"):
            eps = u.Quantity(self.eps, span.unit)
        else:
            eps = self.eps

        return x * (span + eps) + mn

    def _get_min_max(self, channel, field_str: str, x):
        mn = self.min_ref.get(channel=channel, field_str=field_str)
        mx = self.max_ref.get(channel=channel, field_str=field_str)

        mn = self._coerce_like(mn, x)
        mx = self._coerce_like(mx, x)

        return mn, mx


@dataclass(frozen=True)
class MeanStDevProcessor(MapProcessor):
    """
    Standardize to N(mu, sigma).

    Works for:
      - global scalar stats
      - local pixelwise stats

    Stats are looked up by (channel, field_str) at runtime.
    """
    selection_space = "pre"
    report_space = "pre"
    pre_name = "meanstdv_map"
    post_name = "map"

    mean_ref: LazyStat
    stdv_ref: LazyStat
    eps: float = 1e-12

    def preprocess(self, x, channel, field_str: str = "I"):
        mean, stdv = self._get_mean_stdv(channel=channel, field_str=field_str, x=x)

        if hasattr(x, "unit") and not hasattr(mean, "unit"):
            mean = u.Quantity(mean, x.unit)
            stdv = u.Quantity(stdv, x.unit)

        if hasattr(stdv, "unit"):
            eps = self.eps * stdv.unit
        else:
            eps = self.eps

        return (x - mean) / (stdv + eps)

    def postprocess(self, x, channel, field_str: str = "I"):
        mean, stdv = self._get_mean_stdv(channel=channel, field_str=field_str, x=x)

        if not hasattr(x, "unit") and hasattr(mean, "unit"):
            x = u.Quantity(x, u.dimensionless_unscaled)

        if hasattr(stdv, "unit"):
            eps = self.eps * stdv.unit
        else:
            eps = self.eps

        return x * (stdv + eps) + mean

    def _get_mean_stdv(self, channel, field_str: str, x):
        mean = self.mean_ref.get(channel=channel, field_str=field_str)
        stdv = self.stdv_ref.get(channel=channel, field_str=field_str)

        mean = self._coerce_like(mean, x)
        stdv = self._coerce_like(stdv, x)

        return mean, stdv


def make_px_processor(
    mode: Optional[str],
    cfg: DictConfig,
) -> MapProcessor:
    """
    Factory for global map scaling.

    Supported modes
    ---------------
    None, "null", "none":
        no preprocessing

    "global_minmax":
        late-load vmin/vmax from dataset_stats
    """
    if mode is None or mode in {"null", "none"}:
        return IdentityMapProcessor()

    stage_str = "preprocess_maps"
    ch = ConfigHelper(cfg, stage_str=stage_str)
    n = Namer(cfg)

    if mode == "global_minmax":
        dataset_stats_asset = ch.get_assets_in(n, stage_str)['dataset_stats']
        min_ref = LazyDatasetChannelStat(
            stats_asset=dataset_stats_asset,
            which="vmin",
        )
        max_ref = LazyDatasetChannelStat(
            stats_asset=dataset_stats_asset,
            which="vmax",
        )
        return MinMaxProcessor(
            min_ref=min_ref,
            max_ref=max_ref,
        )

    if mode == "local_minmax":
        assets_in = ch.get_assets_in(n, stage_str)

        stat_name_map = FeatureAssetNames(cmb="cmb_stats", obs="obs_stats")
        feature_stats = FeatureAsset.select(
            assets_in,
            cfg.features_type,
            instrument=make_instrument(cfg=cfg),
            name_map=stat_name_map,
        )

        min_ref = LazyFeaturePixelStat(
            feature_stats_asset=feature_stats,
            which_idx=STAT_MIN,
        )
        max_ref = LazyFeaturePixelStat(
            feature_stats_asset=feature_stats,
            which_idx=STAT_MAX,
        )

        return MinMaxProcessor(
            min_ref=min_ref,
            max_ref=max_ref,
        )

    if mode == "local_meanstdv":
        assets_in = ch.get_assets_in(n, stage_str)

        stat_name_map = FeatureAssetNames(cmb="cmb_stats", obs="obs_stats")
        feature_stats = FeatureAsset.select(
            assets_in,
            cfg.features_type,
            instrument=make_instrument(cfg=cfg),
            name_map=stat_name_map,
        )

        mean_ref = LazyFeaturePixelStat(
            feature_stats_asset=feature_stats,
            which_idx=STAT_MEAN,
        )
        stdv_ref = LazyFeaturePixelStat(
            feature_stats_asset=feature_stats,
            which_idx=STAT_STDV,
        )

        return MeanStDevProcessor(
            mean_ref=mean_ref,
            stdv_ref=stdv_ref,
        )

    raise ValueError(f"Unknown map preprocessing mode: {mode!r}")
