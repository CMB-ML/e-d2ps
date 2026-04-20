from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


Array = np.ndarray


# Lock in saved-stat convention everywhere.
STAT_MIN = 0
STAT_MAX = 1
STAT_MEAN = 2
STAT_STDV = 3


def compute_stacked_stats(
    data: Array,
    axis: int = 0,
    keepdims: bool = False,
    dtype=None,
) -> Array:
    """
    Return stats stacked as

        out[STAT_MIN]  = min
        out[STAT_MAX]  = max
        out[STAT_MEAN] = mean
        out[STAT_STDV] = stdv
    """
    data = np.asarray(data)

    mn = np.min(data, axis=axis, keepdims=keepdims)
    mx = np.max(data, axis=axis, keepdims=keepdims)
    mean = np.mean(data, axis=axis, keepdims=keepdims, dtype=dtype)
    stdv = np.std(data, axis=axis, keepdims=keepdims, dtype=dtype)

    return np.stack([mn, mx, mean, stdv], axis=0)


class Processor(ABC):
    """
    Reversible preprocessing transform.
    """

    selection_space = "pre"
    report_space = "post"
    pre_name = "processed"
    post_name = "raw"

    @abstractmethod
    def preprocess(self, x):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, x):
        raise NotImplementedError

    @staticmethod
    def _coerce_like(val, x):
        if isinstance(x, torch.Tensor):
            return torch.as_tensor(val, device=x.device, dtype=x.dtype)
        return val


@dataclass(frozen=True)
class IdentityProcessor(Processor):
    selection_space = "pre"
    report_space = "pre"
    pre_name = "coef"
    post_name = "coef"

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class LazyStat(ABC):
    @abstractmethod
    def get(self):
        raise NotImplementedError


@dataclass
class LazyStackedStat(LazyStat):
    """
    Late-load one stat from a stacked stats asset.

    Expected saved payload shape:
        arr[STAT_MIN]
        arr[STAT_MAX]
        arr[STAT_MEAN]
        arr[STAT_STDV]
    """
    stats_asset: object
    which_idx: int

    _cached_payload: Optional[Array] = field(default=None, init=False, repr=False)

    def get(self):
        if self._cached_payload is None:
            payload = self.stats_asset.read()
            payload = np.asarray(payload)

            if payload.shape[0] < 4:
                raise ValueError(
                    f"Expected stacked stats with first axis size >= 4, got {payload.shape}"
                )

            self._cached_payload = payload

        return self._cached_payload[self.which_idx]


@dataclass(frozen=True)
class LocalMinMaxProcessor(Processor):
    selection_space = "pre"
    report_space = "pre"
    pre_name = "minmax_coef"
    mid_name = "ps_coef"
    post_name = "ps_true"

    min_ref: LazyStat
    max_ref: LazyStat
    eps: float = 1e-12

    def preprocess(self, x):
        mn, mx = self._get_min_max(x)
        return (x - mn) / (mx - mn + self.eps)

    def postprocess(self, x):
        mn, mx = self._get_min_max(x)
        return x * (mx - mn + self.eps) + mn

    def _get_min_max(self, x):
        mn = self._coerce_like(self.min_ref.get(), x)
        mx = self._coerce_like(self.max_ref.get(), x)
        return mn, mx


@dataclass(frozen=True)
class LocalMeanStDevProcessor(Processor):
    selection_space = "pre"
    report_space = "pre"
    pre_name = "meanstdv_coef"
    mid_name = "ps_coef"
    post_name = "ps_true"

    mean_ref: LazyStat
    stdv_ref: LazyStat
    eps: float = 1e-12

    def preprocess(self, x):
        mean, stdv = self._get_mean_stdv(x)
        return (x - mean) / (stdv + self.eps)

    def postprocess(self, x):
        mean, stdv = self._get_mean_stdv(x)
        return x * (stdv + self.eps) + mean

    def _get_mean_stdv(self, x):
        mean = self._coerce_like(self.mean_ref.get(), x)
        stdv = self._coerce_like(self.stdv_ref.get(), x)
        return mean, stdv


def make_coef_processor(
    mode: Optional[str],
    *,
    coef_stats_asset,
) -> Processor:
    """
    Factory for coefficient preprocessing.

    Supported modes
    ---------------
    None, "null", "none":
        no preprocessing

    "local_minmax":
        use stacked coef stats [min, max, mean, stdv]

    "local_meanstdv":
        use stacked coef stats [min, max, mean, stdv]
    """
    if mode is None or mode in {"null", "none"}:
        return IdentityProcessor()

    if mode == "local_minmax":
        return LocalMinMaxProcessor(
            min_ref=LazyStackedStat(coef_stats_asset, STAT_MIN),
            max_ref=LazyStackedStat(coef_stats_asset, STAT_MAX),
        )

    if mode == "local_meanstdv":
        return LocalMeanStDevProcessor(
            mean_ref=LazyStackedStat(coef_stats_asset, STAT_MEAN),
            stdv_ref=LazyStackedStat(coef_stats_asset, STAT_STDV),
        )

    raise ValueError(f"Unknown coef preprocessing mode: {mode!r}")
