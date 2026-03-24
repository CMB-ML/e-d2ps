from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import torch

from cmbml.core.config_helper import ConfigHelper
from cmbml.core.namers import Namer


Array = np.ndarray


class ParamsProcessor(ABC):
    """
    Reversible preprocessing transform for cosmo params.

    `preprocess` is applied before the model.
    `postprocess` undoes that transform afterward.
    """
    selection_space = "pre"
    report_space = "post"
    pre_name = "processed"
    post_name = "raw"

    @abstractmethod
    def preprocess(self, x: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, x: Array) -> Array:
        """
        Note that this is not actually needed with 
        the current workflow!
        """
        raise NotImplementedError

    @staticmethod
    def _coerce_like(arr: np.ndarray, x):
        if isinstance(x, torch.Tensor):
            return torch.as_tensor(arr, 
                                   device=x.device, 
                                   dtype=x.dtype)
        return arr


@dataclass(frozen=True)
class IdentityProcessor(ParamsProcessor):
    """No preprocessing."""
    selection_space = "pre"
    report_space = "pre"
    pre_name = "params"
    post_name = "params"

    def preprocess(self, x: Array) -> Array:
        return x

    def postprocess(self, x: Array) -> Array:
        return x


@dataclass(frozen=True)
class StandardizeProcessor(ParamsProcessor):
    """
    Standardize cosmological parameters with lazy loading.

    preprocess(x)  = (x - mean) / stdv
    postprocess(x) = x * stdv + mean

    Mean/st.dev are loaded from file on first use, then cached.
    """
    selection_space = "pre"
    report_space = "post"
    pre_name = "standardized_params"
    post_name = "params"

    stat_path: str | Path
    eps: float = 1e-12

    _mean: Array | None = field(default=None, init=False, repr=False)
    _stdv: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stat_path", Path(self.stat_path))

    def preprocess(self, x: Array) -> Array:
        mean, stdv = self._get_mean_stdv()
        mean = self._coerce_like(mean, x)
        stdv = self._coerce_like(stdv, x)
        self._check_last_dim(x, mean)
        return (x - mean) / (stdv + self.eps)

    def postprocess(self, x: Array):
        mean, stdv = self._get_mean_stdv()
        mean = self._coerce_like(mean, x)
        stdv = self._coerce_like(stdv, x)
        self._check_last_dim(x, mean)
        return x * (stdv + self.eps) + mean

    def _get_mean_stdv(self) -> tuple[Array, Array]:
        mean = self._mean
        stdv = self._stdv

        if mean is None or stdv is None:
            stat_path = self.stat_path

            if not stat_path.exists():
                raise FileNotFoundError(
                    f"Param statistics file does not exist: {stat_path}"
                )

            stats = ConfigHelper().read(stat_path)

            mean = np.asarray(stats["mean"], dtype=float)
            stdv = np.asarray(stats["stdv"], dtype=float)

            if mean.ndim != 1:
                raise ValueError(f"mean must be 1D, got shape {mean.shape}")
            if stdv.ndim != 1:
                raise ValueError(f"stdv must be 1D, got shape {stdv.shape}")
            if mean.shape != stdv.shape:
                raise ValueError(
                    f"mean and stdv must have same shape, got {mean.shape} and {stdv.shape}"
                )

            object.__setattr__(self, "_mean", mean)
            object.__setattr__(self, "_stdv", stdv)

        return mean, stdv

    @staticmethod
    def _check_last_dim(x: Array, mean: Array) -> None:
        if x.shape[-1] != mean.shape[0]:
            raise ValueError(
                "Last dimension of x must match length of mean/std: "
                f"{x.shape[-1]} != {mean.shape[0]}"
            )


@dataclass(frozen=True)
class MinMaxProcessor(ParamsProcessor):
    """
    Min-max scale cosmological parameters with lazy loading.

    preprocess(x)  = (x - min) / (max - min)
    postprocess(x) = x * (max - min) + min

    Min/max are loaded from files on first use, then cached.
    """
    selection_space = "pre"
    report_space = "post"
    pre_name = "scaled_params"
    post_name = "params"

    stat_path: str | Path
    eps: float = 1e-12

    _vmin: Array | None = field(default=None, init=False, repr=False)
    _vmax: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stat_path", Path(self.stat_path))

    def preprocess(self, x: Array) -> Array:
        vmin, vmax = self._get_min_max()
        vmin = self._coerce_like(vmin, x)
        vmax = self._coerce_like(vmax, x)
        self._check_last_dim(x, vmin)
        return (x - vmin) / (vmax - vmin + self.eps)

    def postprocess(self, x: Array):
        vmin, vmax = self._get_min_max()
        vmin = self._coerce_like(vmin, x)
        vmax = self._coerce_like(vmax, x)
        self._check_last_dim(x, vmin)
        return x * (vmax - vmin + self.eps) + vmin

    def _get_min_max(self) -> tuple[Array, Array]:
        vmin = self._vmin
        vmax = self._vmax

        if vmin is None or vmax is None:
            stat_path = self.stat_path

            if not stat_path.exists():
                raise FileNotFoundError(
                    f"Param statistics file does not exist: {stat_path}"
                )

            stats = ConfigHelper().read(stat_path)

            vmin = np.asarray(stats["vmin"], dtype=float)
            vmax = np.asarray(stats["vmax"], dtype=float)

            if vmin.ndim != 1:
                raise ValueError(f"mean must be 1D, got shape {vmin.shape}")
            if vmax.ndim != 1:
                raise ValueError(f"stdv must be 1D, got shape {vmax.shape}")
            if vmin.shape != vmax.shape:
                raise ValueError(
                    f"vmin and vmax must have same shape, got {vmin.shape} and {vmax.shape}"
                )

            object.__setattr__(self, "_vmin", vmin)
            object.__setattr__(self, "_vmax", vmax)

        return vmin, vmax

    @staticmethod
    def _check_last_dim(x: Array, vmin: Array) -> None:
        if x.shape[-1] != vmin.shape[0]:
            raise ValueError(
                "Last dimension of x must match length of vmin/vmax: "
                f"{x.shape[-1]} != {vmin.shape[0]}"
            )


def make_cosmo_param_processor(
    mode: DictConfig | None,
    cfg: DictConfig | None = None,
) -> ParamsProcessor:
    """
    Helper for selecting cosmo param pre/post-processing.

    Supported modes
    ---------------
    None, "none"
        No preprocessing.

    "meanstdv"
        Gets `stat_path` from cfg.

    "minmax"
        Gets `stat_path` from cfg.
    """
    if mode is None or mode == "none":
        return IdentityProcessor()

    ch = ConfigHelper(cfg, stage_str=stage_str)
    n = Namer(cfg)
    stat_path = ch.get_assets_in(n, stage_str)['param_stats'].path

    if mode == "meanstdv":
        stage_str = "train"
        return StandardizeProcessor(
            stat_path=stat_path,
        )

    if mode == "minmax":
        stage_str = "train"
        return MinMaxProcessor(
            stat_path=stat_path,
        )

    raise ValueError(f"Unknown preprocessing mode: {mode!r}")
