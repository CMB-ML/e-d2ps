from __future__ import annotations
from typing import Union, Optional

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import torch

from cmbml.core.config_helper import ConfigHelper
from cmbml.core.namers import Namer
from d2ps_nn.utils.parse_ell_list import parse_ells


Array = np.ndarray


class SpectrumProcessor(ABC):
    """
    Reversible preprocessing transform for spectra.

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
        raise NotImplementedError

    @staticmethod
    def _coerce_like(arr: np.ndarray, x):
        if isinstance(x, torch.Tensor):
            return torch.as_tensor(arr, device=x.device, dtype=x.dtype)
        return arr


@dataclass(frozen=True)
class IdentityProcessor(SpectrumProcessor):
    """No preprocessing."""
    selection_space = "pre"
    report_space = "pre"
    pre_name = "Dl"
    post_name = "Dl"

    def preprocess(self, x: Array) -> Array:
        return x

    def postprocess(self, x: Array) -> Array:
        return x


@dataclass(frozen=True)
class ScalarProcessor(SpectrumProcessor):
    """
    Divide by a scalar during preprocessing, multiply during postprocessing.

    Example:
        factor = 4000
        preprocess(x)  -> x / 4000
        postprocess(x) -> x * 4000
    """
    selection_space = "pre"
    report_space = "post"
    pre_name = "Dl_scaled"
    post_name = "Dl"

    factor: float

    def __post_init__(self) -> None:
        if self.factor == 0:
            raise ValueError("factor must be nonzero")

    def preprocess(self, x: Array) -> Array:
        return x / self.factor

    def postprocess(self, x: Array) -> Array:
        return x * self.factor


@dataclass(frozen=True)
class ClProcessor(SpectrumProcessor):
    """
    Convert Dl <-> Cl using the supplied ells.

    Assumes:
        Dl = ell * (ell + 1) * Cl / (2 pi)

    So preprocess converts Dl -> Cl, and postprocess converts Cl -> Dl.

    For ell < ell_min_safe, factor 1 is used to avoid division by zero /
    undefined low-ell behavior. Adjust if you want something stricter.
    """
    selection_space = "pre"
    report_space = "pre"
    pre_name = "Cl"
    post_name = "Dl"

    ells: Array
    ell_min_safe: int = 2

    def __post_init__(self) -> None:
        ells = np.asarray(self.ells, dtype=float)
        if ells.ndim != 1:
            raise ValueError(f"ells must be 1D, got shape {ells.shape}")

        dl_to_cl = np.ones_like(ells, dtype=float)
        cl_to_dl = np.ones_like(ells, dtype=float)

        mask = ells >= self.ell_min_safe
        denom = ells[mask] * (ells[mask] + 1.0)

        dl_to_cl[mask] = (2.0 * np.pi) / denom
        cl_to_dl[mask] = denom / (2.0 * np.pi)

        object.__setattr__(self, "ells", ells)
        object.__setattr__(self, "_dl_to_cl", dl_to_cl)
        object.__setattr__(self, "_cl_to_dl", cl_to_dl)

    def preprocess(self, x: Array) -> Array:
        dl_to_cl = self._coerce_like(self._dl_to_cl, x)
        self._check_last_dim(x)
        return x * dl_to_cl

    def postprocess(self, x: Array) -> Array:
        cl_to_dl = self._coerce_like(self._cl_to_dl, x)
        self._check_last_dim(x)
        return x * cl_to_dl

    def _check_last_dim(self, x: Array) -> None:
        if x.shape[-1] != self.ells.shape[0]:
            raise ValueError(
                "Last dimension of x must match length of ells: "
                f"{x.shape[-1]} != {self.ells.shape[0]}"
            )


@dataclass(frozen=True)
class MeanStdvProcessor(SpectrumProcessor):
    """
    Per-ell standardization with lazy loading.

    preprocess(x)  = (x - mean) / stdv
    postprocess(x) = x * stdv + mean

    Mean/st.dev are loaded from files on first use, then cached.
    """
    selection_space = "pre"
    report_space = "post"
    pre_name = "standardized_Dl"
    post_name = "Dl"

    mean_path: Union[str, Path]
    stdv_path: Union[str, Path]
    eps: float = 1e-12

    _mean: Optional[Array] = field(default=None, init=False, repr=False)
    _stdv: Optional[Array] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean_path", Path(self.mean_path))
        object.__setattr__(self, "stdv_path", Path(self.stdv_path))

    def preprocess(self, x: Array) -> Array:
        mean, stdv = self._get_mean_stdv()
        mean = self._coerce_like(mean, x)
        stdv = self._coerce_like(stdv, x)
        self._check_last_dim(x, mean)
        return (x - mean) / (stdv + self.eps)

    def postprocess(self, x):
        mean, stdv = self._get_mean_stdv()
        mean = self._coerce_like(mean, x)
        stdv = self._coerce_like(stdv, x)
        self._check_last_dim(x, mean)
        return x * (stdv + self.eps) + mean

    def _get_mean_stdv(self) -> tuple[Array, Array]:
        mean = self._mean
        stdv = self._stdv

        if mean is None or stdv is None:
            mean_path = self.mean_path
            stdv_path = self.stdv_path

            if not mean_path.exists():
                raise FileNotFoundError(
                    f"Mean spectrum file does not exist: {mean_path}"
                )
            if not stdv_path.exists():
                raise FileNotFoundError(
                    f"Std spectrum file does not exist: {stdv_path}"
                )

            mean = np.asarray(np.load(mean_path), dtype=float)
            stdv = np.asarray(np.load(stdv_path), dtype=float)

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


def make_ps_processor(
    mode: Optional[DictConfig],
    cfg: Optional[DictConfig] = None,
) -> SpectrumProcessor:
    """
    Helper for selecting PS pre/post-processing.

    Supported modes
    ---------------
    None, "none"
        No preprocessing.

    "scalar"
        Requires `factor`.

    "cl"
        Uses ells from cfg.

    "meanstdv"
        Gets `mean_path` and `stdv_path` from cfg.
    """
    if mode is None or (isinstance(mode, str) and mode.lower() == "none"):
        return IdentityProcessor()

    # if mode is just a number...
    try:
        factor = float(mode)
        return ScalarProcessor(factor=factor)
    except (TypeError, ValueError):
        pass  # if it's not a number

    if mode == "cl":
        ells = parse_ells(cfg.model.use_ells)
        if ells is None:
            raise ValueError("ells is required when mode='cl'")
        return ClProcessor(ells=ells)

    if mode == "meanstdv":
        # Mind sep_prep considerations: need to use train's input
        stage_str = "train"
        ch = ConfigHelper(cfg, stage_str=stage_str)
        n = Namer(cfg)
        mean_path = ch.get_assets_in(n, stage_str)['mean_ps'].path
        stdv_path = ch.get_assets_in(n, stage_str)['stdv_ps'].path
        return MeanStdvProcessor(
            mean_path=mean_path,
            stdv_path=stdv_path,
        )

    raise ValueError(f"Unknown preprocessing mode: {mode!r}")
