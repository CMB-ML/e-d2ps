from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from d2ps_nn.utils.px_scale import STAT_MIN, STAT_MAX, STAT_MEAN, STAT_STDV


ScaleKind = Literal["none", "minmax", "meanstdv"]
ScaleScope = Literal["global", "local"]


@dataclass
class ArrayScaler:
    kind: ScaleKind = "none"
    scope: ScaleScope = "global"
    eps: float = 1e-8

    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    fitted: bool = False

    def fit(self, data: np.ndarray) -> "ArrayScaler":
        if self.kind == "none":
            self.fitted = True
            return self

        if self.scope == "global":
            axis = None
            keepdims = False
        elif self.scope == "local":
            axis = 0
            keepdims = True
        else:
            raise ValueError(f"Unsupported scaler scope: {self.scope}")

        if self.kind == "minmax":
            self.a = np.min(data, axis=axis, keepdims=keepdims)
            self.b = np.max(data, axis=axis, keepdims=keepdims)

        elif self.kind == "meanstdv":
            self.a = np.mean(data, axis=axis, keepdims=keepdims)
            self.b = np.std(data, axis=axis, keepdims=keepdims)

        else:
            raise ValueError(f"Unsupported scaler kind: {self.kind}")

        self.fitted = True
        return self

    def fit_from_stats(self, stats: np.ndarray):
        """Fit an ArrayScaler directly from a pre-computed stats array (4, ...) ."""
        if self.kind == "none":
            self.fitted = True
            return self
        if self.kind == "minmax":
            self.a = stats[STAT_MIN]
            self.b = stats[STAT_MAX]
        elif self.kind == "meanstdv":
            self.a = stats[STAT_MEAN]
            self.b = stats[STAT_STDV]
        else:
            raise ValueError(f"Unsupported scaler kind: {self.kind!r}")
        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()

        if self.kind == "none":
            return data

        if self.kind == "minmax":
            denom = np.maximum(self.b - self.a, self.eps)
            return (data - self.a) / denom

        if self.kind == "meanstdv":
            denom = np.maximum(self.b, self.eps)
            return (data - self.a) / denom

        raise ValueError(f"Unsupported scaler kind: {self.kind}")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()

        if self.kind == "none":
            return data

        if self.kind == "minmax":
            return data * (self.b - self.a) + self.a

        if self.kind == "meanstdv":
            return data * self.b + self.a

        raise ValueError(f"Unsupported scaler kind: {self.kind}")

    def transform_tensor(self, data: torch.Tensor) -> torch.Tensor:
        self._check_fitted()

        if self.kind == "none":
            return data

        a = torch.as_tensor(self.a, dtype=data.dtype, device=data.device)
        b = torch.as_tensor(self.b, dtype=data.dtype, device=data.device)

        if self.kind == "minmax":
            denom = torch.clamp(b - a, min=self.eps)
            return (data - a) / denom

        if self.kind == "meanstdv":
            denom = torch.clamp(b, min=self.eps)
            return (data - a) / denom

        raise ValueError(f"Unsupported scaler kind: {self.kind}")

    def inverse_transform_tensor(self, data: torch.Tensor) -> torch.Tensor:
        self._check_fitted()

        if self.kind == "none":
            return data

        a = torch.as_tensor(self.a, dtype=data.dtype, device=data.device)
        b = torch.as_tensor(self.b, dtype=data.dtype, device=data.device)

        if self.kind == "minmax":
            return data * (b - a) + a

        if self.kind == "meanstdv":
            return data * b + a

        raise ValueError(f"Unsupported scaler kind: {self.kind}")

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet.")

    def __repr__(self):
        return (
            f"ArrayScaler("
            f"kind={self.kind!r}, "
            f"scope={self.scope!r}, "
            f"fitted={self.fitted}, "
            f"a={self._summarize_array(self.a)}, "
            f"b={self._summarize_array(self.b)}"
            f")"
        )

    @staticmethod
    def _summarize_array(arr, max_items=3):
        if arr is None:
            return None

        flat = arr.ravel()

        preview = flat[:max_items].tolist()

        if flat.size > max_items:
            preview.append("...")

        return (
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}, "
            f"preview={preview}"
        )
