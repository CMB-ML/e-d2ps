"""Declarative record schema for training metrics.

One source of truth for (a) the values a training run produces and (b) the
columns written to the loss record. Adding a tracked quantity means adding one
field to one dataclass -- headers and rows both follow automatically, and can
no longer drift out of order relative to each other.

Two record shapes are supported:

  wide  -- one row per epoch. Scalar columns get one column each; per-ell
           columns expand to one column per ell (`FVE_ell2`, `FVE_ell3`, ...).
           Fine for a handful of ells; unwieldy past a few dozen.

  long  -- one row per (epoch, ell). Only per-ell columns appear. This is the
           right shape for full-resolution diagnostics: at ell=1280 the wide
           form is 6405 columns, the long form is 7.

Both are append-safe: a run killed mid-write loses at most one trailing row.


Per-ell sufficient statistics
-----------------------------
`EllStatsAccumulator` streams six quantities per ell over a validation pass:

    n, mean_pred, mean_true, var_pred, var_true, mse

These are sufficient for every second-order diagnostic. With
Sp = (n-1)*var_pred, St = (n-1)*var_true, and unbiased variances throughout:

    R(ell)     = sqrt(var_pred / var_true)
    FVE(ell)   = 1 - mse / var_true
    Spt        = (Sp + St + n*(mean_pred - mean_true)**2 - n*mse) / 2
    rho(ell)   = Spt / sqrt(Sp * St)

Note that rho requires the means. var_pred/var_true/mse alone pin down rho only
if you assume the predicted and true means coincide -- which is the mean-hugging
failure mode you would be measuring in the first place.

Over a fixed validation set with a fitted-once label scaler, `mean_true` and
`var_true` do not change between epochs. They are therefore split off into a
reference record (`EllRefMetrics`) written once per run, leaving only
`mean_pred`, `var_pred` and `mse` in the per-epoch record. The accumulator
still computes the true statistics every pass; `EllStats.check_reference`
compares them against the stored reference so that a silent change -- a scaler
refit across a warm start, a different validation split, a config drift -- is
caught rather than averaged into the diagnostics.

This split assumes the labels really are fixed. Against on-the-fly generated
targets they are not, and the true statistics belong back in the per-epoch
record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

_META_KEY = "record"
_KINDS = ("scalar", "per_ell")


# --------------------------------------------------------------------------- #
# Field declaration
# --------------------------------------------------------------------------- #
def metric(header: str,
           *,
           kind: str = "scalar",
           enabled_by: Optional[str] = None,
           **field_kwargs):
    """Declare a dataclass field that also describes its own record column(s).

    header      Column name. For kind="per_ell" in a wide record the ell is
                appended, e.g. f"{header}_ell{ell}".
    kind        "scalar" (one column) or "per_ell" (one per ell).
    enabled_by  Name of a boolean attribute on the executor (or key in a flags
                mapping). When falsy the column is omitted entirely -- both
                from the header and from every row, so they stay consistent.

    A dataclass field declared with plain `field()` or a bare annotation is not
    recorded at all; use that for values you want to pass around but not store.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown metric kind {kind!r}; expected one of {_KINDS}")
    meta = dict(field_kwargs.pop("metadata", {}))
    meta[_META_KEY] = dict(header=header, kind=kind, enabled_by=enabled_by)
    return field(metadata=meta, **field_kwargs)


@dataclass(frozen=True)
class ColumnSpec:
    name: str                       # attribute name on a metrics object
    header: str
    kind: str = "scalar"
    enabled_by: Optional[str] = None

    @property
    def is_per_ell(self) -> bool:
        return self.kind == "per_ell"


def specs_from(*dataclass_types) -> list[ColumnSpec]:
    """Collect ColumnSpecs from one or more metrics dataclasses, in field order.

    Subclass a metrics dataclass to add fields and they appear here for free,
    appended after the inherited ones.
    """
    specs: list[ColumnSpec] = []
    seen: set[str] = set()
    for dc in dataclass_types:
        for f in fields(dc):
            meta = f.metadata.get(_META_KEY)
            if meta is None:
                continue
            if f.name in seen:
                raise ValueError(
                    f"{f.name!r} is declared as a recorded metric more than once"
                )
            seen.add(f.name)
            specs.append(ColumnSpec(name=f.name, **meta))
    return specs


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class RecordSchema:
    """Turns ColumnSpecs plus runtime state into headers and rows.

    Build it once, at the same point you would previously have written the
    header list. Because `headers()` and `row()` walk the same filtered spec
    list in the same order, they cannot disagree.
    """

    def __init__(self,
                 specs: Sequence[ColumnSpec],
                 *,
                 flags: Any = None,
                 ells: Optional[Sequence[int]] = None):
        self.ells = [int(e) for e in ells] if ells is not None else []
        self.specs = [s for s in specs if self._enabled(s, flags)]
        if any(s.is_per_ell for s in self.specs) and not self.ells:
            raise ValueError("per-ell columns are declared but no ells were supplied")

    @staticmethod
    def _enabled(spec: ColumnSpec, flags: Any) -> bool:
        if spec.enabled_by is None:
            return True
        if flags is None:
            return False
        if isinstance(flags, Mapping):
            return bool(flags.get(spec.enabled_by, False))
        return bool(getattr(flags, spec.enabled_by, False))

    # -- lookup ------------------------------------------------------------- #
    def _resolve(self, name: str, sources: Sequence[Any]) -> Any:
        for src in sources:
            if isinstance(src, Mapping):
                if name in src:
                    return src[name]
            elif hasattr(src, name):
                return getattr(src, name)
        raise KeyError(
            f"no source provides {name!r}; pass the metrics object (or a dict) "
            f"that carries it to row()/long_rows()"
        )

    def _as_ell_array(self, spec: ColumnSpec, value: Any) -> np.ndarray:
        if value is None:
            return np.full(len(self.ells), np.nan)
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != len(self.ells):
            raise ValueError(
                f"{spec.name!r} has {arr.size} values but the schema has "
                f"{len(self.ells)} ells"
            )
        return arr

    # -- wide: one row per epoch -------------------------------------------- #
    def headers(self, leading: Sequence[str] = ()) -> list[str]:
        out = list(leading)
        for s in self.specs:
            if s.is_per_ell:
                out.extend(f"{s.header}_ell{ell}" for ell in self.ells)
            else:
                out.append(s.header)
        return out

    def row(self, *sources: Any, leading: Sequence[Any] = ()) -> list:
        out = list(leading)
        for s in self.specs:
            value = self._resolve(s.name, sources)
            if s.is_per_ell:
                out.extend(self._as_ell_array(s, value).tolist())
            else:
                out.append(value)
        return out

    # -- long: one row per (epoch, ell) ------------------------------------- #
    def long_headers(self, leading: Sequence[str] = ("Epoch",)) -> list[str]:
        return (list(leading) + ["ell"]
                + [s.header for s in self.specs if s.is_per_ell])

    def long_rows(self, *sources: Any, leading: Sequence[Any] = ()) -> list[list]:
        per_ell = [s for s in self.specs if s.is_per_ell]
        if not per_ell:
            return []
        cols = [self._as_ell_array(s, self._resolve(s.name, sources))
                for s in per_ell]
        lead = list(leading)
        return [lead + [ell] + [float(c[i]) for c in cols]
                for i, ell in enumerate(self.ells)]


# --------------------------------------------------------------------------- #
# Per-ell metrics and streaming accumulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EllMetrics:
    """The per-epoch half: statistics that actually move between epochs.

    n_valid is deliberately not a recorded column here -- it is constant across
    ell, so in a long record it belongs in the leading index rather than being
    repeated once per ell.

    mse cannot be dropped in favour of the marginals: it is the only column
    carrying cross-information between prediction and truth, and so the only
    route to rho.
    """
    n_valid: int
    mean_pred: np.ndarray = metric("mean_pred", kind="per_ell")
    var_pred:  np.ndarray = metric("var_pred",  kind="per_ell")
    mse:       np.ndarray = metric("mse",       kind="per_ell")


@dataclass(frozen=True)
class EllRefMetrics:
    """The write-once half: properties of the validation labels themselves.

    Constant across epochs for a fixed validation set under a fitted-once
    scaler. Written to its own sidecar record at the start of a run.
    """
    n_valid: int
    mean_true: np.ndarray = metric("mean_true", kind="per_ell")
    var_true:  np.ndarray = metric("var_true",  kind="per_ell")


@dataclass(frozen=True)
class EllStats:
    """Everything one validation pass yields, before it is split for storage."""
    n_valid: int
    mean_pred: np.ndarray
    mean_true: np.ndarray
    var_pred: np.ndarray
    var_true: np.ndarray
    mse: np.ndarray

    def per_epoch(self) -> EllMetrics:
        return EllMetrics(n_valid=self.n_valid,
                          mean_pred=self.mean_pred,
                          var_pred=self.var_pred,
                          mse=self.mse)

    def reference(self) -> EllRefMetrics:
        return EllRefMetrics(n_valid=self.n_valid,
                             mean_true=self.mean_true,
                             var_true=self.var_true)

    def check_reference(self,
                        ref: EllRefMetrics,
                        *,
                        rtol: float = 1e-6,
                        label: str = "") -> bool:
        """Verify the label statistics still match the stored reference.

        Returns True when they agree. The true statistics are computed every
        pass anyway, so this recovers the integrity check that storing them
        per-epoch would have given, at no storage cost.
        """
        ok = True
        tag = f" [{label}]" if label else ""
        # A negative reference n means the size was not stored (it lives in the
        # per-epoch record); check only the statistics.
        if ref.n_valid >= 0 and ref.n_valid != self.n_valid:
            logger.error(
                f"validation set size changed{tag}: reference n={ref.n_valid}, "
                f"this pass n={self.n_valid}"
            )
            ok = False
        for name in ("mean_true", "var_true"):
            got = np.asarray(getattr(self, name), dtype=float)
            want = np.asarray(getattr(ref, name), dtype=float)
            if got.shape != want.shape:
                logger.error(f"{name} shape changed{tag}: {want.shape} -> {got.shape}")
                ok = False
                continue
            if not np.allclose(got, want, rtol=rtol, atol=0.0, equal_nan=True):
                worst = int(np.nanargmax(np.abs(got - want) / (np.abs(want) + 1e-30)))
                logger.error(
                    f"{name} drifted from the run reference{tag}; worst at index "
                    f"{worst}: {want[worst]:.8g} -> {got[worst]:.8g}. The label "
                    f"scaler or validation split has changed mid-run."
                )
                ok = False
        return ok


class EllStatsAccumulator:
    """Streaming per-ell sufficient statistics; never holds the full pass.

    Accumulates in float64 around a per-ell shift taken from the first batch's
    true values. Variance and MSE are invariant under a shift applied equally
    to pred and true, so the shift cancels exactly while keeping the sums of
    squares small and well conditioned -- which matters for Dl-scale values
    whose variance is many orders of magnitude below their mean square.

    Call under torch.no_grad(), not torch.inference_mode(): reductions computed
    inside inference mode are inference tensors and cannot be read back out.
    """

    def __init__(self,
                 n_ells: int,
                 device,
                 cols: Optional[Sequence[int]] = None,
                 dtype: torch.dtype = torch.float64):
        self._dtype = dtype
        self.n_ells = int(n_ells)
        self.cols = (None if cols is None
                     else torch.as_tensor(list(cols), device=device, dtype=torch.long))
        if self.cols is not None and self.cols.numel() != self.n_ells:
            raise ValueError(
                f"cols has {self.cols.numel()} entries but n_ells is {self.n_ells}"
            )
        zeros = lambda: torch.zeros(self.n_ells, device=device, dtype=dtype)  # noqa: E731
        self.shift = None
        self.n = 0
        self.sum_p, self.sum_t = zeros(), zeros()
        self.sum_pp, self.sum_tt, self.sum_pt = zeros(), zeros(), zeros()

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        if self.cols is not None:
            x = x.index_select(1, self.cols)
        elif x.shape[1] != self.n_ells:
            raise ValueError(
                f"expected {self.n_ells} output columns, got {x.shape[1]}; "
                f"pass cols= if the model output is not one column per ell"
            )
        return x.to(self._dtype)

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        """Accumulate one batch. Both tensors are (batch, n_model_ells)."""
        p = self._prepare(pred)
        t = self._prepare(true)
        if self.shift is None:
            self.shift = t.mean(dim=0)
        p = p - self.shift
        t = t - self.shift
        self.n += int(p.shape[0])
        self.sum_p += p.sum(dim=0)
        self.sum_t += t.sum(dim=0)
        self.sum_pp += (p * p).sum(dim=0)
        self.sum_tt += (t * t).sum(dim=0)
        self.sum_pt += (p * t).sum(dim=0)

    def result(self) -> Optional[EllStats]:
        """Full statistics for the pass. Split with .per_epoch() / .reference()."""
        n = self.n
        if n < 2:
            logger.warning(f"EllStatsAccumulator saw {n} samples; need >= 2")
            return None
        mp = self.sum_p / n
        mt = self.sum_t / n
        var_p = (self.sum_pp - n * mp * mp) / (n - 1)
        var_t = (self.sum_tt - n * mt * mt) / (n - 1)
        mse = (self.sum_pp - 2.0 * self.sum_pt + self.sum_tt) / n
        to_np = lambda x: x.cpu().numpy()  # noqa: E731
        return EllStats(
            n_valid=n,
            mean_pred=to_np(mp + self.shift),
            mean_true=to_np(mt + self.shift),
            var_pred=to_np(var_p.clamp_min(0.0)),
            var_true=to_np(var_t.clamp_min(0.0)),
            mse=to_np(mse.clamp_min(0.0)),
        )


# --------------------------------------------------------------------------- #
# Derived diagnostics (for the repack / analysis side)
# --------------------------------------------------------------------------- #
def derive_diagnostics(n, mean_pred, mean_true, var_pred, var_true, mse,
                       eps: float = 1e-30) -> dict:
    """Recover R, rho and FVE from the stored sufficient statistics.

    Every argument may be a scalar or an array over ell. Variances are assumed
    unbiased (divisor n-1), matching EllStatsAccumulator and torch's default.
    """
    n = np.asarray(n, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)
    var_true = np.asarray(var_true, dtype=float)
    mse = np.asarray(mse, dtype=float)
    bias = np.asarray(mean_pred, dtype=float) - np.asarray(mean_true, dtype=float)

    sp = (n - 1.0) * var_pred
    st = (n - 1.0) * var_true
    spt = 0.5 * (sp + st + n * bias**2 - n * mse)

    return dict(
        R=np.sqrt(var_pred / (var_true + eps)),
        rho=spt / np.sqrt(np.maximum(sp * st, eps)),
        FVE=1.0 - mse / (var_true + eps),
        bias=bias,
    )
