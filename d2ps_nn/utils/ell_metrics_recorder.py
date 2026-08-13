"""Per-ell metric recording as a collaborator object.

Owns everything the per-ell diagnostics need -- the two record schemas, the
streaming accumulator, the reference sidecar and its drift check -- so that an
executor integrates by holding one of these and calling three methods. Nothing
here depends on the executor's class, so the same object serves
TrainingExecutor, its subclasses, and (once its execute() grows the same
submethods) the OTF executor.

Lifecycle, once per run:

    self.ell_recorder.begin_run(is_continuation)

and then once per epoch:

    acc = self.ell_recorder.new_accumulator()      # None when disabled
    ... evaluate(..., ell_accumulator=acc)
    self.ell_recorder.record(epoch)

Disabled recorders return None and no-op, so the calls can sit unguarded in
the base executor and configs that do not opt in are unaffected.

The ell grid is `self.ells` as the executor already defines it -- an arbitrary
numpy array, not necessarily contiguous, whose index j is the model's output
column j. No subsetting is applied.

Two assets, because the two records have different lifetimes. The per-epoch
record is appended to during training and must survive a kill mid-run, so it is
a CSV written one row at a time. The reference sidecar is written once, whole,
from the first validation pass -- it is complete or absent, never partial -- so
it is an npz: exact float64, no formatting question, and the ell grid stored as
an integer array rather than a parsed column.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .metrics_record import (
    EllMetrics,
    EllRefMetrics,
    EllStats,
    EllStatsAccumulator,
    RecordSchema,
    specs_from,
)

logger = logging.getLogger(__name__)


class EllMetricsRecorder:
    """Streams per-ell statistics to a wide per-epoch record plus a sidecar.

    The per-epoch record is wide (one row per epoch) because the handler opens
    and closes the file per append; one append per epoch beats one per ell by
    three orders of magnitude. The sidecar is long (one row per ell) because it
    is written once, and because it is the only place the ell grid exists as
    data rather than as substrings of column names.
    """

    def __init__(self,
                 *,
                 enabled: bool,
                 ells: Sequence[int],
                 record_asset,
                 ref_asset,
                 device,
                 rtol: float = 1e-6):
        self.device = device
        self.rtol = rtol
        self._record_asset = record_asset
        self._ref_asset = ref_asset

        self.ells = np.asarray(ells).ravel()
        self.n_ells = int(self.ells.size)

        have_assets = record_asset is not None and ref_asset is not None
        self.enabled = bool(enabled) and have_assets and self.n_ells > 0

        # Announce the state either way. A recorder that is off produces no
        # files and no rows, which is indistinguishable from one that is broken
        # unless it says so.
        if self.enabled:
            logger.info(f"per-ell tracking ENABLED for {self.n_ells} ells")
        elif not bool(enabled):
            logger.info(
                "per-ell tracking DISABLED (enabled=False); no ell records will "
                "be written. Check the config key this flag is read from."
            )
        else:
            reasons = []
            if record_asset is None:
                reasons.append("no record asset")
            if ref_asset is None:
                reasons.append("no reference asset")
            if self.n_ells == 0:
                reasons.append("empty ell grid")
            logger.warning(
                f"per-ell tracking requested but disabled: {', '.join(reasons)}"
            )

        self._epoch_schema: Optional[RecordSchema] = None
        self._ref_schema: Optional[RecordSchema] = None
        self._reference: Optional[EllRefMetrics] = None
        self._acc: Optional[EllStatsAccumulator] = None

    # ------------------------------------------------------------------ #
    def begin_run(self, is_continuation: bool) -> None:
        """Build schemas and open the records. Safe to call when disabled."""
        if not self.enabled:
            return

        self._epoch_schema = RecordSchema(specs_from(EllMetrics), ells=self.ells)
        self._ref_schema = RecordSchema(specs_from(EllRefMetrics), ells=self.ells)
        header = self._epoch_schema.headers(leading=("Epoch", "N"))

        if is_continuation:
            # start() only points the handler at the path; it neither creates
            # the file nor writes a header. Continuing a run whose ell record
            # does not already exist therefore yields a headerless file.
            self._record_asset.start()
            self._reference = self._load_reference()
            if self._reference is None:
                logger.warning(
                    "continuing a run with no readable ell reference sidecar; it "
                    "will be rewritten from the next validation pass and the drift "
                    "check will not cover epochs before now"
                )
        else:
            self._record_asset.write(data=header)
            self._reference = None

        logger.info(
            f"per-ell tracking on for {self.n_ells} ells "
            f"({self.ells[0]}..{self.ells[-1]}); {len(header)} columns per epoch"
        )

    def new_accumulator(self) -> Optional[EllStatsAccumulator]:
        """Open a validation pass. Returns None when disabled, so the result can
        be handed straight to evaluate()."""
        if not self.enabled:
            return None
        if self._acc is not None:
            raise RuntimeError(
                "a validation pass is already open; call record() before "
                "starting another"
            )
        self._acc = EllStatsAccumulator(self.n_ells, device=self.device)
        return self._acc

    def record(self, epoch: int) -> Optional[EllStats]:
        """Close the open pass: write the epoch row, and either establish the
        reference (first pass) or check against it (every pass after)."""
        if not self.enabled:
            return None
        if self._acc is None:
            raise RuntimeError("record() called with no open validation pass")

        acc, self._acc = self._acc, None
        stats = acc.result()
        if stats is None:
            logger.warning(f"epoch {epoch + 1}: no per-ell statistics to record")
            return None

        self._record_asset.append(
            self._epoch_schema.row(stats.per_epoch(),
                                   leading=(epoch + 1, stats.n_valid))
        )

        if self._reference is None:
            self._write_reference(stats.reference())
            self._reference = stats.reference()
        else:
            stats.check_reference(self._reference, rtol=self.rtol,
                                  label=f"epoch {epoch + 1}")
        return stats

    # ------------------------------------------------------------------ #
    def _write_reference(self, ref: EllRefMetrics) -> None:
        """One atomic write of the whole sidecar."""
        self._ref_asset.write(data=dict(ells=self.ells,
                                        mean_true=np.asarray(ref.mean_true),
                                        var_true=np.asarray(ref.var_true)))
        logger.info(f"wrote ell reference sidecar for {self.n_ells} ells")

    def _load_reference(self) -> Optional[EllRefMetrics]:
        """Read the sidecar back on a continuation."""
        try:
            d = self._ref_asset.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"could not read ell reference sidecar: {e!r}")
            return None

        missing = {"ells", "mean_true", "var_true"} - set(d)
        if missing:
            logger.error(f"ell reference sidecar is missing {sorted(missing)}; ignoring it")
            return None

        if not np.array_equal(np.asarray(d["ells"]).ravel(), self.ells):
            logger.error(
                f"ell reference sidecar covers {np.size(d['ells'])} ells but this "
                f"run uses {self.n_ells}; ignoring it. The sidecar is from a "
                f"different configuration."
            )
            return None

        # n_valid is not stored in the sidecar; it is a per-epoch column, where
        # it can catch a short validation pass. -1 skips only the size check.
        return EllRefMetrics(n_valid=-1,
                             mean_true=np.asarray(d["mean_true"]),
                             var_true=np.asarray(d["var_true"]))
