"""Per-epoch Planck prediction recording as a collaborator object.

Companion to EllMetricsRecorder, but deliberately thinner. There is no dataset
to stream over and no reference to drift against: Planck is a single fixed
observation with no training-style label, so each epoch we run the model on it
once and append the post-processed predicted spectrum as one wide row.

    self.pl_recorder.begin_run(is_continuation)         # once
    self.pl_recorder.record(model, features, post, e)   # per epoch

Disabled recorders no-op, so the calls sit unguarded next to the ell recorder.
Columns are D_ell{ell} for ell in self.ells, in model-output order. The row is
the *post-processed* prediction, so it lands in the same physical units as a
published Planck TT spectrum and needs no further scaling to compare offline.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PlanckRecorder:
    def __init__(self, *, enabled: bool, ells: Sequence[int], record_asset):
        self._record_asset = record_asset
        self.ells = np.asarray(ells).ravel()
        self.n_ells = int(self.ells.size)

        have_asset = record_asset is not None
        self.enabled = bool(enabled) and have_asset and self.n_ells > 0

        if self.enabled:
            logger.info(f"Planck prediction tracking ENABLED for {self.n_ells} ells")
        elif not bool(enabled):
            logger.info("Planck prediction tracking DISABLED (enabled=False)")
        else:
            reasons = []
            if record_asset is None:
                reasons.append("no record asset")
            if self.n_ells == 0:
                reasons.append("empty ell grid")
            logger.warning(
                f"Planck prediction tracking requested but disabled: {', '.join(reasons)}"
            )

    def begin_run(self, is_continuation: bool) -> None:
        if not self.enabled:
            return
        header = ["Epoch"] + [f"D_ell{ell}" for ell in self.ells]
        if is_continuation:
            # start() only points the handler at the path; continuing a run whose
            # Planck record does not yet exist therefore leaves it headerless.
            self._record_asset.start()
        else:
            self._record_asset.write(data=header)
        logger.info(
            f"Planck prediction tracking on for {self.n_ells} ells "
            f"({self.ells[0]}..{self.ells[-1]}); {len(header)} columns per epoch"
        )

    def record(self, model, features, post, epoch: int) -> Optional[torch.Tensor]:
        """Forward the fixed Planck observation once and append one row.

        `features` must already be in the exact format the training dataset
        produces (same px_arrange, same positional encoding, batch dim included).
        This class does no feature engineering -- it only forwards. See the note
        in the executor's set_up_planck.
        """
        if not self.enabled:
            return None

        was_training = model.training
        model.eval()
        p = next(model.parameters())
        try:
            with torch.inference_mode():
                x = features.to(device=p.device, dtype=p.dtype)
                pred = post(model(x))          # (1, n_ells), physical units
        finally:
            if was_training:
                model.train()

        pred = pred.reshape(-1)
        if pred.numel() != self.n_ells:
            logger.error(
                f"epoch {epoch + 1}: Planck prediction has {pred.numel()} values "
                f"but {self.n_ells} ell columns; not recording this epoch"
            )
            return None

        self._record_asset.append([epoch + 1] + pred.detach().cpu().tolist())
        return pred