import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers import CambPowerSpectrum
from d2ps_nn.utils.scale_coef import compute_stacked_stats

logger = logging.getLogger(__name__)


class PreprocessStatSpectraCoefExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="prep_stat_polycoef")

        self.out_stats: Asset = self.assets_out["coef_stats"]
        out_coef_handler: Config

        self.in_cmb_ps_coef: Asset = self.assets_in["cmb_ps_coef"]
        in_cmb_ps_handler: CambPowerSpectrum

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        split = self.splits.get("Train")
        with self.name_tracker.set_context("split", split.name):
            data = self.in_cmb_ps_coef.read()
        stats = compute_stacked_stats(data)
        self.out_stats.write(data=stats)
