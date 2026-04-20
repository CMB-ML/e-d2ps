import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers import NumpyPowerSpectrum

from d2ps_nn.utils.scale_coef import make_coef_processor


logger = logging.getLogger(__name__)


class PreprocessPrepSpectraCoefExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="prep_scale_polycoef")

        self.out_cmb_ps_coef: Asset = self.assets_out["cmb_ps_coef"]
        out_coef_handler: NumpyPowerSpectrum

        self.in_cmb_ps_coef: Asset = self.assets_in["cmb_ps_coef"]
        self.in_cmb_coef_stats: Asset = self.assets_in["coef_stats"]
        in_cmb_ps_handler: NumpyPowerSpectrum

        self.coef_scaler = make_coef_processor(cfg.scale_coef, coef_stats_asset=self.in_cmb_coef_stats)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        
        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split()
    
    def process_split(self):
        data = self.in_cmb_ps_coef.read()
        data = self.coef_scaler.preprocess(data)
        self.out_cmb_ps_coef.write(data=data)
