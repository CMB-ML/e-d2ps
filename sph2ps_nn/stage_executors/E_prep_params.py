from typing import Dict
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor


logger = logging.getLogger(__name__)


class PrepParamsExecutor(BaseStageExecutor):
    """
    Goal: Get statistics of cosmological parameters for preprocessing.
    """
    def __init__(self, cfg: DictConfig, stage_str="prep_params") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_stats = self.assets_out["param_stats"]

        self.in_params: Asset = self.assets_in["params_bundle"]

        self.do_run = cfg.src.input == "params"
        self.log_As = cfg.log_As  # For filename flag only

    def execute(self) -> None:
        if not self.do_run:
            logger.info("Not using params. Skipping stage.")
            return
        logger.debug(f"Running {self.__class__.__name__} execute()")
        split: Split = self.splits.get("Train")
        log_As_flag = "_logAs" if self.log_As else ""
        context = dict(
                logAs_flag=log_As_flag,
                split=split.name
            )
        with self.name_tracker.set_contexts(context):
            self.process(split)

    def process(self, split: Split):
        if split.ps_fidu_fixed:
            raise ValueError("Cannot get stats for a single training instance!")
        data = self.in_params.read()
        stats = dict(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            vmin=np.min(data, axis=0),
            vmax=np.max(data, axis=0)
        )
        self.out_stats.write(data=stats)
