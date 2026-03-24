from typing import Dict
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor


logger = logging.getLogger(__name__)


class PrepSpectraExecutor(BaseStageExecutor):
    """
    Goal: Get statistics of power spectra for preprocessing.
    """
    def __init__(self, cfg: DictConfig, stage_str="prep_spectra") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_stats = self.assets_out["spectra_stats"]

        self.in_ps: Asset = self.assets_in["ps_bundle"]

        self.in_src = cfg.src.input
        self.out_src = cfg.src.output

        self.in_lmax = cfg.get("lmax_in", None)
        self.out_lmax = cfg.lmax_out

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        split: Split = self.splits.get("Train")
        
        # Input
        if self.in_src != "params":
            context = dict(
                split=split.name,
                ellmax=self.in_lmax,
                src=self.in_src
            )
            with self.name_tracker.set_contexts(context):
                self.process(split)

        # Output
        context = dict(
            split=split.name,
            ellmax=self.out_lmax,
            src=self.out_src
        )
        with self.name_tracker.set_contexts(context):
            self.process(split)

    def process(self, split: Split):
        if split.ps_fidu_fixed:
            raise ValueError("Cannot get stats for a single training instance!")
        data = self.in_ps.read()
        stats = dict(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            vmin=np.min(data, axis=0),
            vmax=np.max(data, axis=0)
        )
        self.out_stats.write(data=stats)
