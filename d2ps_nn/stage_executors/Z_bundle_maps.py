import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import HealpyMap


logger = logging.getLogger(__name__)


class BundleMapsExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str="bundle_maps") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_map: Asset = self.assets_out["map_bundle"]
        self.in_map: Asset = self.assets_in["cmb_map"]
        in_map_handler: HealpyMap

        # if cfg.scenario.nside > 32:
        #     raise ValueError("This method was developed for use with 2000 Nside=32 maps.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        epoch_iter = [None] if self.model_epochs is None else self.model_epochs

        for split in self.splits:
            for epoch in epoch_iter:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)
            logger.info(f"Done bundling {split.name} CMB maps.")

    def process(self, split: Split) -> None:
        data = []
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                cmb_map = self.in_map.read()[0]
                data.append(cmb_map)
        data = np.array(data)
        self.out_map.write(data=data)
