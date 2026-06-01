import logging

import numpy as np

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from d2ps_nn.utils.lut_arrange import cmbNNCS_nest2lut, channel_nest2lut


logger = logging.getLogger(__name__)


class MakeLutExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="make_lut")

        self.out_lut: Asset = self.assets_out["lut"]
        out_lut_handler: HealpyMap
        self.nside = cfg.scenario.nside
        self.nside_patch = cfg.model.patches.get("nside_patch", None)

        if self.nside_patch is not None and self.nside_patch != self.nside:
            raise NotImplementedError("Nside of each patch must be same as Nside")

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.lut_method = {
            None: None,
            "none": None,
            "null": None,
            "cmbNNCS": cmbNNCS_nest2lut,
            "channel": channel_nest2lut
        }[self.lut_style]

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        if self.lut_method is None:
            logger.info("No shape LUT is defined in the configs.")
            return

        lut = self.lut_method(self.nside, self.nside_patch)

        with self.name_tracker.set_context("style", self.lut_style):
            self.out_lut.write(data=lut)
