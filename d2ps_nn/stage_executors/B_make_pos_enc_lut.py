import logging

import numpy as np
import healpy as hp
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap


logger = logging.getLogger(__name__)


class MakePosEncLutExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="make_pos_enc_lut")

        self.out_pos_enc_lut: Asset = self.assets_out["pos_enc_lut"]
        out_lut_handler: HealpyMap

        self.in_lut: Asset = self.assets_in["lut"]
        in_obs_map_handler: HealpyMap

        self.nside = cfg.scenario.nside

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        ri_map = np.arange(hp.nside2npix(self.nside))
        x, y, z = hp.pix2vec(self.nside, ri_map)
        lut = self.in_lut.read()
        pos_enc_lut = np.stack([x[lut], 
                                y[lut], 
                                z[lut]],
                                axis=1    # Stack along first axis -> (12, 3, N, N)
                                )
        pos_enc_lut = pos_enc_lut.astype(np.float32)
        self.out_pos_enc_lut.write(data=pos_enc_lut)
