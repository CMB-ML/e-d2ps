import logging

import numpy as np

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.utils.patch_healpix import make_pixel_index_lut


logger = logging.getLogger(__name__)


class MakeLutExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="make_lut")

        self.out_lut: Asset = self.assets_out["lut"]
        out_lut_handler: HealpyMap
        self.nside = cfg.scenario.nside
        self.nside_patch = cfg.model.patches.nside_patch

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        raw_lut = make_pixel_index_lut(nside_obs=self.nside, nside_patches=self.nside_patch)

        nside = self.nside
        regions = [
            [0, 4, 11],
            [1, 5,  8],
            [2, 6,  9],
            [3, 7, 10],
        ]

        lut = np.zeros((4*nside, 3*nside)).astype(np.int64)
        for j, tlps in enumerate(regions):
            for i, tlp in enumerate(tlps):
                r_ids = raw_lut[tlp]
                patch = np.rot90(r_ids, 2)
                lut[j*nside:(j+1)*nside, (i)*nside:(i+1)*nside] = patch

        self.out_lut.write(data=lut)
