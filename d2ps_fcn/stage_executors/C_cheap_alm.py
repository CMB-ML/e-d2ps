import logging

import numpy as np
from tqdm import tqdm
import healpy as hp

from omegaconf import DictConfig

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbml.core.asset_handlers.ps_handler import NumpyPowerSpectrum
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint


logger = logging.getLogger(__name__)


class CheapMakeRealAlmsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_real_alms")

        self.out_alm_bundle: Asset = self.assets_out.get("real_alms", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_map_bundle: Asset = self.assets_in["cmb_map"]
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside

        self.lmax = cfg.model.get("lmax", None)
        self.n_iters = cfg.model.analysis.get("n_map2alm_iters")

        if cfg.scenario.cmb_beam != 0:
            raise NotImplementedError("This executor does not handle beamed CMB maps")

        if self.cfg.map_fields != "I":
            raise NotImplementedError("Only intensity maps are currently supported.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        sims = self.in_cmb_map_bundle.read()
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        alms = []
        for sim in tqdm(split.iter_sims()):
            new_alm = self.process_sim(sims[sim])
            alms.append(new_alm)
        alms = np.array(alms)
        self.out_alm_bundle.write(data=alms)

    def process_sim(self, sim) -> None:
        alms = hp.map2alm(sim, lmax=self.lmax, iter=10)
        return alms
