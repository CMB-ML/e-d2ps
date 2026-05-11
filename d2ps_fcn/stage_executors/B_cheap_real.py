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
from cmbml.utils.physics_ps import get_auto_ps_result
from cmbml.utils.physics_beam import NoBeam, GaussianBeam
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class CheapMakeRealPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_real_ps")

        self.out_auto_real: Asset = self.assets_out.get("auto_real", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_map_real: Asset = self.assets_in["cmb_map_real"]
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside

        self.lmax = cfg.model.get("lmax", None)
        self.anafast_iters = cfg.model.analysis.get("ps_anafast_iters")

        if cfg.scenario.cmb_beam != 0:
            raise NotImplementedError("This executor does not handle beamed CMB maps")

        if self.cfg.map_fields != "I":
            raise NotImplementedError("Only intensity maps are currently supported.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        # Get power spectrum for realization
        real_map: np.ndarray = self.in_cmb_map_real.read()
        if real_map.shape[0] == 3 and self.map_fields == "I":
            real_map = real_map[0]
        self.make_real_ps(real_map)

    def make_real_ps(self, real_map):
        auto_real_ps = get_auto_ps_result(real_map,
                                          lmax=self.lmax,
                                          n_iter=self.anafast_iters,
                                          is_convolved=False)
        ps = auto_real_ps.deconv_dl
        self.out_auto_real.write(data=ps.value)
