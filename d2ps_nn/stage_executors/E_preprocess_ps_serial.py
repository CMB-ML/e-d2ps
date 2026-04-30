import logging

import numpy as np

from omegaconf import DictConfig

from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset, AssetWithPathAlts
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyPowerSpectrum  # Import for typing hint
from cmbml.core.asset_handlers import TextPowerSpectrum
from cmbml.core.asset_handlers import CambPowerSpectrum
# from d2ps_nn.utils.ps_scale import make_cl_lognorm_scale
from d2ps_nn.utils.ps_scale import make_ps_processor
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class PreprocessSpectraExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess_spectra")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_ps: Asset = self.assets_out["cmb_ps"]
        out_ps_handler: NumpyPowerSpectrum
        out_ps_handler: TextPowerSpectrum

        self.in_cmb_ps: AssetWithPathAlts = self.assets_in["cmb_ps"]
        in_cmb_ps_handler: CambPowerSpectrum

        self.ells = parse_ells(cfg.model.use_ells)
        self.scale_ps = make_ps_processor(cfg.model.scale_ps, cfg, stage_str="preprocess_spectra")

        # If using Wishart, we need to skip the monopole and dipole
        # self.skip_mono_dipole = cfg.model.train.loss == "wishart"

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()

        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, 
                      split: Split) -> None:
        # A split is composed of simulations
        # No need to handle fixed ps splits; those are only used for comparison
        for sim in tqdm(split.iter_sims()):
            logger.debug(f"Preprocessing sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        cmb_ps = self.in_cmb_ps.read(use_alt_path=False)
        cmb_ps = cmb_ps[self.ells]

        cmb_ps = self.scale_ps.preprocess(cmb_ps)

        # If using Wishart, we need to skip the monopole and dipole
        # if self.skip_mono_dipole:
            # cmb_ps = cmb_ps[2:]

        self.out_cmb_ps.write(data=cmb_ps)
