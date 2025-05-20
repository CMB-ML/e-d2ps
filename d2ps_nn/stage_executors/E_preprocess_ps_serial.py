import logging

import numpy as np

from omegaconf import DictConfig

from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.utils import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyPowerSpectrum  # Import for typing hint
from cmbml.core.asset_handlers import TextPowerSpectrum
from cmbml.core.asset_handlers import CambPowerSpectrum
from d2ps_nn.utils.ps_scale import make_cl_lognorm_scale


logger = logging.getLogger(__name__)


class PreprocessSpectraExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess_spectra")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_ps: Asset = self.assets_out["cmb_ps"]
        out_ps_handler: NumpyPowerSpectrum
        out_ps_handler: TextPowerSpectrum

        self.in_cmb_ps: Asset = self.assets_in["cmb_ps"]
        in_cmb_ps_handler: CambPowerSpectrum

        self.lmax = cfg.model.lmax
        self.scale_ps_choice = cfg.model.scale_ps
        # If using Wishart, we need to skip the monopole and dipole
        self.skip_mono_dipole = cfg.model.train.loss == "wishart"

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()

        # Assumption: power spectra are currently in Dl ("log-normalized")
        if self.scale_ps_choice == "Cl":
            self.scale_ps = make_cl_lognorm_scale(self.lmax)
        elif self.scale_ps_choice is None:
            self.scale_ps = 1

        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, 
                      split: Split) -> None:
        # A split is composed of simulations
        for sim in tqdm(split.iter_sims()):
            logger.debug(f"Preprocessing sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(split)

    def process_sim(self, split) -> None:
        # Process CMB power spectrum
        use_alt_path = split.ps_fidu_fixed
        if not use_alt_path:
            # logger.warning("TODO: Implement fixed per-split paths.")
            use_alt_path = False

        cmb_ps = self.in_cmb_ps.read(use_alt_path=use_alt_path)
        cmb_ps = cmb_ps[:self.lmax + 1]

        cmb_ps *= self.scale_ps

        # If using Wishart, we need to skip the monopole and dipole
        if self.skip_mono_dipole:
            cmb_ps = cmb_ps[2:]

        self.out_cmb_ps.write(data=cmb_ps)
