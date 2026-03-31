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
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class FindSpectraStatsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="get_spectra_stats")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_mean_ps: Asset = self.assets_out["mean_ps"]
        self.out_stdv_ps: Asset = self.assets_out["stdv_ps"]
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_ps: AssetWithPathAlts = self.assets_in["cmb_ps"]
        in_cmb_ps_handler: CambPowerSpectrum

        self.ells = parse_ells(cfg.model.use_ells)

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
        all_ps = []
        for sim in tqdm(split.iter_sims()):
            logger.debug(f"Preprocessing sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                all_ps.append(self.process_sim(split))
        all_ps = np.array(all_ps)
        mean_ps = np.mean(all_ps, axis=0)
        stdv_ps = np.std(all_ps, axis=0)
        self.out_mean_ps.write(data=mean_ps)
        self.out_stdv_ps.write(data=stdv_ps)

        log_level = logger.info if stdv_ps.min() > 10 else logger.warning
        log_level(f"Minimum st. dev. was {stdv_ps.min():.2e}. Small values may cause issues.")

    def process_sim(self, split) -> None:
        # Process CMB power spectrum
        use_alt_path = split.ps_fidu_fixed
        if not use_alt_path:
            # logger.warning("TODO: Implement fixed per-split paths.")
            use_alt_path = False

        cmb_ps = self.in_cmb_ps.read(use_alt_path=use_alt_path)
        cmb_ps = cmb_ps[self.ells]
        return cmb_ps
