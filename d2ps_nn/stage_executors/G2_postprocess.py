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
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.get_epochs import get_epochs_with_best


logger = logging.getLogger(__name__)


class UnStandardizeSpectraExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="postprocess_spectra_standardize")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_ps: Asset = self.assets_out["cmb_ps"]
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_ps: Asset = self.assets_in["cmb_ps"]
        self.in_mean_ps: Asset = self.assets_in["mean_ps"]
        self.in_stdv_ps: Asset = self.assets_in["stdv_ps"]
        in_cmb_ps_handler: TextPowerSpectrum
        in_stat_ps_handler: NumpyPowerSpectrum

        scale_ps_choice = cfg.model.scale_ps
        if scale_ps_choice is not None:
            raise ValueError("This Executor is incompatible with previous PS scaling.")
        self._model_epochs = cfg.use_epochs

    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        self._model_epochs = get_epochs_with_best(self._model_epochs, self.in_cmb_ps, self.name_tracker)

        self.mean_ps = self.in_mean_ps.read()
        self.stdv_ps = self.in_stdv_ps.read()

        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, 
                      split: Split) -> None:
        for epoch in self.model_epochs:
            for sim in tqdm(split.iter_sims()):
                contexts = dict(epoch=epoch, sim_num=sim)
                logger.debug(f"Preprocessing sim {sim}.")
                with self.name_tracker.set_contexts(contexts):
                    self.process_sim()

    def process_sim(self) -> None:
        cmb_ps = self.in_cmb_ps.read()
        cmb_ps = (cmb_ps * self.stdv_ps) + self.mean_ps
        self.out_cmb_ps.write(data=cmb_ps)
