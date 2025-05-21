import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum


logger = logging.getLogger(__name__)


class PSBundleExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="bundle")

        self.out_ps_pred: Asset = self.assets_out["ps_pred"]
        self.out_ps_real: Asset = self.assets_out["ps_real"]
        out_ps_pred_handler: NumpyPowerSpectrum
        out_ps_real_handler: NumpyPowerSpectrum

        self.in_ps_pred: Asset = self.assets_in["ps_pred"]
        self.in_ps_real: Asset = self.assets_in["ps_real"]
        in_ps_pred_handler: TextPowerSpectrum
        in_ps_real_handler: NumpyPowerSpectrum

        # Not going to clip the power spectra for now...
        # self.lmax          = cfg.model.lmax

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                # Handle the single lump of realization power spectra
                self.process(split, self.in_ps_real, self.out_ps_real)

                # Handler predictions of power spectra for each epoch
                for model_epoch in self.model_epochs:
                    with self.name_tracker.set_context("epoch", model_epoch):
                        self.process(split, self.in_ps_pred, self.out_ps_pred)

    def process(self, split: Split, in_h: Asset, out_h: Asset) -> None:
        """
        Get the power spectra, save the power spectra in a single file. 
        Written to be agnostic to realization or simulation, using generic 
        handlers.

        Parameters
        ----------
        split : Split
            The split to process.
        in_h : Asset
            The input asset handler. Must provide a numpy array.
        out_h : Asset
            The output asset handler. Must provide a numpy array.
        """
        data = []
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                spectrum = in_h.read()
                data.append(spectrum)
        data = np.array(data)
        # data = data[:, :self.lmax + 1]
        out_h.write(data=data)
