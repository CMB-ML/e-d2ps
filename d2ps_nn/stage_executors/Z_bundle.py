import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class BundleExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_ps: Asset = self.assets_out["ps_bundle"]
        self.in_ps: Asset = self.assets_in["ps"]

        self.ells = parse_ells(cfg.model.use_ells)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        epoch_iter = [None] if self.model_epochs is None else self.model_epochs

        for split in self.splits:
            for epoch in epoch_iter:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)

    def process(self, split: Split) -> None:
        # Prepared to handle single spectra for a split
        data = []
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                try:
                    spectrum = self.in_ps.read()
                except AttributeError:
                    spectrum = self.in_ps.read(use_alt_path=False)
                data.append(spectrum)
        data = np.array(data)
        if self.select_ell:
            data = data[:, self.ells]
        self.out_ps.write(data=data)

class BundlePredExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="bundle_pred")
        self.select_ell = not(cfg.get("trim_ell_after_predict", False))

class BundleTrueExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="bundle_true")
        self.select_ell = True

    def process(self, split: Split) -> None:
        if split.ps_fidu_fixed:
            data = self.in_ps.read(use_alt_path=split.ps_fidu_fixed)
            data = data[self.ells]
            self.out_ps.write(data=data)
            return 
        super().process(split)
