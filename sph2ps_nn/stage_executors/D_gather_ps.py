import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum
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

        self.ells = parse_ells(cfg.model.use_ells, self.lmax)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            context = dict(split=split.name, ellmax=self.lmax)
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
        data = data[:, self.ells]
        self.out_ps.write(data=data)

class BundleThryExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        self.lmax = cfg.lmax_out
        super().__init__(cfg, stage_str="bundle_thry")

    def process(self, split: Split) -> None:
        if split.ps_fidu_fixed:
            data = self.in_ps.read(use_alt_path=split.ps_fidu_fixed)
            data = data[self.ells]
            self.out_ps.write(data=data)
            return 
        super().process(split)

class BundleRealExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        self.lmax = cfg.lmax_in
        super().__init__(cfg, stage_str="bundle_real")
