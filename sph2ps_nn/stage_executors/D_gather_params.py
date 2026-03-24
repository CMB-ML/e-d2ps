from typing import Dict
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class GatherParamsExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str="params_bundle") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_bundle: Asset = self.assets_out["params_bundle"]
        
        self.in_params = self.assets_in["cosmo_config"]
        self.force_new = cfg.force_new

        self.do_run = cfg.src.input == "params"
        self.log_As    = cfg.log_As

    def execute(self) -> None:
        if not self.do_run:
            logger.info("Not using params. Skipping stage.")
            return
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            log_As_flag = "_logAs" if self.log_As else ""
            context = dict(
                logAs_flag=log_As_flag,
                split=split.name
            )
            with self.name_tracker.set_contexts(context):
                self.process(split)

    def process(self, split: Split) -> None:
        if split.ps_fidu_fixed:
            params: Dict = self.in_params.read(use_alt_path=split.ps_fidu_fixed)
            params.pop("pivot_scalar", None)
            params: np.ndarray = np.array(list(params.values()))
            self.out_bundle.write(data=params)
            return

        all_params = []
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                try:
                    params: Dict = self.in_params.read()
                except AttributeError:
                    params: Dict = self.in_params.read(use_alt_path=False)
                params.pop("pivot_scalar", None)
                if self.log_As:
                    params["As"] = np.log(1e10 * params["As"])
                params: np.ndarray = np.array(list(params.values()))
                all_params.append(params)
        all_params = np.array(all_params)
        self.out_bundle.write(data=all_params)
