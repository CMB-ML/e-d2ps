from typing import Dict
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import Config, NumpyPowerSpectrum
# from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.scale_coef import compute_stacked_stats, make_coef_processor as make_processor


logger = logging.getLogger(__name__)


class BundleParamsExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str="params_bundle") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_bundle: Asset = self.assets_out["params_bundle"]
        out_bundle_handler: NumpyPowerSpectrum
        
        self.in_params = self.assets_in["cosmo_config"]
        in_params_handler: Config

        self.force_new = cfg.get("force_new", False)

        # self.do_run = cfg.src.input == "params"
        self.log_As = cfg.log_As

    def execute(self) -> None:
        # if not self.do_run:
        #     logger.info("Not using params. Skipping stage.")
        #     return
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


class PreprocessStatParamsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="prep_stat_params")

        self.out_stats: Asset = self.assets_out["params_stats"]
        out_coef_handler: NumpyPowerSpectrum

        self.in_cosmo_params: Asset = self.assets_in["params_bundle"]
        in_params_handler: NumpyPowerSpectrum

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        split = self.splits.get("Train")
        with self.name_tracker.set_context("split", split.name):
            data = self.in_cosmo_params.read()
        stats = compute_stacked_stats(data)
        self.out_stats.write(data=stats)


class PreprocessPrepParamsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="prep_scale_params")

        self.out_cosmo_params: Asset = self.assets_out["cosmo_params"]
        out_params_handler: NumpyPowerSpectrum

        self.in_cosmo_params: Asset = self.assets_in["cosmo_params"]
        self.in_cosmo_params_stats: Asset = self.assets_in["params_stats"]
        in_params_handler: NumpyPowerSpectrum
        in_params_stats_handler: NumpyPowerSpectrum

        # if cfg.scale_params is None:
            

        self.scaler = make_processor(cfg.scale_params, coef_stats_asset=self.in_cosmo_params_stats)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        
        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split()
    
    def process_split(self):
        data = self.in_cosmo_params.read()
        data = self.scaler.preprocess(data)
        self.out_cosmo_params.write(data=data)
