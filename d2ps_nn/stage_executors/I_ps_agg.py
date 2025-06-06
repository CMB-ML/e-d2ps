from typing import Dict, List, NamedTuple, Callable
import logging
from pathlib import Path

from functools import partial
from multiprocessing import Pool, Manager
import numpy as np

from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset, AssetWithPathAlts, GenericHandler
from cmbml.analysis.px_statistics import get_func
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    pred_asset: FrozenAsset
    base_asset: FrozenAsset
    baseline_label: str
    split_name: str
    sim_num: str
    epoch: int
    ells: np.ndarray


class PowerSpectrumAggExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_agg")

        self.out_agg_report: Asset = self.assets_out["agg_report"]
        out_report_handler: PandasCsvHandler

        # self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: PandasCsvHandler

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        
        report = self.in_report.read()
        agg_df = report.groupby(['split', 'epoch', 'baseline']).agg(['mean', 'std']).reset_index()
        self.out_agg_report.write(data=agg_df)
