import logging

import numpy as np
from omegaconf import DictConfig
import random
import torch

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core import Asset
from cmbml.core.asset_handlers.appending_csv_handler import AppendingCsvHandler
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler
from cmbml.core.asset_handlers.pd_parquet_handler import PandasParquetHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.sims.random_seed_manager import SeedFactory

from d2ps_nn.utils.model_summary import trace_module_shapes
from d2ps_fcn.stage_executors._base_otf_executors import BaseFCNExecutor


logger = logging.getLogger(__name__)


class LogToParquetExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str="log_to_parquet") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_parquet: Asset = self.assets_out["loss_parquet"]
        out_model_handler: PandasParquetHandler
        
        self.in_log: Asset = self.assets_in["loss_record"]
        in_log_handler: AppendingCsvHandler

        self.model_seed_template = cfg.model.model_seed
        self.seed_factory = SeedFactory(self.model_seed_template)
        
    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        path = self.in_log.path
        pdh = PandasCsvHandler()
        df = pdh.read(path=path)
        self.out_parquet.write(data=df)
