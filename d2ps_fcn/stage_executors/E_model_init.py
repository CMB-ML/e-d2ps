import logging

import healpy as hp
import numpy as np
from omegaconf import DictConfig
import random
import torch

from cmbml.core import Asset
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.sims.random_seed_manager import SeedFactory

from d2ps_fcn.stage_executors._base_otf_executors import BaseFCNExecutor


logger = logging.getLogger(__name__)


class ModelInitExecutor(BaseFCNExecutor):
    def __init__(self, cfg: DictConfig, stage_str="model_init") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.model_seed_template = cfg.model.model_seed
        self.seed_factory = SeedFactory(self.model_seed_template)
        
    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")

        seed = self.seed_factory.get_seed(model_num=self.model_num)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = self.init_model()

        with torch.no_grad():
            dummy_x = torch.zeros(1, *model.example_in_shape, dtype=self.dtype)
            d_out = model(dummy_x)
            logger.info(f"Model function checked. Input shape: {dummy_x.shape}, output shape: {d_out.shape}")

        context = dict(model_num=self.model_num)
        with self.name_tracker.set_contexts(context):
            self.out_model.write(model=model, epoch="init")
