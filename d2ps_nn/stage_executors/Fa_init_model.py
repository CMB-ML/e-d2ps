import logging

from tqdm import tqdm

import torch
import random
import numpy as np

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from cmbml.sims.random_seed_manager import SeedFactory


logger = logging.getLogger(__name__)


class ModelInitExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig, stage_str="model_init") -> None:
        super().__init__(cfg, stage_str=stage_str)

        # What comes out
        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.model_seed_template = cfg.model.model_seed
        self.model_iter = cfg.model_num
        self.seed_factory = SeedFactory(self.model_seed_template)
        self.permit_bn_creation = cfg.model.get("allow_create_with_norm", False)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        with self.name_tracker.set_context("model_num", self.model_iter):
            if self.out_model.path.exists():
                logger.warning("Model already created. Skipping!")
                return
        if self.norm_kind not in [None, "none", "None"] and not self.permit_bn_creation:
            raise ValueError("Sorry to kill the run. Don't create a model with bn for others to load from.")

        seed = self.seed_factory.get_seed(**dict(iter=self.model_iter))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = self.init_model()

        context = dict(model_num=self.model_iter,
                       pos_enc=self.use_pos_enc_lut)
        with self.name_tracker.set_contexts(context):
            self.out_model.write(model=model, epoch="init")
