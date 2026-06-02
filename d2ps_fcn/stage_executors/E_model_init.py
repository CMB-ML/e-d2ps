import logging

import healpy as hp
import numpy as np
from omegaconf import DictConfig
import random
import torch

from cmbml.core import Asset
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.sims.random_seed_manager import SeedFactory

from d2ps_nn.utils.model_summary import trace_module_shapes
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

        records = trace_module_shapes(model, model.example_in_shape, self.dtype)
        logger.info(f"{'Input':<30} {'':<18} {'':<20} -> {str(model.example_in_shape):<20} [{np.prod(model.example_in_shape)}]")
        for name, cls, in_s, out_s in records:
            if cls == "Identity":
                continue
            if in_s == out_s:
                continue
            out_elem = np.prod(out_s)
            logger.info(f"{name:<30} {cls:<18} {str(in_s):<20} -> {str(out_s):<20} [{out_elem}]")

        context = dict(model_num=self.model_num)
        with self.name_tracker.set_contexts(context):
            self.out_model.write(model=model, epoch="init")
