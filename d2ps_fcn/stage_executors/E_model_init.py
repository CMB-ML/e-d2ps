import logging

import healpy as hp
import numpy as np
from omegaconf import DictConfig
import random
import torch

from cmbml.core import Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.sims.random_seed_manager import SeedFactory

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.select_ps import SpectrumAsset

# from d2ps_fcn.models.fcn import FCNDouble, FCNSingle
from d2ps_fcn.models.fcn_from_string import FCNFromString


logger = logging.getLogger(__name__)


class ModelInitExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="model_init") -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Output shape
        self.ells = parse_ells(cfg.model.use_ells)

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.out_power          = cfg.out_power
        self.fcn_string         = cfg.fcn_string
        self.dtype              = self.dtype_mapping[cfg.model.dtype]

        self.model_init = cfg.model_init
        self.model_seed_template = cfg.model.model_seed
        self.seed_factory = SeedFactory(self.model_seed_template)
        self.input_kind = cfg.input_kind
        self.lmax = self.ells[-1]
        if self.input_kind not in ["alms", "maps"]:
            raise NotImplementedError("Must use input_kind of either 'alms' or 'maps'")

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")

        seed = self.seed_factory.get_seed(**dict(model_init=self.model_init))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = self.init_model()

        if self.input_kind in ["alms", "alm"]:
            n_input = hp.Alm.getsize(self.lmax)
        elif self.input_kind in ["maps", "map"]:
            n_input = hp.nside2npix(self.nside)
        else:
            raise NotImplementedError("Must use input_kind of either 'alms' or 'maps'")

        with torch.no_grad():
            dummy_x = torch.zeros(1, n_input, dtype=self.dtype)
            _ = model(dummy_x)

        context = dict(model_num=self.model_init)
        with self.name_tracker.set_contexts(context):
            self.out_model.write(model=model, epoch="init")

    def init_model(self):
        model_dict = dict(
            arch_str=self.fcn_string,
            n_in=hp.nside2npix(self.nside),
            n_out=len(self.ells))
        model = FCNFromString(**model_dict)
        logger.info(f"Model dict: {model_dict}")
        return model
