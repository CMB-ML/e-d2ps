import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from d2ps_nn.stage_executors.F_train import get_loss
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset


logger = logging.getLogger(__name__)


class CheckGPULoadExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        self.batch_size    = cfg.model.train.batch_size
        self.loss_key      = cfg.model.train.loss

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        # Set up model and loss function
        model = self.init_model()
        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)

        # Make random tensor to check model size
        input_shape = (self.batch_size, *model.in_shape)
        output_shape = (self.batch_size, len(self.ells))
        train_features = torch.randn(input_shape).to(self.device)
        train_label = torch.randn(output_shape).to(self.device)
        logger.info(f"{train_features.shape}, {train_label.shape}")

        # Check model size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        output = model(train_features)
        logger.info(f"Output shape: {output.shape}")

        logger.info(f"Memory after forward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        loss = loss_function(output, train_label)
        loss.backward()

        logger.info(f"Memory after backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
