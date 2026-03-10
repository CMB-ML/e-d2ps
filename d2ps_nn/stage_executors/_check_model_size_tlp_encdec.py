import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from d2ps_nn.dataset import TrainDataset
from d2ps_nn.models import TLPEncAggDec
from d2ps_nn.stage_executors.F_train import get_loss
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset


logger = logging.getLogger(__name__)


class CheckGPULoadExecutor_TLPEncDec(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_ps_asset: Asset = self.assets_in["cmb_ps"]
        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap

        self.dtype         = self.dtype_mapping[cfg.model.dtype]  # TODO: Ensure this is used

        self.choose_device(cfg.model.train.device)  # See parent class
        if isinstance(self.device, torch.device):
            if self.device.type == 'cuda' and self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        elif isinstance(self.device, str) and self.device.startswith("cuda:"):
            torch.cuda.set_device(int(self.device.split(":")[1]))
        elif isinstance(self.device, int):
            torch.cuda.set_device(self.device)
        self.batch_size    = cfg.model.train.batch_size
        self.num_workers   = cfg.model.train.num_loader_workers
        self.learning_rate = cfg.model.train.learning_rate
        self.n_epochs      = cfg.model.train.n_epochs
        self.restart_epoch = cfg.model.train.restart_epoch
        self.checkpoint    = cfg.model.train.checkpoint_every
        self.extra_check   = cfg.model.train.extra_check

        self.norm          = None
        self.ells          = parse_ells(cfg.model.use_ells)
        self.loss_key      = cfg.model.train.loss

        self.final_activation = cfg.model.final_activation

        self.model_n_init_features = cfg.model.n_init_features
        self.model_n_levels = cfg.model.n_levels
        self.model_note = cfg.model.note

        self.use_batch_norm_enc = cfg.model.use_batch_norm_enc
        self.use_batch_norm_dec = cfg.model.use_batch_norm_dec
        self.dropout_rate = cfg.model.dropout_rate
        self.activation = cfg.model.activation
        if self.activation is None:
            raise ValueError("Activation cannot be None. Please specify 'relu' or 'gelu' in config.")
        self.pooling = cfg.model.pooling
        if self.pooling is None:
            raise ValueError("Please specify 'avg' or 'max'.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        batch_size = self.batch_size
        nside = self.nside
        n_channels = self.in_features.n_features

        if self.final_activation == "softplus":
            final_activation = torch.nn.Softplus()
        elif not self.final_activation:
            final_activation = None
        else:
            raise ValueError(f"Unknown final activation: {self.final_activation}")

        logger.info(f"Creating model")
        model = TLPEncAggDec(
                             nside=self.nside,
                             n_ells=len(self.ells),
                             n_in_channels=n_channels,  # CMB Only, not observations
                             final_activation=final_activation,
                             n_init_features=self.model_n_init_features,
                             n_levels=self.model_n_levels,
                             use_batch_norm_enc=self.use_batch_norm_enc,   # ADDED
                             use_batch_norm_dec=self.use_batch_norm_dec,   # ADDED
                             dropout_rate=self.dropout_rate,               # ADDED
                             activation=self.activation,                   # ADDED
                             note=self.model_note
                             )

        model = model.to(self.device)
        logger.info(f"Model created!")

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)

        input_shape = (batch_size, 12, n_channels, nside, nside)
        output_shape = (batch_size, len(self.ells))

        # Make random tensor to check model size
        train_features = torch.randn(input_shape).to(self.device)
        train_label = torch.randn(output_shape).to(self.device)
        logger.info(f"{train_features.shape}, {train_label.shape}")
        # exit()

        # Check model size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        output = model(train_features)
        logger.info(f"Output shape: {output.shape}")

        logger.info(f"Memory after forward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        loss = loss_function(output, train_label)
        loss.backward()

        logger.info(f"Memory after backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
