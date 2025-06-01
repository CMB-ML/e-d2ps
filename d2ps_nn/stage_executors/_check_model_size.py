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
from d2ps_nn.encdec import SimpleEncDec
from d2ps_nn.stage_executors.F_train import get_loss
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class CheckGPULoadExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        self.in_ps_asset: Asset = self.assets_in["cmb_ps"]
        self.in_maps_asset: Asset = self.assets_in["obs_maps"]
        self.in_model: Asset = self.assets_in["model"]
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.dtype         = self.dtype_mapping[cfg.model.dtype]  # TODO: Ensure this is used

        self.choose_device(cfg.model.train.device)  # See parent class
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

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        loss_record_headers = ['Epoch', 'Training Loss', 'Validation Loss']
        self.out_loss_record.write(data=loss_record_headers)

        if self.final_activation == "softplus":
            final_activation = torch.nn.Softplus()
        elif not self.final_activation:
            final_activation = None
        else:
            raise ValueError(f"Unknown final activation: {self.final_activation}")

        logger.info(f"Creating model")
        model = SimpleEncDec(
                             n_ells=len(self.ells),
                             n_in_channels=1,  # CMB Only, not observations
                             final_activation=final_activation,
                             n_init_features=self.model_n_init_features,
                             n_levels=self.model_n_levels,
                             note=self.model_note
                             )

        model = model.to(self.device)
        logger.info(f"Model created!")

        batch_size = self.batch_size
        nside = self.nside
        channels = len(self.instrument.dets)

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)

        input_shape = (batch_size, channels, nside*4, nside*3)
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

        logger.info(f"Memory after forward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        loss = loss_function(output, train_label)
        loss.backward()

        logger.info(f"Memory after backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
