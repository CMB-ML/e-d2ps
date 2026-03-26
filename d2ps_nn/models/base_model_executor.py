import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from d2ps_nn.models import (
    TLPEncAggDec, 
    TLPEnc2AggDec, 
    SimpleEncDec,
    MobileViTCMBRegressor
)
from d2ps_nn.utils.loss import get_loss
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset


logger = logging.getLogger(__name__)


class BaseD2PSExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Output shape
        self.ells = parse_ells(cfg.model.use_ells)

        # Input shape
        self.features_type = cfg.features_type
        self.use_pos_enc_lut = cfg.use_pos_enc_lut
        self.px_arrange = cfg.px_arrange
        if self.features_type not in ["obs", "cmb"]:
            raise ValueError("Features type must be set. "
                "'obs' and 'cmb' are accepted.")
        if self.px_arrange not in ["cmbNNCS", "channel"]:
            raise ValueError("Pixel arrangement must be set. "
                "'channel' and 'cmbNNCS' are accepted.")

        # Basic parameters
        self.dtype = self.dtype_mapping[cfg.model.dtype]
        self.nn_model = cfg.nn_model

        # Model run location
        self.choose_device(cfg.model.train.device)  # See parent class
        if isinstance(self.device, torch.device):
            if self.device.type == 'cuda' and self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        elif isinstance(self.device, str) and self.device.startswith("cuda:"):
            torch.cuda.set_device(int(self.device.split(":")[1]))
        elif isinstance(self.device, int):
            torch.cuda.set_device(self.device)
        logger.info(f"Model running on {self.device}")

        # Architecture
        if self.nn_model == "MobileViTCMBRegressor":
            self.init_model = self.init_mobileViT
            self.norm_kind     = cfg.norm_kind
        
        elif self.nn_model in ["SimpleEncDec", "TLPEnc2AggDec", "TLPEncAggDec"]:
            self.init_model = self.init_simple
            self.model_n_init_features = cfg.model.n_init_features
            self.model_n_levels = cfg.model.n_levels

            self.use_batch_norm_enc = cfg.model.use_batch_norm_enc
            self.use_batch_norm_dec = cfg.model.use_batch_norm_dec
            self.activation = cfg.model.activation
            self.dropout_rate = cfg.model.dropout_rate

            self.pooling = cfg.model.pooling
            self.final_activation = cfg.model.final_activation

            if self.activation is None:
                raise ValueError("Activation cannot be None. Please specify 'relu' or 'gelu' in config.")
        
        else:
            raise ValueError(f"Model {self.nn_model} not recognized.")

        # Other
        self.model_note = cfg.model.note

    def init_simple(self):
        # Get the model class
        if self.nn_model == "TLPEncAggDec":
            use_model = TLPEncAggDec
        elif self.nn_model == "TLPEnc2AggDec":
            use_model = TLPEnc2AggDec
        elif self.nn_model == "SimpleEncDec":
            use_model = SimpleEncDec
        else:
            raise ValueError(f"{use_model} is not configured.")
        logger.info(f"Model: {self.nn_model}")

        # Set up model parameters
        n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
        if self.use_pos_enc_lut:
            n_channels += 3

        model_dict = dict(
            nside=self.nside,
            n_ells=len(self.ells),
            n_in_channels=n_channels,
            n_init_features=self.model_n_init_features,
            n_levels=self.model_n_levels,
            use_batch_norm_enc=self.use_batch_norm_enc,
            use_batch_norm_dec=self.use_batch_norm_dec,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            note=self.model_note
        )

        if self.final_activation == "softplus":
            model_dict["final_activation"] = torch.nn.Softplus()
        elif not self.final_activation:
            model_dict["final_activation"] = None
        else:
            raise ValueError(f"Unknown final activation: {self.final_activation}")

        if use_model in [SimpleEncDec]:
            model_dict["pooling"] = self.pooling

        # Make the model
        logger.info(f"Model params: {model_dict}")
        model = use_model(**model_dict)

        model = model.to(self.device)
        logger.info(f"Model created!")

        return model

    def init_mobileViT(self):
        # Set up model parameters
        n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
        if self.use_pos_enc_lut:
            n_channels += 3

        logger.info(f"Creating model...")
        model_dict = dict(
            nside=self.nside,
            in_channels=n_channels,
            n_outputs=len(self.ells),
            norm_kind=self.norm_kind,
            # widths=(32, 64, 96, 128),
            widths=(24, 48, 64, 96),
            # transformer_dims=(96, 120, 144),
            transformer_dims=(64, 80, 96),
            patch_size=(2, 2),
            # transformer_depths=(2, 2, 2),
            transformer_depths=(1, 1, 1),
            num_heads=(4, 4, 4),
            mlp_ratio=2.0,
            dropout=0.1,
            head_hidden=512,
            # head_hidden=256
        )


        logger.info(f"Model params: {model_dict}")
        model = MobileViTCMBRegressor(**model_dict)
        model = model.to(self.device)

        logger.info(f"Model created!")

        return model
