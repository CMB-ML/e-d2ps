import logging

from tqdm import tqdm

from functools import partial
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
from d2ps_nn.models.polynomial_layers import (
    PolynomialSpectrumLayer,
    IdentitySpectrumLayer
)
from d2ps_nn.models.cosmo_param_layer import ParameterSpectrumLayer
from d2ps_nn.models.wrapped_models import WrappedSpectrumModel

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

        # Spectrum-layer parameters
        self.spectrum_layer_kind = cfg.spectrum_layer_kind
        self.return_mode = cfg.return_mode

        if self.spectrum_layer_kind not in ["identity", "polynomial", "cosmo_param"]:
            raise ValueError(
                f"cfg.spectrum_layer_kind={self.spectrum_layer_kind!r} not recognized. "
                "Expected 'identity', 'polynomial', or 'cosmo_param'."
            )

        if self.return_mode not in ["spectrum", "coefs", "both"]:
            raise ValueError(
                f"cfg.return_mode={self.return_mode!r} not recognized. "
                "Expected 'spectrum', 'coefs', or 'both'."
            )

        if self.spectrum_layer_kind == "cosmo_param" and self.return_mode != "coefs":
            raise NotImplementedError(
                f"When using cosmological params, return_mode must be set to coefs."
            )

        if self.spectrum_layer_kind == "polynomial":
            self.basis_kind = cfg.basis_kind
            self.degree = cfg.degree
            self.out_range = tuple(cfg.out_range)

            if self.degree < 0:
                raise ValueError(f"cfg.degree must be >= 0, got {self.degree}")

            self.backbone_n_outputs = self.degree + 1
        elif self.spectrum_layer_kind == "cosmo_param":
            self.basis_kind = None
            self.degree = len(cfg.cosmo_param_names)
            self.out_range = None
            self.backbone_n_outputs = self.degree
            self.pivot_scalar = cfg.pivot_scalar
            self.param_names = cfg.cosmo_param_names
        else:
            self.basis_kind = None
            self.degree = None
            self.out_range = None
            self.backbone_n_outputs = len(self.ells)

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
            self.init_backbone = self.init_mobileViT
            self.norm_kind     = cfg.norm_kind
        
        elif self.nn_model in ["SimpleEncDec", "TLPEnc2AggDec", "TLPEncAggDec"]:
            self.init_backbone = self.init_simple_backbone

            self.model_n_init_features = cfg.model.n_init_features
            self.model_n_levels = cfg.model.n_levels
            self.norm_kind = cfg.model.norm_kind
            self.activation = cfg.model.activation
            self.dropout_rate = cfg.model.dropout_rate
            self.down_reductions = cfg.model.down_reductions
            self.spatial_reduction = cfg.model.spatial_reduction
            self.final_activation = cfg.model.final_activation
            if self.activation is None:
                raise ValueError("Activation cannot be None. Please specify 'relu' or 'gelu' in config.")        
        else:
            raise ValueError(f"Model {self.nn_model} not recognized.")
        
        if self.nn_model in ["TLPEnc2AggDec", "TLPEncAggDec"]:
            self.face_reduction = cfg.model.face_reduction

        # Other
        self.model_note = cfg.model.note

    def _get_n_input_channels(self) -> int:
        n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
        if self.use_pos_enc_lut:
            n_channels += 3
        return n_channels

    def init_model(self):
        backbone_layer = self.init_backbone()
        spectrum_layer = self.init_spectrum_layer()

        model = WrappedSpectrumModel(
            backbone_layer=backbone_layer,
            spectrum_layer=spectrum_layer,
            return_mode=self.return_mode,
        )

        model.model_dict = dict(
            wrapper_class=type(model).__name__,
            return_mode=self.return_mode,
            n_input_channels=self._get_n_input_channels(),
            n_target_ells=len(self.ells),
            backbone_n_outputs=self.backbone_n_outputs,
            backbone_layer_class=type(backbone_layer).__name__,
            backbone_layer_kwargs=backbone_layer.model_dict,
            spectrum_layer_class=type(spectrum_layer).__name__,
            spectrum_layer_kwargs=spectrum_layer.model_dict,
        )

        return model.to(self.device)

    def init_simple_backbone(self):
        # Select backbone class
        MODEL_MAP = {
            "SimpleEncDec": SimpleEncDec,
            "TLPEncAggDec": TLPEncAggDec,
            "TLPEnc2AggDec": TLPEnc2AggDec,
        }

        use_model = MODEL_MAP[self.nn_model]

        logger.info(f"Backbone: {self.nn_model}")

        n_channels = self._get_n_input_channels()

        layer_dict = dict(
            nside=self.nside,
            n_outputs=self.backbone_n_outputs,
            n_in_channels=n_channels,
            n_init_features=self.model_n_init_features,
            n_levels=self.model_n_levels,
            norm_kind=self.norm_kind,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            down_reductions=self.down_reductions,
            spatial_reduction=self.spatial_reduction,
            final_activation=self.final_activation,
            note=self.model_note,
        )

        if use_model in [TLPEncAggDec, TLPEnc2AggDec]:
            layer_dict["face_reduction"]    = self.face_reduction

        logger.info(f"Backbone params: {layer_dict}")

        layer = use_model(**layer_dict)
        layer.model_dict = layer_dict

        logger.info("Backbone created!")

        return layer

    def init_mobileViT_backbone(self):
        n_channels = self._get_n_input_channels()

        layer_dict = dict(
            nside=self.nside,
            in_channels=n_channels,
            n_outputs=self.backbone_n_outputs,
            norm_kind=self.norm_kind,
            widths=(32, 64, 96, 128),
            transformer_dims=(96, 120, 144),
            patch_size=(2, 2),
            transformer_depths=(2, 2, 2),
            num_heads=(4, 4, 4),
            mlp_ratio=2.0,
            dropout=0.1,
            head_hidden=512,
        )

        logger.info(f"Backbone params: {layer_dict}")

        layer = MobileViTCMBRegressor(**layer_dict)
        layer.model_dict = layer_dict

        logger.info("Backbone created!")

        return layer

    def init_spectrum_layer(self):
        if self.spectrum_layer_kind == "identity":
            layer_dict = dict(n_outputs=len(self.ells))
            layer = IdentitySpectrumLayer(**layer_dict)

        elif self.spectrum_layer_kind == "polynomial":
            layer_dict = dict(
                n_ell=len(self.ells),
                degree=self.degree,
                basis_kind=self.basis_kind,
                out_range=self.out_range,
                dtype=self.dtype,
            )
            layer = PolynomialSpectrumLayer(**layer_dict)

        elif self.spectrum_layer_kind == "cosmo_param":
            layer_dict = dict(
                param_names = self.param_names,
                ells = self.ells,
                pivot_scalar=self.pivot_scalar
            )
            layer = ParameterSpectrumLayer(**layer_dict)

        else:
            raise ValueError(
                f"spectrum_layer_kind={self.spectrum_layer_kind!r} not recognized."
            )

        logger.info(f"Spectrum layer parameters: {layer_dict}")

        layer.model_dict = layer_dict
        logger.info("Spectrum layer created!")

        return layer


    # def init_simple(self):
    #     # Get the model class
    #     if self.nn_model == "TLPEncAggDec":
    #         use_model = TLPEncAggDec
    #     elif self.nn_model == "TLPEnc2AggDec":
    #         use_model = TLPEnc2AggDec
    #     elif self.nn_model == "SimpleEncDec":
    #         use_model = SimpleEncDec
    #     else:
    #         raise ValueError(f"{use_model} is not configured.")
    #     logger.info(f"Model: {self.nn_model}")

    #     # Set up model parameters
    #     n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
    #     if self.use_pos_enc_lut:
    #         n_channels += 3

    #     model_dict = dict(
    #         nside=self.nside,
    #         n_ells=len(self.ells),
    #         n_in_channels=n_channels,
    #         n_init_features=self.model_n_init_features,
    #         n_levels=self.model_n_levels,
    #         norm_kind=self.norm_kind,
    #         dropout_rate=self.dropout_rate,
    #         activation=self.activation,
    #         down_reductions=self.down_reductions,
    #         spatial_reduction=self.spatial_reduction,
    #         final_activation=self.final_activation,
    #         note=self.model_note
    #     )

    #     if use_model in [TLPEncAggDec, TLPEnc2AggDec]:
    #         model_dict["face_reduction"]    = self.face_reduction

    #     # Make the model
    #     logger.info(f"Model params: {model_dict}")
    #     model = use_model(**model_dict)
    #     model.model_dict = model_dict

    #     model = model.to(self.device)
    #     logger.info(f"Model created!")

    #     return model

    # def init_mobileViT(self):
    #     # Set up model parameters
    #     n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
    #     if self.use_pos_enc_lut:
    #         n_channels += 3

    #     logger.info(f"Creating model...")
    #     model_dict = dict(
    #         nside=self.nside,
    #         in_channels=n_channels,
    #         n_outputs=len(self.ells),
    #         norm_kind=self.norm_kind,
    #         widths=(32, 64, 96, 128),
    #         transformer_dims=(96, 120, 144),
    #         patch_size=(2, 2),
    #         transformer_depths=(2, 2, 2),
    #         num_heads=(4, 4, 4),
    #         mlp_ratio=2.0,
    #         dropout=0.1,
    #         head_hidden=512,
    #     )
    #     logger.info(f"Model params: {model_dict}")
    #     model = MobileViTCMBRegressor(**model_dict)
    #     model.model_dict = model_dict
    #     model = model.to(self.device)

    #     logger.info(f"Model created!")

    #     return model
