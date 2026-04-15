import logging

from omegaconf import DictConfig
import torch

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.models import (
    TLPEncAggDec,
    TLPEnc2AggDec,
    SimpleEncDec,
    MobileViTCMBRegressor,
)
from d2ps_nn.models.polynomial_layers import (
    PolynomialSpectrumLayer,
    IdentitySpectrumLayer,
)
from d2ps_nn.models.wrapped_models import WrappedSpectrumModel

from d2ps_nn.utils.loss import get_loss
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset


logger = logging.getLogger(__name__)


class BaseD2PSExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Output shape / target indexing
        self.ells = parse_ells(cfg.model.use_ells)

        # Input shape
        self.features_type = cfg.features_type
        self.use_pos_enc_lut = cfg.use_pos_enc_lut
        self.px_arrange = cfg.px_arrange

        if self.features_type not in ["obs", "cmb"]:
            raise ValueError(
                "Features type must be set. 'obs' and 'cmb' are accepted."
            )
        if self.px_arrange not in ["cmbNNCS", "channel"]:
            raise ValueError(
                "Pixel arrangement must be set. 'channel' and 'cmbNNCS' are accepted."
            )

        # Basic parameters
        self.dtype = self.dtype_mapping[cfg.model.dtype]
        self.nn_model = cfg.nn_model

        # Spectrum-layer parameters
        self.spectrum_layer_kind = cfg.spectrum_layer_kind
        self.return_mode = cfg.return_mode

        if self.spectrum_layer_kind not in ["identity", "polynomial"]:
            raise ValueError(
                f"cfg.spectrum_layer_kind={self.spectrum_layer_kind!r} not recognized. "
                "Expected 'identity' or 'polynomial'."
            )

        if self.return_mode not in ["spectrum", "coeffs", "both"]:
            raise ValueError(
                f"cfg.return_mode={self.return_mode!r} not recognized. "
                "Expected 'spectrum', 'coeffs', or 'both'."
            )

        if self.spectrum_layer_kind == "polynomial":
            self.basis_kind = cfg.basis_kind
            self.degree = cfg.degree
            self.out_range = tuple(cfg.out_range)

            if self.degree < 0:
                raise ValueError(f"cfg.degree must be >= 0, got {self.degree}")

            self.backbone_n_outputs = self.degree + 1
        else:
            self.basis_kind = None
            self.degree = None
            self.out_range = None
            self.backbone_n_outputs = len(self.ells)

        # Model run location
        self.choose_device(cfg.model.train.device)
        if isinstance(self.device, torch.device):
            if self.device.type == "cuda" and self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        elif isinstance(self.device, str) and self.device.startswith("cuda:"):
            torch.cuda.set_device(int(self.device.split(":")[1]))
        elif isinstance(self.device, int):
            torch.cuda.set_device(self.device)

        logger.info(f"Model running on {self.device}")

        # Architecture-specific parameters
        if self.nn_model == "MobileViTCMBRegressor":
            self.init_backbone = self.init_mobileViT_backbone
            self.norm_kind = cfg.norm_kind

        elif self.nn_model in ["SimpleEncDec", "TLPEnc2AggDec", "TLPEncAggDec"]:
            self.init_backbone = self.init_simple_backbone

            self.model_n_init_features = cfg.model.n_init_features
            self.model_n_levels = cfg.model.n_levels

            self.norm_kind = cfg.model.norm_kind
            self.activation = cfg.model.activation
            self.dropout_rate = cfg.model.dropout_rate

            self.downsample_pooling = cfg.model.downsample_pooling
            self.bottleneck_pooling = cfg.model.bottleneck_pooling
            self.final_activation = cfg.model.final_activation

            if self.activation is None:
                raise ValueError(
                    "Activation cannot be None. Please specify 'relu' or 'gelu' in config."
                )

        else:
            raise ValueError(f"Model {self.nn_model} not recognized.")

        # Other
        self.model_note = cfg.model.note

        logger.info(
            "Spectrum layer config: "
            f"kind={self.spectrum_layer_kind}, "
            f"return_mode={self.return_mode}, "
            f"basis_kind={self.basis_kind}, "
            f"degree={self.degree}, "
            f"out_range={self.out_range}, "
            f"backbone_n_outputs={self.backbone_n_outputs}"
        )

    def _get_n_input_channels(self) -> int:
        n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
        if self.use_pos_enc_lut:
            n_channels += 3
        return n_channels

    def _build_spectrum_layer(self):
        if self.spectrum_layer_kind == "identity":
            layer = IdentitySpectrumLayer(n_outputs=len(self.ells))

        elif self.spectrum_layer_kind == "polynomial":
            layer = PolynomialSpectrumLayer(
                n_ell=len(self.ells),
                degree=self.degree,
                basis_kind=self.basis_kind,
                out_range=self.out_range,
                dtype=self.dtype,
            )

        else:
            raise ValueError(
                f"spectrum_layer_kind={self.spectrum_layer_kind!r} not recognized."
            )

        return layer.to(self.device)

    def init_model(self):
        """
        Build:
            backbone -> spectrum_layer -> wrapped model

        The backbone always emits shape (B, self.backbone_n_outputs).

        - identity layer:     backbone_n_outputs = len(self.ells)
        - polynomial layer:   backbone_n_outputs = degree + 1
        """
        backbone = self.init_backbone()
        spectrum_layer = self._build_spectrum_layer()

        model = WrappedSpectrumModel(
            backbone=backbone,
            spectrum_layer=spectrum_layer,
            return_mode=self.return_mode,
        ).to(self.device)

        model.model_dict = dict(
            nn_model=self.nn_model,
            spectrum_layer_kind=self.spectrum_layer_kind,
            return_mode=self.return_mode,
            n_ell=len(self.ells),
            backbone_n_outputs=self.backbone_n_outputs,
            basis_kind=self.basis_kind,
            degree=self.degree,
            out_range=self.out_range,
            note=self.model_note,
        )

        logger.info("Wrapped model created!")
        logger.info(f"Wrapped model params: {model.model_dict}")

        return model

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

        model_dict = dict(
            nside=self.nside,
            n_outputs=self.backbone_n_outputs,
            n_in_channels=n_channels,
            n_init_features=self.model_n_init_features,
            n_levels=self.model_n_levels,
            norm_kind=self.norm_kind,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            final_activation=self.final_activation,
            note=self.model_note,
        )

        if use_model in [SimpleEncDec]:
            model_dict["downsample_pooling"] = self.downsample_pooling
            model_dict["bottleneck_pooling"] = self.bottleneck_pooling

        logger.info(f"Backbone params: {model_dict}")

        model = use_model(**model_dict)
        model.model_dict = model_dict
        model = model.to(self.device)

        logger.info("Backbone created!")

        return model

    def init_mobileViT_backbone(self):
        n_channels = self._get_n_input_channels()

        model_dict = dict(
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

        logger.info(f"Backbone params: {model_dict}")

        model = MobileViTCMBRegressor(**model_dict)
        model.model_dict = model_dict
        model = model.to(self.device)

        logger.info("Backbone created!")

        return model
    