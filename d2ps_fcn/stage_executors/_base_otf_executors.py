"""Shared base classes for the FCN model_init / train / infer executors.

Hierarchy:
    BasePyTorchModelExecutor          (framework)
    └── BaseFCNExecutor               (this file)
        ├── ModelInitExecutor
        └── BaseOTFExecutor           (this file)
            ├── TrainOTFExecutor
            └── InferOTFExecutor

Extension points:
- New model architectures: override / extend init_model() and _compute_n_input()
  on BaseFCNExecutor. Today both only know about FCNFromString.
- New spectrum sources or scaler kinds: extend fit_label_scaler() on
  BaseOTFExecutor.
"""

import logging

import healpy as hp
import numpy as np
import torch
from omegaconf import DictConfig

from cmbml.core import Asset
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_ps import norm_ps_type
from d2ps_nn.utils.px_scale import make_px_processor
from d2ps_nn.utils.px_shape import make_px_shaper
from d2ps_fcn.scaler_hydra_helper import make_scaler_from_prep_spec
from d2ps_fcn.models.fcn_from_string import FCNFromString

from d2ps_fcn.models import (
    TLPEncAggDec, 
    TLPEnc2AggDec, 
    SimpleEncDec
)
from d2ps_nn.models import (
    MobileViTCMBRegressor
)
# TODO: Re-add?
# from d2ps_nn.models.polynomial_layers import (
#     PolynomialSpectrumLayer,
#     IdentitySpectrumLayer
# )
# from d2ps_nn.models.cosmo_param_layer import ParameterSpectrumLayer
# from d2ps_nn.models.wrapped_models import WrappedSpectrumModel


logger = logging.getLogger(__name__)


class BaseFCNExecutor(BasePyTorchModelExecutor):
    """Common config reads + model construction for every FCN executor.

    Holds only what all three stages need. Anything spectrum-, scaler-, or
    asset-specific belongs on a more specialized subclass.
    """

    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Universal config reads
        self.ells       = parse_ells(cfg.model.use_ells)
        self.lmax       = self.ells[-1]
        self.dtype      = self.dtype_mapping[cfg.model.dtype]
        self.model_num = cfg.model.model_num

        # input_kind today is only set on the model_init config, but the
        # value it determines (n_input) is a property of the model itself,
        # so it lives on the base. Default to 'maps' to preserve existing
        # train/infer config files.
        input_kind = getattr(cfg, "input_kind", "maps")
        if input_kind in ("alms", "alm"):
            raise NotImplementedError(
                f"input_kind={input_kind!r} not supported; 'alms' are to be reintroduced later."
            )
        if input_kind not in ("alms", "alm", "maps", "map"):
            raise NotImplementedError(
                f"input_kind={input_kind!r} not supported; use 'alms' or 'maps'."
            )
        self.input_kind = input_kind
        self.features_type = "cmb"  # placeholder
        self.use_pos_enc_lut = cfg.get("use_pos_enc_lut", False)

        # ############################
        # # Spectrum-layer parameters
        # ############################
        self.backbone_n_outputs = len(self.ells)
        # self.spectrum_layer_kind = cfg.spectrum_layer_kind
        # self.return_mode = cfg.return_mode

        # if self.spectrum_layer_kind not in ["identity", "polynomial", "cosmo_param"]:
        #     raise ValueError(
        #         f"cfg.spectrum_layer_kind={self.spectrum_layer_kind!r} not recognized. "
        #         "Expected 'identity', 'polynomial', or 'cosmo_param'."
        #     )

        # if self.return_mode not in ["spectrum", "coefs", "both"]:
        #     raise ValueError(
        #         f"cfg.return_mode={self.return_mode!r} not recognized. "
        #         "Expected 'spectrum', 'coefs', or 'both'."
        #     )

        # if self.spectrum_layer_kind == "cosmo_param" and self.return_mode != "coefs":
        #     raise NotImplementedError(
        #         f"When using cosmological params, return_mode must be set to coefs."
        #     )

        # if self.spectrum_layer_kind == "polynomial":
        #     self.basis_kind = cfg.basis_kind
        #     self.degree = cfg.degree
        #     self.out_range = tuple(cfg.out_range)

        #     if self.degree < 0:
        #         raise ValueError(f"cfg.degree must be >= 0, got {self.degree}")

        #     self.backbone_n_outputs = self.degree + 1
        # elif self.spectrum_layer_kind == "cosmo_param":
        #     self.basis_kind = None
        #     self.degree = len(cfg.cosmo_param_names)
        #     self.out_range = None
        #     self.backbone_n_outputs = self.degree
        #     self.pivot_scalar = cfg.pivot_scalar
        #     self.param_names = cfg.cosmo_param_names
        # else:
        #     self.basis_kind = None
        #     self.degree = None
        #     self.out_range = None
        #     self.backbone_n_outputs = len(self.ells)

        ############################
        # Model specific
        ############################
        self.nn_model = cfg.nn_model

        if self.nn_model == "fcn":
            self.init_backbone = self.init_simple_backbone
            self.fcn_string = cfg.fcn_string

        elif self.nn_model == "MobileViTCMBRegressor":
            self.init_backbone = self.init_mobileViT_backbone
            self.norm_kind     = cfg.norm_kind
        
        elif self.nn_model in ["SimpleEncDec", "TLPEnc2AggDec", "TLPEncAggDec"]:
            self.init_backbone = self.init_simple_backbone
            self.fcn_string = cfg.fcn_string

            self.model_n_init_features = cfg.model.n_init_features
            self.model_n_levels = cfg.model.n_levels
            self.norm_kind = cfg.model.norm_kind
            self.activation = cfg.model.activation
            self.dropout_rate = cfg.model.dropout_rate
            self.down_reductions = cfg.model.down_reductions
            self.spatial_reduction = cfg.model.spatial_reduction
            # self.final_activation = cfg.model.final_activation
            if self.activation is None:
                raise ValueError("Activation cannot be None. Please specify 'relu' or 'gelu' in config.")        
        else:
            raise ValueError(f"Model {self.nn_model} not recognized.")
        
        if self.nn_model in ["TLPEnc2AggDec", "TLPEncAggDec"]:
            self.face_reduction = cfg.model.face_reduction
        
        # Other
        self.model_note = cfg.model.get("note", None)

    def _compute_n_input(self) -> int:
        """Single source of truth for the model's input dimension.

        init_model() uses it to size the first layer. ModelInitExecutor uses
        it to build a dummy forward pass.
        """
        if self.input_kind in ("alms", "alm"):
            return hp.Alm.getsize(self.lmax)
        return hp.nside2npix(self.nside)

    def init_model(self):
        backbone_layer = self.init_backbone()
        # spectrum_layer = self.init_spectrum_layer()

        # model = WrappedSpectrumModel(
        #     backbone_layer=backbone_layer,
        #     spectrum_layer=spectrum_layer,
        #     return_mode=self.return_mode,
        # )

        # model.model_dict = dict(
        #     wrapper_class=type(model).__name__,
        #     return_mode=self.return_mode,
        #     n_input_channels=self._get_n_input_channels(),
        #     n_target_ells=len(self.ells),
        #     backbone_n_outputs=self.backbone_n_outputs,
        #     backbone_layer_class=type(backbone_layer).__name__,
        #     backbone_layer_kwargs=backbone_layer.model_dict,
        #     spectrum_layer_class=type(spectrum_layer).__name__,
        #     spectrum_layer_kwargs=spectrum_layer.model_dict,
        # )

        # return model.to(self.device)
        return backbone_layer

    def init_simple_backbone(self):
        # Select backbone class
        MODEL_MAP = {
            "SimpleEncDec": SimpleEncDec,
            "TLPEncAggDec": TLPEncAggDec,
            "TLPEnc2AggDec": TLPEnc2AggDec,
            "fcn": FCNFromString,
        }

        if self.nn_model == "fcn":
            model_dict = dict(
                arch_str=self.fcn_string,
                n_in=self._compute_n_input(),
                n_out=len(self.ells),
            )
            model = FCNFromString(**model_dict)
            logger.info(f"Model dict: {model_dict}")
            return model

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
            # final_activation=self.final_activation,
            decoder_factory=make_decoder_factory(self.fcn_string),
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

    # def init_spectrum_layer(self):
    #     if self.spectrum_layer_kind == "identity":
    #         layer_dict = dict(n_outputs=len(self.ells))
    #         layer = IdentitySpectrumLayer(**layer_dict)

    #     elif self.spectrum_layer_kind == "polynomial":
    #         layer_dict = dict(
    #             n_ell=len(self.ells),
    #             degree=self.degree,
    #             basis_kind=self.basis_kind,
    #             out_range=self.out_range,
    #             dtype=self.dtype,
    #         )
    #         layer = PolynomialSpectrumLayer(**layer_dict)

    #     elif self.spectrum_layer_kind == "cosmo_param":
    #         layer_dict = dict(
    #             param_names = self.param_names,
    #             ells = self.ells,
    #             pivot_scalar=self.pivot_scalar
    #         )
    #         layer = ParameterSpectrumLayer(**layer_dict)

    #     else:
    #         raise ValueError(
    #             f"spectrum_layer_kind={self.spectrum_layer_kind!r} not recognized."
    #         )

    #     logger.info(f"Spectrum layer parameters: {layer_dict}")

    #     layer.model_dict = layer_dict
    #     logger.info("Spectrum layer created!")

    #     return layer

    def _get_n_input_channels(self) -> int:
        n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1
        if self.use_pos_enc_lut:
            n_channels += 3
        return n_channels


class BaseOTFExecutor(BaseFCNExecutor):
    """Adds spectrum + scaler + processor setup shared by train and infer.

    ModelInit does not need any of this and inherits directly from
    BaseFCNExecutor.
    """

    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Shared assets
        self.in_rps_stats: Asset = self.assets_in["rps_stats"]
        self.in_dl_bank:       Asset = self.assets_in["dl_bank"]
        self.in_lut:           Asset = self.assets_in["lut"]

        # Spectrum + SHT
        self.spectrum_type = norm_ps_type(cfg.spectrum_type)  # "theory" | "realization"
        self.sht_engine    = str(cfg.sim_engine).lower()

        # Preprocessing config
        self.label_prep    = cfg.model.prep_label
        self.feature_prep  = cfg.model.prep_feature
        self.px_arrange    = cfg.px_arrange

        # Processor + scaler: constructed here, fitted in execute() via
        # fit_label_scaler() below.
        self.map_processor = make_px_processor(mode=self.feature_prep, cfg=cfg, stage_str=stage_str)
        self.map_shaper    = make_px_shaper(cfg=cfg, stage_str=stage_str)
        self.label_scaler  = make_scaler_from_prep_spec(self.label_prep)
        logger.info(f"map_processor type: {type(self.map_processor).__name__}")
        logger.info(f"label_scaler: {self.label_scaler}")       

    def load_dl_bank(self) -> np.ndarray:
        """Read a Dl bank from `asset`, validate, clip to lmax+1."""
        dl_bank = self.in_dl_bank.read()
        if dl_bank.shape[1] < self.lmax + 1:
            raise ValueError(
                "Dl bank does not have enough ell. Regenerate it or change lmax."
            )
        dl_bank = dl_bank[:, : self.lmax + 1]
        return dl_bank

    def fit_label_scaler(self, dl_bank) -> int:
        """Fit self.label_scaler from the supplied Dl bank.

        Raises if the bank is too short or if the label ell count doesn't match 
        the model's output channel count.

        Subclasses call this near the top of execute() with whichever bank
        they own (in_train_cmb_ps for training, ps_bundle for inference).
        """
        if self.spectrum_type == "theory":
            dl_for_labels = dl_bank[:, 2:]            # ells 2..lmax
            self.label_scaler.fit(dl_for_labels)
            n_label_ells = dl_for_labels.shape[1]
        else:  # "realization"
            rps_stats = self.in_rps_stats.read()      # (4, lmax+1)
            rps_for_labels = rps_stats[:, 2:]
            self.label_scaler.fit_from_stats(rps_for_labels)
            n_label_ells = rps_for_labels.shape[1]

        if n_label_ells != len(self.ells):
            raise ValueError(
                f"Mismatch: {self.spectrum_type} labels have {n_label_ells} "
                f"ells but model expects {len(self.ells)} output channels."
            )


def make_decoder_factory(dec_arch_str):
    """Config string -> decoder_factory, or None to use the model's default.

    The 'N' head is mandatory: a 'P' (pixel-pair) head would expand n_in to
    n_in**2, which is meaningless for a decoder operating on aggregator output.
    """
    if dec_arch_str is None:
        return None
    head = dec_arch_str.split("_", 1)[0]
    if head != "N":
        raise ValueError(
            f"decoder dec_arch_str must use the plain 'N' head, got '{head}'."
        )
    def factory(n_in, n_out):
        return FCNFromString(arch_str=dec_arch_str, n_in=n_in, n_out=n_out)
    return factory
