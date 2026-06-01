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
from d2ps_fcn.scaler_hydra_helper import make_scaler_from_prep_spec
from d2ps_fcn.models.fcn_from_string import FCNFromString


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
        self.fcn_string = cfg.fcn_string
        self.out_power  = cfg.out_power
        self.model_init = cfg.model.model_init

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

    def _compute_n_input(self) -> int:
        """Single source of truth for the model's input dimension.

        init_model() uses it to size the first layer. ModelInitExecutor uses
        it to build a dummy forward pass.
        """
        if self.input_kind in ("alms", "alm"):
            return hp.Alm.getsize(self.lmax)
        return hp.nside2npix(self.nside)

    def init_model(self) -> torch.nn.Module:
        """Construct an untrained model from config.

        Model will either be saved in model_init stage or weights/etc. 
        will be overwritten when model is loaded.
        """
        model_dict = dict(
            arch_str=self.fcn_string,
            n_in=self._compute_n_input(),
            n_out=len(self.ells),
        )
        model = FCNFromString(**model_dict)
        logger.info(f"Model dict: {model_dict}")
        return model


class BaseOTFExecutor(BaseFCNExecutor):
    """Adds spectrum + scaler + processor setup shared by train and infer.

    ModelInit does not need any of this and inherits directly from
    BaseFCNExecutor.
    """

    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Common asset
        self.in_dl_bank:       Asset = self.assets_in["dl_bank"]

        # Spectrum + SHT
        self.spectrum_type = norm_ps_type(cfg.spectrum_type)  # "theory" | "realization"
        self.sht_engine    = str(cfg.sim_engine).lower()

        # Preprocessing config
        self.label_prep    = cfg.model.prep_label
        self.feature_prep  = cfg.model.prep_feature

        # Processor + scaler: constructed here, fitted in execute() via
        # fit_label_scaler() below.
        self.map_processor = make_px_processor(
            mode=self.feature_prep, cfg=cfg, stage_str=stage_str,
        )
        self.label_scaler  = make_scaler_from_prep_spec(self.label_prep)
        logger.info(f"map_processor type: {type(self.map_processor).__name__}")
        logger.info(f"label_scaler: {self.label_scaler}")

        # Shared asset. Other assets are subclass-specific.
        # (rps_stats is unused in the theory branch but cheap to wire.)
        self.in_rps_stats: Asset = self.assets_in["rps_stats"]

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

        Returns n_label_ells. Raises if the bank is too short or if the
        label ell count doesn't match the model's output channel count.

        Subclasses call this near the top of execute() with whichever bank
        they own (in_train_cmb_ps for training, ps_bundle for inference).
        """
        if dl_bank.shape[1] < self.lmax + 1:
            raise ValueError(
                "Dl bank does not have enough ell. "
                "Regenerate it or change lmax."
            )
        dl_bank = dl_bank[:, : self.lmax + 1]

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
        # return n_label_ells
