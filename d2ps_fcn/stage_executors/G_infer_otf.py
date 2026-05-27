import logging

# from tqdm import tqdm
import time

import healpy as hp
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_ps import (
    SpectrumAsset, 
    SpectrumAssetNames,
    norm_ps_type
)

from d2ps_fcn.models.fcn_from_string import FCNFromString
from d2ps_fcn.scaler_hydra_helper import make_scaler_from_prep_spec
from d2ps_nn.utils.px_scale import make_px_processor, MapProcessor
from d2ps_fcn.dataset_otf import TrainingMapSampler, make_map_sampler
from d2ps_fcn.dataset_fixed import FixedFeatureDataset
from d2ps_fcn.realization_spectra import make_real_spectra_maker
from d2ps_fcn.stage_executors.F_train_otf import TrainOTFExecutor
from d2ps_nn.utils.get_epochs import get_epochs_with_best


logger = logging.getLogger(__name__)


class InferOTFExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="infer_otf") -> None:
        super().__init__(cfg, stage_str=stage_str)
        # BasePyTorchModelExecutor.__init__(self, cfg, stage_str=stage_str)

        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]
        self.spectrum_type = norm_ps_type(cfg.spectrum_type)  # "theory" | "realization"

        # Assets
        self.ps_bundle:    Asset = self.assets_in["ps_bundle"]  # For label scaler
        self.in_features:  Asset = self.assets_in["cmb_map"]
        self.in_map_stats: Asset = self.assets_in["cmb_stats"]
        self.in_rps_stats: Asset = self.assets_in["rps_stats"]
        self.in_model:     Asset = self.assets_in["model"]

        self.out_pred:     Asset = self.assets_out["cmb_ps"]

        # Config
        self.infer_batch_size = cfg.model.train.valid_batch_size
        self.n_batches        = cfg.model.train.n_batches
        self.n_epochs         = cfg.model.train.n_epochs
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.out_power        = cfg.out_power
        self.activation       = cfg.model.activation
        self.fcn_string       = cfg.fcn_string

        self.sht_engine       = str(cfg.sim_engine).lower()
         
        logger.info(f"Running for {self.n_epochs} epochs.")

        self.label_prep       = cfg.model.prep_label
        self.feature_prep     = cfg.model.prep_feature

        # Processor constructed in __init__ since it only wires asset references
        self.map_processor = make_px_processor(mode=self.feature_prep, cfg=cfg, stage_str=stage_str)

        # Scalers constructed here; fitted in execute()
        self.label_scaler   = make_scaler_from_prep_spec(self.label_prep)
        logger.info(f"map_processor type: {type(self.map_processor).__name__}")
        logger.info(f"label_scaler: {self.label_scaler}")

        self.model = None  # Placeholder
        self._model_epochs = cfg.use_epochs
    
    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self._model_epochs = get_epochs_with_best(self._model_epochs, self.in_model, self.name_tracker)
        self.choose_device()

        # Load bank and fit scalers
        logger.debug(f"Loading bank of Dls")
        dl_bank = self.ps_bundle.read()           # (n_spectra, some_lmax)
        if dl_bank.shape[1] < self.lmax+1:
            raise ValueError("Dl bank does not have enough ell. Please regenerate it or change lmax.")
        dl_bank = dl_bank[:, :self.lmax+1]
        logger.info(f"Done loading bank of Dls")

        if self.spectrum_type == "theory":
            dl_for_labels = dl_bank[:, 2:]            # ells 2..lmax
            self.label_scaler.fit(dl_for_labels)
            n_label_ells = dl_for_labels.shape[1]
        else:  # "realization"
            rps_stats = self.in_rps_stats.read()      # (4, lmax+1)
            rps_for_labels = rps_stats[:, 2:]         # (4, lmax-1), ells 2..lmax
            self.label_scaler.fit_from_stats(rps_for_labels)
            n_label_ells = rps_for_labels.shape[1]

        if n_label_ells != len(self.ells):
            raise ValueError(
                f"Mismatch: {self.spectrum_type} labels have {n_label_ells} ells "
                f"but model expects {len(self.ells)} output channels."
            )
        
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(epoch)

    def process_epoch(self, epoch):
        self.model = self.init_model()
        context = dict(epoch=epoch)
        with self.name_tracker.set_contexts(context):
            self.in_model.read(model=self.model, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split):
        dataset = self.set_up_fixed_dataset(split)
        dataloader = DataLoader(
            dataset,
            batch_size=self.infer_batch_size,
            shuffle=False,
        )

        bundle = []
        with torch.no_grad():
            for features in dataloader:
                output = self.model(features)
                output = self.label_scaler.inverse_transform_tensor(output).cpu().numpy()
                bundle.append(output)

        out = np.vstack(bundle)
        self.out_pred.write(data=out)

    def init_model(self):
        model_dict = dict(
            arch_str=self.fcn_string,
            n_in=hp.nside2npix(self.nside),
            n_out=len(self.ells))
        model = FCNFromString(**model_dict)
        logger.info(f"Model dict: {model_dict}")
        return model

    def set_up_fixed_dataset(self, split: Split) -> FixedFeatureDataset:
        with self.name_tracker.set_context("split", split.name):
            feat_path = self.make_fn_template(split, self.in_features)

        dataset = FixedFeatureDataset(
            n_sims=split.n_sims,
            feature_path=feat_path,
            feature_handler=self.in_features.handler,
            feature_scaler=self.map_processor,
            label_scaler=self.label_scaler,
            dtype=self.dtype,
            fit_scalers=False,
            device=self.device,
        )
        return dataset
