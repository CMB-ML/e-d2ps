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
from d2ps_fcn.stage_executors._base_otf_executors import BaseOTFExecutor


logger = logging.getLogger(__name__)


class InferOTFExecutor(BaseOTFExecutor):
    def __init__(self, cfg: DictConfig, stage_str="infer_otf") -> None:
        super().__init__(cfg, stage_str=stage_str)
        # BasePyTorchModelExecutor.__init__(self, cfg, stage_str=stage_str)

        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]
        self.spectrum_type = norm_ps_type(cfg.spectrum_type)  # "theory" | "realization"

        # Assets
        self.in_features:  Asset = self.assets_in["cmb_map"]
        self.in_map_stats: Asset = self.assets_in["cmb_stats"]
        self.in_rps_stats: Asset = self.assets_in["rps_stats"]
        self.in_model:     Asset = self.assets_in["model"]

        self.out_pred:     Asset = self.assets_out["cmb_ps"]

        # Config
        self.infer_batch_size = cfg.model.train.valid_batch_size
        self.n_epochs         = cfg.model.train.n_epochs
        self.n_batches        = cfg.model.train.n_batches
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self._model_epochs = cfg.use_epochs

        logger.info(f"map_processor type: {type(self.map_processor).__name__}")
        logger.info(f"label_scaler: {self.label_scaler}")

        self.model = None  # Placeholder
    
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
        dl_bank = self.load_dl_bank()
        self.fit_label_scaler(dl_bank)
        logger.info(f"Done loading bank of Dls")
        
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

    # def init_model(self):
    #     model_dict = dict(
    #         arch_str=self.fcn_string,
    #         n_in=hp.nside2npix(self.nside),
    #         n_out=len(self.ells))
    #     model = FCNFromString(**model_dict)
    #     logger.info(f"Model dict: {model_dict}")
    #     return model

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
