import logging

from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, HealpyMap, NumpyMap
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from d2ps_nn.dataset import TestDataset
from d2ps_nn.encdec import SimpleEncDec
from d2ps_nn.utils.ps_scale import make_cl_lognorm_scale


logger = logging.getLogger(__name__)


class PredictExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_ps"]
        out_cmb_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_model_asset: Asset = self.assets_in["model"]
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel

        self.choose_device(cfg.model.test.device)
        self.batch_size = cfg.model.test.batch_size
        self.dtype = self.dtype_mapping[cfg.model.dtype]

        self.model = None  # Placeholder for model

        self.lmax          = cfg.model.lmax
        self.skip_mono_dipole = cfg.model.train.loss == "wishart"
        self.scale_ps_choice = cfg.model.scale_ps

        self.final_activation = cfg.model.final_activation
        self.model_n_init_features = cfg.model.n_init_features
        self.model_n_levels = cfg.model.n_levels
        self.model_note = cfg.model.note

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        # Set up scaling for post processing
        if self.scale_ps_choice == "Cl":
            self.scale_ps = make_cl_lognorm_scale(self.lmax)
        elif self.scale_ps_choice is None:
            self.scale_ps = 1
        else:
            self.scale_ps = self.scale_ps_choice

        # When using Wishart loss, we need to skip the monopole and dipole
        if self.skip_mono_dipole:
            use_lmax = self.lmax - 2
            self.scale_ps = self.scale_ps[2:]
        else:
            use_lmax = self.lmax

        if self.final_activation == "softplus":
            final_activation = torch.nn.Softplus()
        elif not self.final_activation:
            final_activation = None
        else:
            raise ValueError(f"Unknown final activation: {self.final_activation}")

        # It would likely be safer to have this within the loop, right before read()
        #    But this should work and be faster (especially with larger models)
        self.model = SimpleEncDec(
                                  l_max=use_lmax,
                                  n_in_channels=len(self.instrument.dets),
                                  final_activation=final_activation,
                                  n_init_features=self.model_n_init_features,
                                  n_levels=self.model_n_levels,
                                  note=self.model_note
                                  )

        self.model.eval().to(self.device)

        with torch.no_grad():  # We don't need gradients for prediction
            for model_epoch in self.model_epochs:
                for split in self.splits:
                    context_dict = dict(split=split.name, epoch=model_epoch)
                    with self.name_tracker.set_contexts(context_dict):
                        self.in_model_asset.read(model=self.model, epoch=model_epoch)
                        self.process_split(split, model_epoch)

    def process_split(self, split, epoch) -> None:
        pad_mono_dipole = self.skip_mono_dipole

        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            )
        # for test_features, sim_idcs in dataloader:
        with tqdm(dataloader, desc=f"Epoch {epoch:<{3}}") as pbar:
            for test_features, sim_idcs in pbar:
                test_features = test_features.to(device=self.device, dtype=self.dtype)
                predictions = self.model(test_features).cpu().numpy()
                predictions = predictions / self.scale_ps
                for pred, sim_idx in zip(predictions, sim_idcs):
                    sim_idx = int(sim_idx)
                    with self.name_tracker.set_context("sim_num", sim_idx):
                        if pad_mono_dipole:
                            pred = np.concatenate(([0,0], pred))
                        self.out_cmb_asset.write(data=pred)

    def set_up_dataset(self, template_split: Split) -> TestDataset:
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TestDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler
            )
        return dataset
