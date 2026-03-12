import logging

from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, HealpyMap, NumpyMap
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from d2ps_nn.dataset import TestDataset
from d2ps_nn.models import SimpleEncDec
# from d2ps_nn.utils.ps_scale import make_cl_lognorm_scale
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.get_epochs import get_epochs_with_best


logger = logging.getLogger(__name__)


class PredictExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_ps"]
        out_cmb_handler: HealpyMap

        self.in_feat_asset = FeatureAsset.select(self.assets_in, 
                                                 cfg.features_type, 
                                                 self.instrument)
        self.in_model_asset: Asset = self.assets_in["model"]
        self.in_pos_enc_lut: Asset = self.assets_in["pos_enc_lut"]
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel

        self.scale_ps_choice  = cfg.model.scale_ps

        self.batch_size = cfg.model.test.batch_size
        self._model_epochs = cfg.use_epochs

        self.model = None  # Placeholder for model

    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self) -> None:
        self._model_epochs = get_epochs_with_best(self._model_epochs, self.in_model_asset, self.name_tracker)
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.model = self.init_model()
        self.model.eval().to(self.device)

        self.scale_ps = get_ps_scale(self.scale_ps_choice, self.ells)

        with torch.no_grad():  # We don't need gradients for prediction
            for model_epoch in self.model_epochs:
                for split in self.splits:
                    context_dict = dict(split=split.name, epoch=model_epoch)
                    with self.name_tracker.set_contexts(context_dict):
                        self.in_model_asset.read(model=self.model, epoch=model_epoch)
                        self.process_split(split, model_epoch)

    def process_split(self, split, epoch) -> None:
        # pad_mono_dipole = self.skip_mono_dipole

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
                        # if pad_mono_dipole:
                        #     pred = np.concatenate(([0,0], pred))
                        self.out_cmb_asset.write(data=pred)

    def set_up_dataset(self, template_split: Split) -> TestDataset:
        obs_path_template = self.make_fn_template(template_split, self.in_feat_asset)
        freqs = self.instrument.dets.keys() if self.features_type == "obs" else [None]

        dataset = TestDataset(
            n_sims = template_split.n_sims,
            freqs = freqs,
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=self.in_feat_asset.handler,
            px_arrange=self.px_arrange,
            pos_enc_lut=self.in_pos_enc_lut.read() if self.use_pos_enc_lut else None
            )
        return dataset
