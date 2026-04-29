import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, HealpyMap, NumpyMap
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from d2ps_nn.dataset import TestDataset
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.ps_scale import make_ps_processor
from d2ps_nn.utils.scale_coef import make_coef_processor


logger = logging.getLogger(__name__)


class PredictCoefsExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        # What comes out
        self.out_cmb_asset: Asset = self.assets_out["cmb_ps"]
        out_cmb_handler: HealpyMap

        # What goes in
        self.in_feat_asset = FeatureAsset.select(self.assets_in, 
                                                 cfg.features_type, 
                                                 self.instrument)
        self.in_model_asset: Asset = self.assets_in["model"]
        self.in_pos_enc_lut: Asset = self.assets_in["pos_enc_lut"]
        self.in_cmb_coef_stats: Asset = self.assets_in["coef_stats"]
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.coef_scaler = make_coef_processor(cfg.scale_coef, coef_stats_asset=self.in_cmb_coef_stats)

        # How it runs
        self.batch_size = cfg.model.test.batch_size
        self._model_epochs = cfg.use_epochs

        # How data is processed
        self.scale_ps = make_ps_processor(cfg.model.scale_ps, cfg)
        self.scale_px = cfg.model.scale_px

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
        self.model.coef_processor = self.coef_scaler
        self.model.eval().to(self.device)

        with torch.no_grad():  # We don't need gradients for prediction
            for model_epoch in self.model_epochs:
                for split in self.splits:
                    context_dict = dict(split=split.name, epoch=model_epoch)
                    with self.name_tracker.set_contexts(context_dict):
                        self.in_model_asset.read(model=self.model, epoch=model_epoch)
                        self.process_split(split, model_epoch)

    def process_split(self, split, epoch) -> None:
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
                pred = self.model(test_features).cpu().numpy()
                pred = self.scale_ps.postprocess(pred)
                for pred, sim_idx in zip(pred, sim_idcs):
                    sim_idx = int(sim_idx)
                    with self.name_tracker.set_context("sim_num", sim_idx):
                        self.out_cmb_asset.write(data=pred)

    def set_up_dataset(self, template_split: Split) -> TestDataset:
        context = dict(px_arrange=self.px_arrange, scale=self.scale_px)
        with self.name_tracker.set_contexts(context):
            obs_path_template = self.make_fn_template(template_split, self.in_feat_asset)
        freqs = self.instrument.dets.keys() if self.features_type == "obs" else [None]

        dataset = TestDataset(
            n_sims = template_split.n_sims,
            freqs = freqs,
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=self.in_feat_asset.handler,
            px_arrange=self.px_arrange,
            pos_enc_lut=self.get_pos_enc_lut()
            )
        return dataset

    def get_pos_enc_lut(self):
        if not self.use_pos_enc_lut:
            return None
        with self.name_tracker.set_context("style", self.lut_style):
            pos_enc_lut = self.in_pos_enc_lut.read() 
        return pos_enc_lut
