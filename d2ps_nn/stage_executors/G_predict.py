import logging
import re
from pathlib import Path

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
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.ps_scale import make_ps_processor


logger = logging.getLogger(__name__)


class PredictExecutor(BaseD2PSExecutor):
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
        self.in_mask: Asset = self.assets_in["mask"]
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel

        self.lut_style = cfg.model.patches.get("lut_style", None)

        # How it runs
        self.batch_size = cfg.model.test.batch_size
        self._model_epochs = cfg.use_epochs

        # How data is processed
        self.scale_ps = make_ps_processor(cfg.model.scale_ps, cfg)
        self.scale_px = cfg.model.scale_px

        self.model = None  # Placeholder for model
        self.use_mask = cfg.model.mask_features

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

        with torch.no_grad():
            for requested_epoch in self.model_epochs:
                label, concrete = self._resolve_epoch(requested_epoch)
                with self.name_tracker.set_context("epoch", concrete):
                    self.in_model_asset.read(model=self.model, epoch=concrete)
                logger.info("Model loaded: %s (labeled '%s')", concrete, label)

                for split in self.splits:
                    with self.name_tracker.set_contexts(dict(split=split.name, epoch=label)):
                        self.process_split(split, label)

    def _resolve_epoch(self, requested_epoch: str) -> tuple[str, str]:
        """Map a requested epoch to (label, concrete).

        label    -> what outputs are tagged with (the request; stable for downstream)
        concrete -> which checkpoint file actually gets loaded

        Identical for everything except "best", which prefers an explicit
        model_best.pth and otherwise resolves to the newest model_best_ep_##.pth.
        """
        if requested_epoch != "best":
            return requested_epoch, requested_epoch

        with self.name_tracker.set_context("epoch", "best"):
            if self.in_model_asset.path.exists():
                return "best", "best"
            model_dir = self.in_model_asset.path.parent

        concrete = self._find_latest_best_checkpoint(model_dir)
        logger.info("model_best.pth not found; 'best' resolves to %s", concrete)
        return "best", concrete

    def _find_latest_best_checkpoint(self, model_dir: Path) -> str:
        """Return the epoch token for the highest-numbered model_best_ep checkpoint,
        e.g. 'best_ep14' or 'best_ep_114' — whichever the file on disk uses.
        Derived from the actual filename so it round-trips through model_{epoch}.pth.
        """
        pattern = re.compile(r"model_best_ep_?(\d+)\.pth$")   # tolerate optional underscore
        best_path, best_ep = None, -1
        for p in model_dir.glob("model_best_ep*.pth"):        # glob w/o underscore too
            if (m := pattern.search(p.name)) and (ep := int(m.group(1))) > best_ep:
                best_path, best_ep = p, ep
        if best_path is None:
            raise FileNotFoundError(
                f"No 'best' checkpoint in {model_dir}: neither model_best.pth "
                f"nor any model_best_ep##.pth was found."
            )
        return best_path.stem.removeprefix("model_")          # 'model_best_ep14' -> 'best_ep14'

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
            pos_enc_lut=self.get_pos_enc_lut(),
            mask=self.get_mask()
            )
        return dataset

    def get_pos_enc_lut(self):
        if not self.use_pos_enc_lut:
            logger.info("No positional encoding will be used.")
            return None
        with self.name_tracker.set_context("style", self.lut_style):
            pos_enc_lut = self.in_pos_enc_lut.read() 
            logger.info("Positional encoding will be used.")
        return pos_enc_lut

    def get_mask(self):
        if not self.use_mask:
            logger.info("Features will not be masked.")
            return None
        with self.name_tracker.set_context("px_arrange", self.px_arrange):
            mask = self.in_mask.read()
            logger.info("Features will be masked.")
        return mask
