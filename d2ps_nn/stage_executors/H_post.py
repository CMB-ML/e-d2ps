import logging

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import Config, HealpyMap, NumpyMap
from d2ps_nn.utils.minmax_scale import minmax_unscale
from cmbml.utils.map_fields_helper import map_field_str2int


logger = logging.getLogger(__name__)


class PostExecutor(BaseStageExecutor):
    """
    Goal: Reassemble a map from patches.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="post")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_handler: HealpyMap

        self.in_lut_asset: Asset = self.assets_in["lut"]
        self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        in_cmb_handler: HealpyMap
        in_lut_handler: NumpyMap
        in_dataset_stats_handler: Config

        self.scaling = cfg.model.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)
        
        self.dataset_stats = None
        self.lut = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.load_dataset_stats()
        self.load_lut()

        for model_epoch in self.model_epochs:
            for split in self.splits:
                context_dict = dict(split=split.name, epoch=model_epoch)
                with self.name_tracker.set_contexts(context_dict):
                    self.process_split(split)

    def load_dataset_stats(self) -> None:
        """
        Load the dataset stats from the asset.
        """
        if self.scaling == "minmax":
            self.dataset_stats = self.in_dataset_stats.read()

    def load_lut(self) -> None:
        """
        Load the LUT from the asset.
        """
        self.lut = self.in_lut_asset.read()

    def process_split(self, split) -> None:
        for sim in tqdm(split.iter_sims()):
            logger.debug(f"Postprocessing sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        # Process CMB
        cmb_map = self.in_cmb_asset.read()
        map_stats = self.dataset_stats['cmb']
        out = self.process_map(cmb_map, map_stats)
        self.out_cmb_asset.write(data=out)

    def process_map(self, in_map, dataset_stats):
        """
        Process a map using the model.
        """
        # Unscale the map
        # Current map shape is F x W x H
        F = in_map.shape[0]
        # Rearrange the map to be F x Nside

        out_map = np.zeros((F, hp.nside2npix(self.nside)), dtype=in_map.dtype)
        for field_str in self.map_fields:
            # Rearrange the map
            field_int = map_field_str2int(field_str)
            out_map[field_int][self.lut] = in_map[field_int]
            
            # Rescale the map
            if self.scaling == "minmax":
                out_map = minmax_unscale(out_map, 
                                        dataset_stats[field_str]["vmin"],
                                        dataset_stats[field_str]["vmax"])

        return out_map
