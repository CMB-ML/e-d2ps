from typing import Dict, List, Any
import logging
from functools import partial

import numpy as np

from omegaconf import DictConfig

from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyPowerSpectrum  # Import for typing hint
from cmbml.core.asset_handlers import NumpyMap
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers import CambPowerSpectrum
from cmbml.core.asset_handlers import HealpyMap
from cmbml.utils.map_fields_helper import map_field_str2int
from d2ps_nn.utils.minmax_scale import minmax_scale
from d2ps_nn.utils.select_features import FeatureAsset


logger = logging.getLogger(__name__)


class PreprocessMapsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess_maps")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        out_obs_patch: Asset = self.assets_out["obs_maps"]
        out_cmb_patch: Asset = self.assets_out["cmb_map"]
        out_patches_handler: NumpyMap

        if cfg.features_type == "cmb":
            self.out_feat = out_cmb_patch
        elif cfg.features_type == "obs":
            self.out_feat = out_obs_patch
        else:
            msg = f"Unsupported features type: {cfg.features_type}. Supported types are 'cmb' and 'obs'."
            raise ValueError(msg)

        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_lut: Asset = self.assets_in["lut"]
        in_obs_map_handler: HealpyMap
        in_dataset_stats_handler: Config
        in_lut_handler: NumpyMap

        self.scaling = cfg.model.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)

        self.lut = None
        self.dataset_stats = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        self.load_lut()
        self.load_dataset_stats()
        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def load_lut(self) -> None:
        self.lut = self.in_lut.read()

    def load_dataset_stats(self) -> None:
        # TODO: Use a class to better handle scaling/normalization
        if self.scaling == "minmax":
            self.dataset_stats = self.in_dataset_stats.read()

    def process_split(self, 
                      split: Split) -> None:
        # A split is composed of simulations
        for sim in tqdm(split.iter_sims()):
            logger.debug(f"Preprocessing sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(split)

    def process_sim(self, split) -> None:
        # Process observations
        for channel, asset in self.in_features.feature_iter():
            obs_map = asset.read()
            obs_map = obs_map.astype(np.float32)
            map_stats = self.get_dataset_stats(channel)
            out = self.process_map(obs_map, map_stats)
            self.out_feat.write(data=out)

    def process_map(self,
                    map_data: np.ndarray,  # Or Astropy Quantity
                    dataset_stats: Dict[str, Dict[str, Any]],
                    ) -> None:
        """
        From map data, extract patches, scale, and save them to an asset.
        """
        out = []
        for field_str in self.map_fields:
            field_int = map_field_str2int(field_str)
            m = map_data[field_int][self.lut]
            if self.scaling == "minmax":
                m = minmax_scale(m, 
                                 dataset_stats[field_str]["vmin"], 
                                 dataset_stats[field_str]["vmax"])
            out.append(m)
        out = np.array(out)
        return out

    def get_dataset_stats(self, freq):
        dataset_stats = None
        try:
            dataset_stats = self.dataset_stats[freq]
        except TypeError:
            pass  # Allow TypeError to go through? Unknown why this is here.
        except Exception:
            raise TypeError(f"No stats for {freq} in file {self.in_dataset_stats.path}")
        return dataset_stats
