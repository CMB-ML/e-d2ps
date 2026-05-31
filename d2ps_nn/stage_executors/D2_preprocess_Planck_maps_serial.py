from typing import Dict, List, Any
import logging
from functools import partial

import pysm3.units as u
import healpy as hp
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
from d2ps_nn.utils.numpy_dtype import get_np_dtype
from d2ps_nn.utils.px_scale import make_px_processor, MapProcessor
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class PreprocessPlanckMapsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess_planck_maps")

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

        # Handles either observations or ground-truth CMB maps
        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        self.in_temp_stat: Asset = self.assets_in["temp_stat"]
        self.in_mask = self.assets_in["mask"]
        self.in_lut: Asset = self.assets_in["lut"]
        in_obs_map_handler: HealpyMap
        in_dataset_stats_handler: Config
        in_lut_handler: NumpyMap

        self.scaling = cfg.model.get("scale_px", None)
        self.scale_px: MapProcessor = make_px_processor(cfg.model.scale_px, cfg)

        self.dtype = get_np_dtype(cfg.model.dtype)

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.lut = None
        self.dataset_stats = None
        self.temp_stat = None
        self.gal_mask = None
        self.nside = cfg.scenario.nside

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        self.load_temp_stat()
        self.gal_mask = self.prep_mask()
        self.load_lut()
        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def load_temp_stat(self) -> None:
        self.temp_stat = self.in_temp_stat.read()

    def load_lut(self) -> None:
        with self.name_tracker.set_context("style", self.lut_style):
            self.lut = self.in_lut.read()

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
            obs_map = asset.read(precision=self.dtype)
            out = self.process_map(obs_map, channel)
            context = dict(px_arrange=self.lut_style,
                           scale=self.scaling)
            with self.name_tracker.set_contexts(context):
                self.out_feat.write(data=out)

    def process_map(self,
                    map_data: np.ndarray,  # Or Astropy Quantity
                    channel
                    ) -> None:
        """
        From map data, extract patches, scale, and save them to an asset.
        """
        out = []
        for field_str in self.map_fields:
            field_int = map_field_str2int(field_str)
            m = map_data[field_int]
            m_mean = self.get_masked_mean(m)
            m = m - (m_mean - self.temp_stat[channel])
            m = self.scale_px.preprocess(m, channel, field_str=field_str)
            out.append(m[self.lut])
        out = np.array(out)
        return out

    def prep_mask(self):
        raw_mask = self.in_mask.read(map_fields=7)  # GAL20 mask is in field 0
        proccessed_mask = downgrade_mask(raw_mask, nside_out=self.nside, threshold=0.9)
        return proccessed_mask

    def get_masked_mean(self, m):
        unit = m.unit
        m = m.copy()
        m = hp.ma(m)
        m.mask = 1 - self.gal_mask
        return u.Quantity(m.mean(), unit)
