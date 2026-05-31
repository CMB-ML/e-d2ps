from typing import Dict, List, Any
import logging
from functools import partial

import healpy as hp
import numpy as np
import pysm3.units as u

import matplotlib.pyplot as plt

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
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class FindPxSafeMeanExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="get_px_safe_mean")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cfg: Asset = self.assets_out["temp_stat"]
        out_handler: Config

        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        self.in_mask = self.assets_in["mask"]
        in_obs_map_handler: HealpyMap

        self.nside = cfg.scenario.nside

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()

        split: Split = self.splits.get("Train")
        n_sims = split.n_sims

        mask = self.prep_mask()

        count = 0

        match_temps = {}
        match_units = {}
        for channel, asset in self.in_features.feature_iter():
            match_temps[channel] = []

        for sim in tqdm(split.iter_sims(), total=n_sims):
            context = dict(sim_num=sim, split=split.name)
            with self.name_tracker.set_contexts(context):
                count += 1
                for channel, asset in self.in_features.feature_iter():
                    m = asset.read(precision=np.float64)[0]
                    match_units[channel] = m.unit
                    m = hp.ma(m)
                    m.mask = 1 - mask  # Invert mask due to convention differences
                    match_temps[channel].append(m.mean())
            break
        match_temp = {}
        for channel, val_list in match_temps.items():
            match_temp[channel] = u.Quantity(np.mean(val_list), match_units[channel])
        
        self.out_cfg.write(data = match_temp)

    def prep_mask(self):
        raw_mask = self.in_mask.read(map_fields=7)  # GAL20 mask is in field 0
        proccessed_mask = downgrade_mask(raw_mask, nside_out=self.nside, threshold=0.9)
        return proccessed_mask
