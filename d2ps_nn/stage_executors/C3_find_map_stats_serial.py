from typing import Dict, List, Any
import logging
from functools import partial

import healpy as hp
import numpy as np
import pysm3.units as u

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


logger = logging.getLogger(__name__)


class FindPxMapStatsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="get_px_map_stats")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_features = FeatureAsset.select(self.assets_out, 
                                                cfg.features_type, 
                                                self.instrument)
        out_handler: NumpyMap

        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        in_obs_map_handler: HealpyMap
        self.nside = cfg.scenario.nside
        # self.map_prep_beam = cfg.get("map_prep_beam", 0)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()

        split: Split = self.splits.get("Train")
        n_sims = split.n_sims

        if n_sims < 1:
            raise ValueError("Need at least one simulation.")
        if n_sims < 2:
            raise ValueError("Need at least two simulations to compute sample std.")

        npix = hp.nside2npix(self.nside)
        channels = [channel for channel, _ in self.in_features.feature_iter()]

        min_maps = {
            channel: np.full(npix, np.inf, dtype=np.float64)
            for channel in channels
        }
        max_maps = {
            channel: np.full(npix, -np.inf, dtype=np.float64)
            for channel in channels
        }
        mean_maps = {
            channel: np.zeros(npix, dtype=np.float64)
            for channel in channels
        }
        m2_maps = {
            channel: np.zeros(npix, dtype=np.float64)
            for channel in channels
        }

        count = 0
        # map_prep_beam = u.Quantity(self.map_prep_beam, u.degree).to(u.radian).value
        # logger.info(f"Using {self.map_prep_beam} degree beam on maps.")

        for sim in tqdm(split.iter_sims(), total=n_sims):
            logger.debug(f"Updating conditioning stats for sim {sim}.")
            context = dict(sim_num=sim, split=split.name)
            with self.name_tracker.set_contexts(context):
                count += 1
                for channel, asset in self.in_features.feature_iter():
                    x = asset.read(precision=np.float64)[0].value

                    # x = hp.smoothing(x, fwhm=map_prep_beam)

                    min_maps[channel] = np.minimum(min_maps[channel], x)
                    max_maps[channel] = np.maximum(max_maps[channel], x)

                    delta = x - mean_maps[channel]
                    mean_maps[channel] += delta / count
                    delta2 = x - mean_maps[channel]
                    m2_maps[channel] += delta * delta2

        stdv_maps = {
            channel: np.sqrt(m2_maps[channel] / (count - 1))
            for channel in channels
        }

        for channel, asset in self.out_features.feature_iter():
            out_maps = np.stack([
                min_maps[channel],
                max_maps[channel],
                mean_maps[channel],
                stdv_maps[channel],
            ])
            # with self.name_tracker.set_context("map_prep_beam", self.map_prep_beam):
            asset.write(data=out_maps)
