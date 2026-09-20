import logging

import numpy as np
from omegaconf import DictConfig, ListConfig
import pysm3.units as u

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyMap, Config, HealpyMap

from cmbml.utils.physics_mask import downgrade_mask
from d2ps_nn.utils.masks.ps_mask_helper import process_ps_mask, channel_to_field_label
from d2ps_nn.utils.masks.mask_composition import gal_mask_label, compose_masks, provenance_cards
import warnings
from astropy.io.fits.verify import VerifyWarning

from cmbml.utils.fits_inspection import get_header_cards


logger = logging.getLogger(__name__)


class ComposeMasksExecutor(BaseStageExecutor):
    def __init__(self, cfg, stage_str="compose_masks"):
        super().__init__(cfg, stage_str=stage_str)
        self.instrument = make_instrument(cfg=cfg)
        self.gal_source = cfg.masks.gal_source
        self.gal_cut_cfg = cfg.masks.gal_cut

        self.in_ps = self.assets_in["ps_mask"]
        self.in_nilc = self.assets_in["nilc_mask"]
        self.in_galcut = self.assets_in["galcut_mask"]
        self.out_mask = self.assets_out["mask"]

    def execute(self):
        for ch in self.instrument.dets:
            with self.name_tracker.set_context("freq", ch):
                ps = self.in_ps.read()[0]
                ps_cards = get_header_cards(self.in_ps.path)
            gal_label = gal_mask_label(ch, self.gal_source, self.gal_cut_cfg)
            gal, gal_cards = self._read_gal(gal_label)
            mask = compose_masks(ps, gal)

            logger.info(f"{ch} GHz: PS + {gal_label}, f_sky = {mask.mean():.4f}")
            header = (
                [("GAL_SRC", gal_label, "Galactic mask")]
                + provenance_cards(ps_cards, "PS")
                + provenance_cards(gal_cards, "GAL")
            )
            with self.name_tracker.set_contexts(dict(freq=ch, gal_mask=gal_label.lower())):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", VerifyWarning)  # HIERARCH creation notices
                    self.out_mask.write(data=mask.astype(np.uint8), extra_header=header)

    def _read_gal(self, gal_label):
        if gal_label == "NILC":
            return self.in_nilc.read()[0], get_header_cards(self.in_nilc.path)
        with self.name_tracker.set_context("gal_cut", gal_label):
            return self.in_galcut.read()[0], get_header_cards(self.in_galcut.path)
