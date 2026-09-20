import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import NumpyMap, Config, HealpyMap
from cmbml.utils.planck_instrument import make_instrument, Instrument

from d2ps_nn.utils.masks.mask_composition import gal_mask_label
from d2ps_nn.utils.masks.io import read_channel_mask


logger = logging.getLogger(__name__)


class ArrangeMasksExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str="arrange_masks") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_mask: Asset = self.assets_out["mask"]
        self.out_note: Asset = self.assets_out["note"]
        out_mask_handler: NumpyMap
        out_note_handler: Config

        self.in_mask: Asset = self.assets_in["mask"]
        self.in_lut: Asset = self.assets_in["lut"]
        in_mask_handler: HealpyMap
        in_lut_handler: NumpyMap

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.gal_source = cfg.masks.gal_source
        self.gal_cut_cfg = cfg.masks.gal_cut

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.lut = None
        self.lut_path = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.load_lut()

        for ch in self.instrument.dets:
            mask, mask_info = read_channel_mask(
                self.in_mask, self.name_tracker, ch, self.gal_source, self.gal_cut_cfg
            )

            if mask.size != self.lut.size:
                raise ValueError(f"Mask ({mask.size}) and LUT ({self.lut.size}) sizes differ")
            arranged = mask[self.lut].astype(np.float32)

            note = dict(
                source_mask=mask_info,
                lut=dict(path=self.lut_path, style=self.lut_style),
                dtype=str(arranged.dtype),
            )

            gal_mask = gal_mask_label(ch, self.gal_source, self.gal_cut_cfg).lower()
            context = dict(freq=ch, gal_mask=gal_mask, px_arrange=self.lut_style)
            with self.name_tracker.set_contexts(context):
                self.out_mask.write(data=arranged)
                self.out_note.write(data=note)

    def load_lut(self) -> None:
        with self.name_tracker.set_context("style", self.lut_style):
            self.lut = self.in_lut.read()
            self.lut_path = str(self.in_lut.path)
