import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyMap, Config, HealpyMap

from d2ps_nn.utils.numpy_dtype import get_np_dtype
from d2ps_nn.utils.masks.ps_mask_helper import process_ps_mask, channel_to_field_label


logger = logging.getLogger(__name__)


class PreprocessMasksExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        # The following string must match the pipeline yaml
        if stage_str is None:
            raise NotImplementedError("This is intended to act as a base class")
        super().__init__(cfg, stage_str=stage_str)

        self.instrument: Instrument = make_instrument(cfg=cfg)
        for det in self.instrument.dets:
            if det in [30, 44, 70]:
                raise NotImplementedError("This only handles HFI maps currently.")
                # Need to add the point source map for LFI to extend this.

        self.out_mask: Asset = self.assets_out["mask"]
        self.out_note: Asset = self.assets_out["note"]
        out_mask_handler: NumpyMap
        out_note_handler: Config

        self.in_mask = self.assets_in["mask"]
        self.in_lut: Asset = self.assets_in["lut"]
        in_mask_handler: HealpyMap
        in_lut_handler: NumpyMap

        self.nside_out = self.cfg.scenario.nside
        self.px_arrange = cfg.px_arrange

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.lut = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.load_lut()

    def load_lut(self) -> None:
        with self.name_tracker.set_context("style", self.lut_style):
            self.lut = self.in_lut.read()

    def apply_lut(self, mask) -> np.ndarray:
        n_pix_lut = self.lut.size
        n_pix_mask = mask.size
        if n_pix_lut != n_pix_mask:
            raise ValueError("Mask and LUT should have the same number of pixels")
        return mask[self.lut]


class PrepGalNILCMaskExecutor(PreprocessMasksExecutor):
    pass


class PrepPSMaskExecutor(PreprocessMasksExecutor):
    def __init__(self, cfg, stage_str="preprocess_point_source_mask"):
        super().__init__(cfg, stage_str=stage_str)
        self.smooth_lmax = self.get_stage_element_silent("ps_mask_lmax")
        self.smooth_n_iter = self.get_stage_element_silent("ps_mask_iters")

    def execute(self):
        super().execute()
        for ch, det in self.instrument.dets.items():
            field_label = channel_to_field_label(ch)
            # field_idx = require_field_index(self.in_mask.path, field_label, hdu=1)
            raw = self.in_mask.read(map_field_strs=field_label)[0]
            mask = process_ps_mask(
                raw,
                fwhm=det.fwhm,
                nside_out=self.nside_out,
                lmax=self.smooth_lmax,
                n_iter=self.smooth_n_iter,
            )
            mask = self.apply_lut(mask)
            context = dict(freq=ch, px_arrange=self.px_arrange)
            comments = dict(
                lmax=self.smooth_lmax, 
                fwhm=det.fwhm, 
                n_iter=self.smooth_n_iter
                )
            with self.name_tracker.set_contexts(context):
                self.out_mask.write(data=mask.astype(np.uint8))
                self.out_note.write(data=comments)


class PrepGalCutMaskExecutor(PreprocessMasksExecutor):
    pass
