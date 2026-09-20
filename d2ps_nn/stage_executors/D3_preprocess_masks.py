import logging

import numpy as np
from omegaconf import DictConfig, ListConfig
import pysm3.units as u

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyMap, Config, HealpyMap

from cmbml.utils.physics_mask import downgrade_mask
from d2ps_nn.utils.masks.ps_mask_helper import process_ps_mask, channel_to_field_label


logger = logging.getLogger(__name__)


class PreprocessMasksExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        # The following string must match the pipeline yaml
        if stage_str is None:
            raise NotImplementedError("This is intended to act as a base class")
        super().__init__(cfg, stage_str=stage_str)

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_mask: Asset = self.assets_out["mask"]
        out_mask_handler: HealpyMap

        self.in_mask = self.assets_in["mask"]
        in_mask_handler: HealpyMap
        self.nside_out = self.cfg.scenario.nside

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")


class PrepGalNILCMaskExecutor(PreprocessMasksExecutor):
    def __init__(self, cfg, stage_str="prep_nilc_mask"):
        super().__init__(cfg, stage_str=stage_str)
        self.mask_threshold = self.cfg.model.analysis.mask_threshold

    def execute(self):
        super().execute()
        mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        mask = downgrade_mask(mask, 
                              nside_out=self.nside_out,
                              threshold=self.mask_threshold)
        # comments = dict(threshold=self.mask_threshold)
        header = [
            ("MSKTHRSH", self.mask_threshold, "downgrade_mask threshold"),
        ]
        self.out_mask.write(data=mask.astype(np.uint8), 
                            extra_header=header,
                            column_names=["MASK"])


class PrepPSMaskExecutor(PreprocessMasksExecutor):
    def __init__(self, cfg, stage_str="prep_point_source_mask"):
        super().__init__(cfg, stage_str=stage_str)
        self.smooth_lmax = self.get_stage_element_silent("ps_mask_lmax")
        self.smooth_n_iter = self.get_stage_element_silent("ps_mask_iters")

        for det in self.instrument.dets:
            if det in [30, 44, 70]:
                raise NotImplementedError("This only handles HFI maps currently.")
                # Need to add the point source map for LFI to extend this.

    def execute(self):
        super().execute()
        for ch, det in self.instrument.dets.items():
            field_label = channel_to_field_label(ch)
            raw = self.in_mask.read(map_field_strs=field_label)[0]
            mask = process_ps_mask(
                raw,
                fwhm=det.fwhm,
                nside_out=self.nside_out,
                lmax=self.smooth_lmax,
                n_iter=self.smooth_n_iter,
            )
            header = [
                ("SRCFIELD", field_label, "Field in source PS mask file"),
                ("BEAMFWHM", float(det.fwhm.to_value(u.arcmin)), "Smoothing FWHM [arcmin]"),
                ("SM_LMAX", self.smooth_lmax, "Smoothing lmax; blank = 3*nside_in-1"),
                ("SM_NITER", self.smooth_n_iter, "Smoothing iter; blank = 3"),
            ]
            with self.name_tracker.set_context("freq", ch):
                self.out_mask.write(data=mask.astype(np.uint8), 
                                    extra_header=header,
                                    column_names=["MASK"])


class PrepGalCutMaskExecutor(PreprocessMasksExecutor):
    def __init__(self, cfg, stage_str="prep_galcut_mask"):
        super().__init__(cfg, stage_str=stage_str)
        self.mask_threshold = self.cfg.model.analysis.mask_threshold
        gal_cuts = self.get_stage_element("gal_cuts")
        self.gal_cuts = self.fmt_gal_cuts(gal_cuts)

    @staticmethod
    def fmt_gal_cuts(gal_cuts):
        if not isinstance(gal_cuts, (list, ListConfig)):
            gal_cuts = [gal_cuts]
        out = []
        for gc in gal_cuts:
            gc = str(gc).strip().upper()
            out.append(gc if gc.startswith("GAL") else f"GAL{int(gc):03d}")
        return out
    
    def execute(self):
        super().execute()
        for gal_cut in self.gal_cuts:
            mask = self.in_mask.read(map_field_strs=gal_cut)[0]
            mask = downgrade_mask(mask,
                                  nside_out=self.nside_out,
                                  threshold=self.mask_threshold)
            header = [
                ("SRCFIELD", gal_cut, "Field in source galactic mask file"),
                ("MSKTHRSH", self.mask_threshold, "downgrade_mask threshold"),
            ]
            with self.name_tracker.set_context("gal_cut", gal_cut):
                self.out_mask.write(data=mask.astype(np.uint8), 
                                    extra_header=header,
                                    column_names=["MASK"])
