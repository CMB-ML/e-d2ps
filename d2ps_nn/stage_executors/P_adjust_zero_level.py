import logging

import numpy as np
from omegaconf import DictConfig
import pysm3.units as u

from cmbml.core import Asset, BaseStageExecutor
from cmbml.core.asset_handlers import Config, HealpyMap
from cmbml.utils.fits_inspection import get_header_cards
from cmbml.utils.planck_instrument import make_instrument, Instrument
from d2ps_nn.utils.masks.mask_composition import gal_mask_label, provenance_cards
from d2ps_nn.utils.masks.masked_monopole import adjust_zero_level
from d2ps_nn.utils.masks.io import read_channel_mask


logger = logging.getLogger(__name__)


class AdjustZeroLevelExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str="adjust_zero_level") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_obs_maps: Asset = self.assets_out["obs_maps"]
        out_obs_maps_handler: HealpyMap

        self.in_obs_maps: Asset = self.assets_in["obs_maps"]
        self.in_mask: Asset = self.assets_in["mask"]
        self.in_monopole: Asset = self.assets_in["monopole"]
        in_obs_maps_handler: HealpyMap
        in_mask_handler: HealpyMap
        in_monopole_handler: Config

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.gal_source = cfg.masks.gal_source
        self.gal_cut_cfg = cfg.masks.gal_cut

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        with self.name_tracker.set_context("gal_mask_source", self.gal_source):
            monopole = self.in_monopole.read()
            mono_path = str(self.in_monopole.path)

        for ch in self.instrument.dets:
            stats = monopole["channels"][ch]

            mask, info = read_channel_mask(self.in_mask, self.name_tracker, ch,
                                        self.gal_source, self.gal_cut_cfg)
            if info != stats["mask"]:
                raise ValueError(
                    f"{ch} GHz: mask differs from when the monopole target was computed.\n"
                    f"  now:      {info}\n  recorded: {stats['mask']}\n"
                    "Rerun get_monopole."
                )

            context = dict(split="Test", sim_num=0, freq=ch)

            with self.name_tracker.set_contexts(context):
                m = self.in_obs_maps.read()[0]

            adjusted, offset, measured = adjust_zero_level(m, mask, stats["mean"])
            sigma = u.Quantity(stats["std"]).to(m.unit)
            logger.info(
                f"{context}: measured {measured:.4g}, target {stats['mean']:.4g}, "
                f"offset {offset:.4g} ({(offset / sigma).value:.1f} sigma)"
            )

            unit_str = m.unit.to_string()
            header = [
                ("ZL_MEAS", float(measured.value), f"Masked mean before adj [{unit_str}]"),
                ("ZL_TRGT", float(u.Quantity(stats["mean"]).to_value(m.unit)), f"Target masked mean [{unit_str}]"),
                ("ZL_OFF", float(offset.value), f"Subtracted offset [{unit_str}]"),
                ("ZL_MASK", info["path"], "Mask used for zero level"),
                ("ZL_FILE", mono_path, "Monopole target file"),
            ]
            with self.name_tracker.set_contexts(context):
                self.out_obs_maps.write(data=adjusted, extra_header=header)

    def load_mask(self, ch):
        gal_mask = gal_mask_label(ch, self.gal_source, self.gal_cut_cfg).lower()
        with self.name_tracker.set_contexts(dict(freq=ch, gal_mask=gal_mask)):
            mask = self.in_mask.read()[0].astype(bool)
            path = str(self.in_mask.path)
        return mask, path

    def check_mask(self, ch, mask, mask_path, stats):
        recorded = stats["mask"]
        if mask_path != recorded["path"]:
            raise ValueError(
                f"{ch} GHz: mask {mask_path} does not match the mask used for the "
                f"monopole target ({recorded['path']}). Check cfg.masks."
            )
        current = {k: v for k, v, _ in provenance_cards(get_header_cards(mask_path))}
        if current != recorded["header"]:
            raise ValueError(
                f"{ch} GHz: mask header differs from when the monopole target was computed.\n"
                f"  now:      {current}\n  recorded: {recorded['header']}\n"
                "Rerun get_monopole."
            )
        if not np.isclose(mask.mean(), recorded["f_sky"], rtol=0, atol=1e-12):
            raise ValueError(
                f"{ch} GHz: mask f_sky {mask.mean()} differs from recorded {recorded['f_sky']}. "
                "Rerun get_monopole."
            )
