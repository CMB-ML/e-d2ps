from typing import Dict
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class GatherSpectraExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str="ps_bundle") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_bundle: Asset = self.assets_out["ps_bundle"]
        
        self.in_ps: Dict[str, Asset] = dict(
            thry=self.assets_in["thry_ps"],
            real=self.assets_in["real_ps"],
            params=None
        )

        self.force_new = cfg.force_new

        self.in_src = cfg.src.input
        self.out_src = cfg.src.output

        self.in_lmax = cfg.get("lmax_in", None)
        self.out_lmax = cfg.lmax_out

        self.use_ells_str = cfg.model.use_ells

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        # Input
        asset = self.in_ps[self.in_src]
        if asset is not None:
            ells = parse_ells(self.use_ells_str, self.in_lmax)
            for split in self.splits:
                context = dict(split=split.name, ellmax=self.in_lmax, src=self.in_src)
                with self.name_tracker.set_contexts(context):
                    self.process(split, asset, ells)

        # Output
        asset = self.in_ps[self.out_src]
        ells = parse_ells(self.use_ells_str, self.in_lmax)
        for split in self.splits:
            context = dict(split=split.name, ellmax=self.out_lmax, src=self.out_src)
            with self.name_tracker.set_contexts(context):
                self.process(split, asset, ells)

    def process(self, split: Split, asset: Asset, ells) -> None:
        if not self.force_new:
            if self.ok_to_skip(self.out_bundle):
                return
        
        # Prepared to handle single spectra for a split
        if split.ps_fidu_fixed:
            try:
                data = asset.read(use_alt_path=split.ps_fidu_fixed)
                data = data[ells]
                self.out_bundle.write(data=data)
                return 
            except AttributeError:
                pass
            except KeyError:        # If it's looking for sim_num
                pass

        # Handle (general case) different spectrum per sim
        data = []
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                try:
                    spectrum = asset.read()
                except AttributeError:
                    spectrum = asset.read(use_alt_path=False)
                data.append(spectrum)
        data = np.array(data)
        data = data[:, ells]
        self.out_bundle.write(data=data)

    def ok_to_skip(self, asset: Asset):
        # Setting sim_num = 0 is ok even for fixed ps
        with self.name_tracker.set_context("sim_num", 0):
            found = asset.path.exists()
            if found:
                return found
            if not found:
                try:
                    found = asset.alt_path.exists()
                    return found
                except:
                    return False
