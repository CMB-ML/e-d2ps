import logging

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class ShowChosenEllsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_chosen_ells")

        self.out_fig: Asset = self.assets_out["fig"]
        out_lut_handler: Figure

        self.in_ps_prep: Asset = self.assets_in["ps_prep"]
        self.in_ps_thry: Asset = self.assets_in["ps_thry"]
        in_ps_prep_handler: TextPowerSpectrum
        in_ps_thry_handler: CambPowerSpectrum

        self.ells = parse_ells(cfg.model.use_ells)
        # self.scale_ps_choice = cfg.model.scale_ps

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        split = self.splits[0]
        sim = split.iter_sims().__next__()

        context = dict(
            split=split.name,
            sim_num=sim,
        )
        with self.name_tracker.set_contexts(context):
            ps_prep = self.in_ps_prep.read()
            ps_thry = self.in_ps_thry.read(use_alt_path=False)

        full_ells = np.arange(len(ps_thry))
        part_ells = self.ells

        fig, ax = plt.subplots()
        ax.plot(full_ells, ps_thry, label="Full spectrum")
        ax.scatter(part_ells, ps_prep, label="Chosen ells")
        ax.set_xlim([0, 1000])
        fn = self.out_fig.path
        logger.info(f"Saving figure to {fn}")
        self.out_fig.write()  # Make directory only
        # fn = self.out_fig.path.name  # Save in working directory, maybe more convenient
        plt.savefig(fn)
        plt.close()
