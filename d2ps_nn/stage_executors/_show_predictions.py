import logging

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class ShowPredictionsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_predictions")

        self.out_fig: Asset = self.assets_out["fig"]
        out_lut_handler: Figure

        self.in_ps_pred: Asset = self.assets_in["ps_pred"]
        self.in_ps_thry: Asset = self.assets_in["ps_thry"]
        in_ps_prep_handler: TextPowerSpectrum
        in_ps_thry_handler: CambPowerSpectrum

        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        show_max_l = self.display_lmax

        split = self.splits[0]

        for epoch in self.model_epochs:
            all_theory_ps = []
            all_prep_ps = []
            # with self.name_tracker.set_context('epoch', epoch):
                # logger.info(f"Processing epoch {epoch} for split {split.name}")
                # self.process_epoch(split, epoch, show_max_l, scale_prep_ps, all_theory_ps, all_prep_ps)
            with self.name_tracker.set_context('epoch', epoch):
                for sim in split.iter_sims():
                    context = dict(
                        split=split.name,
                        sim_num=sim
                    )
                    with self.name_tracker.set_contexts(context):
                        ps_prep = self.in_ps_pred.read()
                        ps_thry = self.in_ps_thry.read(use_alt_path=False)
                    all_theory_ps.append(ps_thry[:show_max_l])
                    all_prep_ps.append(ps_prep)

                full_ells = np.arange(show_max_l)
                part_ells = self.ells

                logger.info(f"There are {len(part_ells)} ells chosen.")

                fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
                ax2 = ax1

                # It may be easier to see when it's two separate plots
                # You may have to reorder some of these lines :/
                # fig, axs = plt.subplots(1, 1, figsize=(12, 6))
                # ax1, ax2 = axs
                # ax1.set_title("Full spectrum")
                # ax2.set_title("Chosen ells")
                # ax2.set_xlabel(r"$\ell$")
                # ax2.set_ylabel(r"$D_\ell$")

                for ps_thry in all_theory_ps:
                    ax1.plot(full_ells, ps_thry, alpha=0.3)
                for ps_prep in all_prep_ps:
                    ax2.scatter(part_ells, ps_prep, alpha=0.7, zorder=10)

                ax1.set_xlim([0, show_max_l])
                # ax1.set_ylim([0, 1.5])
                ax1.set_xlabel(r"$\ell$")
                ax1.set_ylabel(r"$D_\ell$")

                ax2.set_xlim([0, show_max_l])
                # ax2.set_ylim([0, 1.5])

                ax1.set_title(f"Predictions and theory, Epoch {epoch}")

                fn = self.out_fig.path
                logger.info(f"Saving figure to {fn}")
                self.out_fig.write()  # Make directory only
                # fn = self.out_fig.path.name  # This may be more convenient
                plt.savefig(fn)
                plt.close(fig)
