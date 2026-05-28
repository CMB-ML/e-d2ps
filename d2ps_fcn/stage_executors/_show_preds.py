import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from omegaconf import DictConfig
import pysm3.units as u

from cmbml.core import Split, Asset, AssetWithPathAlts
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum, PandasCsvHandler
from cmbml.utils.number_report import float_to_latex
# from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.select_ps import (
    # SpectrumAsset, 
    # SpectrumAssetNames,
    norm_ps_type
)


logger = logging.getLogger(__name__)


class OTFShowPredsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_preds_otf")

        self.out_fig: Asset = self.assets_out["fig"]
        out_lut_handler: Figure

        self.in_ps_pred: Asset = self.assets_in["ps_pred"]
        self.in_ps_thry: Asset = self.assets_in["ps_thry"]
        self.in_ps_real: Asset = self.assets_in["ps_real"]
        in_ps_prep_handler: TextPowerSpectrum
        in_ps_thry_handler: CambPowerSpectrum
        in_report_handler: PandasCsvHandler

        self.spectrum_type = norm_ps_type(cfg.spectrum_type)  # "theory" | "realization"

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax
        self.display_n_sims = cfg.n_show_cap
        self.fig_label = cfg.fig_model_name
        self._model_epochs = cfg.use_epochs
        self.fig_type = cfg.fig_type

    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self) -> None:
        self._model_epochs = get_epochs_with_best(self._model_epochs, self.in_ps_pred, self.name_tracker)
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.out_fig.handler.set_fig_types(self.fig_type)

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        show_max_l = self.display_lmax

        for epoch in self.model_epochs:
            with self.name_tracker.set_context('epoch', epoch):
                context = dict(
                    split=split.name,
                )
                with self.name_tracker.set_contexts(context):
                    all_thry_ps = self.in_ps_thry.read()[:self.display_n_sims, :]
                    all_real_ps = self.in_ps_real.read()[:self.display_n_sims, :]
                    all_pred_ps = self.in_ps_pred.read()[:self.display_n_sims, :]

                ells = self.ells

                fig, ax = plt.subplots(1, 1, figsize=(12, 6))

                max_scatter_multipoles = 20

                for i, ps_thry in enumerate(all_thry_ps):
                    if self.spectrum_type == "theory":
                        ax.plot(ells, ps_thry, 
                                alpha=1, linewidth=0.5,
                                label=f'Theory {i}', 
                                color=mpl_colors[i % len(mpl_colors)])
                        ax.plot(ells, all_real_ps[i], 
                                alpha=0.2, linewidth=0.5,
                                label=f'Realization {i}', 
                                color=mpl_colors[i % len(mpl_colors)])
                    else:  # Realization
                        ax.plot(ells, all_real_ps[i], 
                                alpha=1, linewidth=0.5,
                                label=f'Realization {i}', 
                                color=mpl_colors[i % len(mpl_colors)])
                        ax.plot(ells, ps_thry, 
                                alpha=0.5, linewidth=0.5,
                                label=f'Theory {i}', 
                                color=mpl_colors[i % len(mpl_colors)])
                for i, ps_pred in enumerate(all_pred_ps):
                    if len(ells) < max_scatter_multipoles:
                        ax.scatter(ells, ps_pred, 
                                    alpha=0.7, zorder=10,
                                    label=f'Prediction {i}',
                                    color=mpl_colors[i % len(mpl_colors)])
                    else:
                        ax.plot(ells, ps_pred, 
                                alpha=0.7, ls=(i, (5,5)), linewidth=2, 
                                label=f'Prediction {i}', 
                                color=mpl_colors[i % len(mpl_colors)])

                # Proxies for the legend
                theory_proxy = Line2D([0], [0], color="black", linewidth=0.5, alpha=1)
                # Pred proxy depends on how many multipoles are included
                if len(ells) < max_scatter_multipoles:
                    pred_proxy = Line2D([0], [0], color="black", marker="o",
                                        linestyle="none", alpha=0.7)
                else:
                    pred_proxy = Line2D([0], [0], color="black", linewidth=2,
                                        ls=(0, (3, 3)), alpha=0.7)

                # Put the legend wherever you like; fig.legend works across axes
                handles = [theory_proxy, pred_proxy]
                labels  = ["Target", "Prediction"]
                ax.legend(handles=handles, labels=labels, loc="upper right")

                ax.set_xlim([0, show_max_l])
                ax.set_ylim([0, 6700])
                ax.set_xlabel(r"$\ell$")
                unit_str = (self.units**2).to_string('latex_inline')
                ax.set_ylabel(r"$D_\ell$ [" + unit_str + "]")

                ax.set_xlim([0, show_max_l])
                # ax2.set_ylim([0, 1.5])

                ax.set_title(f"Predictions and Targets, {self.fig_label}")

                self.out_fig.write(fig=fig)  # Make directory only
