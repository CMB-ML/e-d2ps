import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from omegaconf import DictConfig

import pysm3.units as u

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum, PandasCsvHandler
from cmbml.utils.number_report import float_to_latex
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


def _seq_colors(cmap_name, n):
    cmap = cm.get_cmap(cmap_name)
    return [cmap(x) for x in np.linspace(0.65, 0.90, max(n, 1))][:n]


class ShowBWPredictionsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Creates the first display_n_sims + top_n + bottom_n predictions 
        as individual plots.
        top_n and bottom_n are defined in the config file
        The top_n predictions are chosen based on lowest MSE (bottom by highest)
        """
        super().__init__(cfg, stage_str="show_predictions_bw")

        self.out_fig: Asset = self.assets_out["fig"]
        out_lut_handler: Figure

        self.in_ps_pred: Asset = self.assets_in["ps_pred"]
        self.in_ps_thry: Asset = self.assets_in["ps_thry"]
        self.in_report: Asset = self.assets_in["report"]
        in_ps_prep_handler: TextPowerSpectrum
        in_ps_thry_handler: CambPowerSpectrum
        in_report_handler: PandasCsvHandler

        self.fig_type = cfg.fig_type

        self.units = u.Unit(cfg.scenario.units)

        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax
        self.top_n = cfg.model.analysis.top_n
        self.bottom_n = cfg.model.analysis.bottom_n
        self.fig_label = cfg.fig_model_name

        self.best_colors = _seq_colors("Blues",  self.top_n)
        self.worst_colors = _seq_colors("Reds",  self.bottom_n)[::-1]

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.out_fig.handler.set_fig_types(self.fig_type)

        if self.top_n <= 0 and self.bottom_n <=0:
            logger.info("Settings for top_n and bottom_n result in no figure.")
            return

        for split in self.splits:
            # if "F" not in split.name:
            #     continue
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split):
        show_max_l = self.display_lmax
        use_alt_path = True if split.ps_fidu_fixed else False

        report_data = self.in_report.read()
        report_data['epoch'] = report_data['epoch'].astype(str)  # Ensure epoch is string for comparison

        full_ells = np.arange(show_max_l)
        part_ells = self.ells

        def show_instance(ax, thry, pred, color, idx):
            ax.plot(full_ells[2:], thry[2:], 
                        alpha=0.7, linewidth=0.5,
                        color=color)
            if len(part_ells) < 20:
                ax.scatter(part_ells, pred, 
                            alpha=0.7, zorder=10,
                            color=color)
            else:
                ax.plot(part_ells, pred, 
                        alpha=0.7, ls=(idx, (5,5)), linewidth=2, 
                        color=color)

        for epoch in self.model_epochs:
            # Load all spectra (they're small, it's fast)
            all_thry_ps, all_pred_ps = {}, {}
            context = dict(split=split.name, epoch=epoch)
            with self.name_tracker.set_contexts(context):
                for sim in split.iter_sims():
                    with self.name_tracker.set_context("sim_num", sim):
                        ps_pred = self.in_ps_pred.read()
                        ps_thry = self.in_ps_thry.read(use_alt_path=use_alt_path)
                    all_thry_ps[sim] = ps_thry[:show_max_l]
                    # TODO: Fix this. Patch exists to handle Realization Power Spectra,
                    #       which are produced using a different way of defining ells
                    #       (thye use lmax; this later method uses use_ells)
                    if ps_pred.shape[0] == 1281:
                        ps_pred = ps_pred[2:]
                    all_pred_ps[sim] = ps_pred

            # Use the report to find best and worst instances
            key = 'mse'
            if epoch == "":
                lookup_epoch = "nan"
            else:
                lookup_epoch = str(epoch)
            mask = (
                (report_data['split'] == split.name) &
                (report_data['epoch'] == lookup_epoch)
            )
            subset = report_data.loc[mask]
            if subset.empty:
                raise ValueError(f"No matching rows for split={split.name}, epoch={epoch}")

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            worst_order = subset.nlargest(self.bottom_n, key)['sim'].tolist() if self.bottom_n > 0 else []
            for i, this_sim in enumerate(worst_order):
                this_color = self.worst_colors[i]
                this_thry_ps = all_thry_ps[this_sim]
                this_pred_ps = all_pred_ps[this_sim]
                show_instance(ax, 
                              this_thry_ps, this_pred_ps, 
                              this_color, i)

            best_order = subset.nsmallest(self.top_n, key)['sim'].tolist() if self.top_n > 0 else []
            for j, this_sim in enumerate(best_order):
                this_color = self.best_colors[j]
                this_thry_ps = all_thry_ps[this_sim]
                this_pred_ps = all_pred_ps[this_sim]
                show_instance(ax, 
                              this_thry_ps, this_pred_ps, 
                              this_color, i+j)

            ax.set_xlim([0, show_max_l])
            ax.set_ylim([0, 6700])
            ax.set_xlabel(r"$\ell$")
            unit_str = (self.units**2).to_string('latex_inline')
            ax.set_ylabel(r"$D_\ell$ [" + unit_str + "]")
            ax.set_title(f"Best and worst instances in {split.name}, {self.fig_label}")

            target_proxy = Line2D([0],[0], color="black", lw=0.8, label="Target")
            pred_proxy   = Line2D([0],[0], color="black", lw=2, ls=(0,(3,3)), label="Prediction")

            def sim_mse(s):
                return subset.loc[subset['sim'] == s, 'mse'].iloc[0]

            sim_proxies = []
            for s, c in zip(worst_order, self.worst_colors):
                mse_str = float_to_latex(sim_mse(s))
                sim_proxies.append(Line2D([0],[0], color=c, lw=2, label=f"Sim {s:04d}, MSE: ${mse_str}$"))
            for s, c in zip(best_order, self.best_colors):
                if s in worst_order:  # avoid dup if overlap
                    continue
                mse_str = float_to_latex(sim_mse(s))
                sim_proxies.append(Line2D([0],[0], color=c, lw=2, label=f"Sim {s:04d}, MSE: ${mse_str}$"))

            ax.legend(handles=[target_proxy, pred_proxy, *sim_proxies], 
                      loc="upper right")

            with self.name_tracker.set_contexts(context):
                self.out_fig.write(fig=fig)
