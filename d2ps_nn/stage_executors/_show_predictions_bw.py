import logging

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum, PandasCsvHandler
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


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

        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax
        self.first_n = cfg.model.analysis.first_n
        self.top_n = cfg.model.analysis.top_n
        self.bottom_n = cfg.model.analysis.bottom_n

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        show_max_l = self.display_lmax

        split = self.splits[0]
        report_data = self.in_report.read()
        report_data['epoch'] = report_data['epoch'].astype(str)  # Ensure epoch is string for comparison

        for epoch in self.model_epochs:
            show_sims = set()
            if self.top_n > 0 or self.bottom_n > 0:
                # Get the top_n sims based on MSE
                key = 'mse'
                mask = (
                    (report_data['split'] == split.name) &
                    (report_data['epoch'] == str(epoch))  # Use the last epoch
                )
                subset = report_data.loc[mask]
                if subset.empty:
                    raise ValueError(f"No matching rows for split={split.name}, epoch={epoch}")

                if self.top_n > 0:
                    top_subset = subset.nsmallest(self.top_n, key)["sim"].tolist()
                    show_sims.update(top_subset)
                if self.bottom_n > 0:
                    bottom_subset = subset.nlargest(self.bottom_n, key)["sim"].tolist()
                    show_sims.update(bottom_subset)

            show_sims = list(sorted(show_sims))

            all_theory_ps = {}
            all_pred_ps = {}
            # with self.name_tracker.set_context('epoch', epoch):
                # logger.info(f"Processing epoch {epoch} for split {split.name}")
                # self.process_epoch(split, epoch, show_max_l, scale_prep_ps, all_theory_ps, all_prep_ps)
            context = dict(
                split=split.name,
                epoch=epoch
            )
            with self.name_tracker.set_contexts(context):
                for sim in show_sims:
                    with self.name_tracker.set_context("sim_num", sim):
                        ps_pred = self.in_ps_pred.read()
                        ps_thry = self.in_ps_thry.read(use_alt_path=False)
                    all_theory_ps[sim] = ps_thry[:show_max_l]
                    all_pred_ps[sim] = ps_pred

                full_ells = np.arange(show_max_l)
                part_ells = self.ells

                fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
                ax2 = ax1
                first_best = True
                for i, this_sim in enumerate(show_sims):
                    this_theory_ps = all_theory_ps[this_sim]
                    this_pred_ps = all_pred_ps[this_sim]

                    is_best = this_sim in top_subset
                    if is_best:
                        color = 'black'
                        first_best = False
                        alpha_pred = 0.5
                        if first_best:
                            label_thry = f'Theory'
                            label_pred = f'Prediction'
                        else:
                            label_thry = None
                            label_pred = None
                        linestyle = (i, (5,5))
                    else:
                        color = mpl_colors[i % len(mpl_colors)]
                        label_thry = f'Theory {this_sim}'
                        label_pred = f'Prediction {this_sim}'
                        alpha_pred = 0.7
                        linestyle = (i, (5,5))

                    ax1.plot(full_ells, this_theory_ps, 
                             alpha=1, linewidth=0.5,
                             color=color,
                             label=label_pred,)
                    if len(part_ells) < 20:
                        ax2.scatter(part_ells, this_pred_ps, 
                                    alpha=alpha_pred, zorder=10,
                                    color=color,
                                    label=label_thry)
                    else:
                        ax1.plot(part_ells, this_pred_ps, 
                                alpha=alpha_pred, ls=linestyle, linewidth=2, 
                                color=color,
                                label=label_thry)

                ax1.set_xlim([0, show_max_l])
                ax1.set_ylim([0, 6000])
                ax1.set_xlabel(r"$\ell$")
                ax1.set_ylabel(r"$D_\ell$")
                ax1.set_title(f"{split.name}, Epoch {epoch}")

                # with self.name_tracker.set_context("sim_num", this_sim):
                fn = self.out_fig.path
                logger.info(f"Saving figure to {fn}")
                self.out_fig.write()  # Make directory only
                # fn = self.out_fig.path.name  # This may be more convenient
                plt.savefig(fn)
                plt.close(fig)
