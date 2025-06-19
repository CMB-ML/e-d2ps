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


class ShowIndivPredictionsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Creates the first display_n_sims + top_n + bottom_n predictions 
        as individual plots.
        top_n and bottom_n are defined in the config file
        The top_n predictions are chosen based on lowest MSE (bottom by highest)
        """
        super().__init__(cfg, stage_str="show_predictions_indiv")

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
        self.display_n_sims = cfg.n_show_cap
        self.top_n = cfg.model.analysis.top_n
        self.bottom_n = cfg.model.analysis.bottom_n

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        show_max_l = self.display_lmax

        split = self.splits[0]
        report_data = self.in_report.read()
        report_data['epoch'] = report_data['epoch'].astype(str)  # Ensure epoch is string for comparison

        for epoch in self.model_epochs:
            show_sims = set(range(self.display_n_sims))
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
                    top_subset = subset.nsmallest(self.top_n, key)
                    show_sims.update(top_subset["sim"].tolist())
                elif self.bottom_n > 0:
                    bottom_subset = subset.nlargest(self.bottom_n, key)
                    show_sims.update(bottom_subset["sim"].tolist())

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

                for this_sim in show_sims:
                    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
                    ax2 = ax1

                    # for ps_thry in all_theory_ps:
                    #     ax1.plot(full_ells, ps_thry, alpha=0.3)
                    # for ps_pred in all_pred_ps:
                    #     ax2.scatter(part_ells, ps_pred, alpha=0.7, zorder=10)

                    this_theory_ps = all_theory_ps[this_sim]
                    this_pred_ps = all_pred_ps[this_sim]

                    ax1.plot(full_ells, this_theory_ps, alpha=1, label='Theory')
                    ax1.plot(part_ells, this_pred_ps, alpha=1, ls="--", linewidth=2, label='Prediction')
                    # ax2.scatter(part_ells, all_pred_ps[sim], alpha=0.5, zorder=10)

                    ax1.set_xlim([0, show_max_l])
                    ax1.set_ylim([0, 6000])
                    ax1.set_xlabel(r"$\ell$")
                    ax1.set_ylabel(r"$D_\ell$")

                    ax2.set_xlim([0, show_max_l])
                    # ax2.set_ylim([0, 1.5])

                    ax1.set_title(f"Predictions and theory, Epoch {epoch}, Sim {this_sim}")

                    if report_data is not None:
                        key = 'mse'
                        mask = (
                            (report_data['sim'] == this_sim) &
                            (report_data['split'] == split.name) &
                            (report_data['epoch'] == str(epoch))
                        )
                        subset = report_data.loc[mask]
                        if subset.empty:
                            raise ValueError(f"No matching rows for sim={this_sim}, split={split.name}, epoch={epoch}")
                        else:
                            mse = subset[key].values[0]
                        ax1.text(0.95, 0.95, f"MSE: {mse:.1e}")

                    with self.name_tracker.set_context("sim_num", this_sim):
                        fn = self.out_fig.path
                        logger.info(f"Saving figure to {fn}")
                        self.out_fig.write()  # Make directory only
                    # fn = self.out_fig.path.name  # This may be more convenient
                    plt.savefig(fn)
                    plt.close(fig)
