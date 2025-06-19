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


class ShowPredictionsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_predictions")

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

    def execute(self) -> None:
        mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        logger.debug(f"Running {self.__class__.__name__} execute()")

        show_max_l = self.display_lmax

        split = self.splits[0]

        try:
            report_data = self.in_report.read()
        except FileNotFoundError:
            logger.warning(f"Report file {self.in_report.path} not found. Skipping report data.")
            report_data = None

        for epoch in self.model_epochs:
            all_theory_ps = []
            all_pred_ps = []
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
                        ps_pred = self.in_ps_pred.read()
                        ps_thry = self.in_ps_thry.read(use_alt_path=False)
                    all_theory_ps.append(ps_thry[:show_max_l])
                    all_pred_ps.append(ps_pred)
                    if self.display_n_sims and sim >= self.display_n_sims:
                        break

                full_ells = np.arange(show_max_l)
                part_ells = self.ells

                logger.info(f"There are {len(part_ells)} ells chosen.")

                fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
                ax2 = ax1

                # Confirming results in the other executor
                # ps_thry = np.array(all_theory_ps)
                # ps_thry = ps_thry[:, part_ells]
                # ps_pred = np.array(all_pred_ps)
                # delta = ps_thry - ps_pred
                # mse = np.mean(delta**2)
                # logger.info(f"Epoch {epoch}, MSE: {mse:.4f}")

                for i, ps_thry in enumerate(all_theory_ps):
                    # ax1.plot(full_ells, ps_thry, alpha=0.3)
                    ax1.plot(full_ells, ps_thry, 
                             alpha=1, 
                             label=f'Theory {i}', 
                             color=mpl_colors[i % len(mpl_colors)])
                for i, ps_pred in enumerate(all_pred_ps):
                    # ax2.scatter(part_ells, ps_pred, alpha=0.7, zorder=10)
                    ax1.plot(part_ells, ps_pred, 
                             alpha=1, ls="--", linewidth=2, 
                             label=f'Prediction {i}', 
                             color=mpl_colors[i % len(mpl_colors)])



                ax1.set_xlim([0, show_max_l])
                # ax1.set_ylim([0, 1.5])
                ax1.set_xlabel(r"$\ell$")
                ax1.set_ylabel(r"$D_\ell$")

                ax2.set_xlim([0, show_max_l])
                # ax2.set_ylim([0, 1.5])

                ax1.set_title(f"Predictions and theory, Epoch {epoch}")


                if report_data is not None:
                    key = ('mse', 'mean')
                    mask = (
                        (report_data[('split', '')] == split.name) &
                        (report_data[('epoch', '')] == epoch)
                    )
                    subset = report_data.loc[mask]
                    if subset.empty:
                        raise ValueError(f"No matching rows for split={split.name}, epoch={epoch}")
                    else:
                        mse = subset[key].values[0]
                    ax1.text(0.95, 0.95, f"MSE: {mse:.1e}")

                ax1.legend()

                fn = self.out_fig.path
                logger.info(f"Saving figure to {fn}")
                self.out_fig.write()  # Make directory only
                # fn = self.out_fig.path.name  # This may be more convenient
                plt.savefig(fn)
                plt.close(fig)
