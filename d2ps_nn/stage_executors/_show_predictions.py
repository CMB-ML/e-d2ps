import logging

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum, PandasCsvHandler
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.get_epochs import get_epochs_with_best


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
        self.fig_label = cfg.fig_model_name
        self._model_epochs = cfg.use_epochs

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

                # logger.info(f"There are {len(part_ells)} ells chosen.")

                fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
                ax2 = ax1

                # Confirming results in the other executor
                # ps_thry = np.array(all_theory_ps)
                # ps_thry = ps_thry[:, part_ells]
                # ps_pred = np.array(all_pred_ps)
                # delta = ps_thry - ps_pred
                # mse = np.mean(delta**2)
                # logger.info(f"Epoch {epoch}, MSE: {mse:.4f}")

                for ps_thry in all_theory_ps:
                    ax1.plot(full_ells, ps_thry, alpha=0.3)
                for ps_pred in all_pred_ps:
                    ax2.scatter(part_ells, ps_pred, alpha=0.7, zorder=10)

                ax1.set_xlim([0, show_max_l])
                # ax1.set_ylim([0, 1.5])
                ax1.set_xlabel(r"$\ell$")
                ax1.set_ylabel(r"$D_\ell$")

                ax2.set_xlim([0, show_max_l])
                # ax2.set_ylim([0, 1.5])

                ax1.set_title(f"Predictions and theory, Epoch {epoch}, {self.fig_label}")

                # Unneeded currently; could be useful when re-adding sequential best epochs
                # if isinstance(epoch, str) and epoch.startswith("best_"):
                #     epoch = int(epoch.split('_')[1]) 

                if report_data is not None:
                    key = ('mse', 'mean')
                    mask = (
                        (report_data[('split', '')] == split.name) &
                        (report_data[('epoch', '')] == str(epoch))
                    )
                    subset = report_data.loc[mask]
                    if subset.empty:
                        raise ValueError(f"No matching rows for split={split.name}, epoch={epoch}")
                    else:
                        mse = subset[key].values[0]
                    ax1.text(0.95, 0.95, f"MSE: {mse:.1e}")

                fn = self.out_fig.path
                # fn = self.out_fig.path.name  # This may be more convenient
                logger.info(f"Saving figure to {fn}")
                self.out_fig.write()  # Make directory only
                plt.savefig(fn)
                plt.close(fig)
