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
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from cmbml.utils.cosmic_variance import get_var_per_ell


logger = logging.getLogger(__name__)


class ShowChiSquare(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="fig_chi_square")

        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: Figure

        self.in_report: Asset = self.assets_in["per_vector_report"]
        in_pv_report: TextPowerSpectrum
        self.ells = parse_ells(cfg.model.use_ells)

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
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
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.out_fig.handler.set_fig_types(self.fig_type)

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch()

    def process_epoch(self) -> None:
        trace_labels = [
            "pred_mean",
            "pred_sd",
            "thry_mean",
            "ensemble_chi_square"
        ]
        traces = t = {}
        for t_l in trace_labels:
            with self.name_tracker.set_context("metric", t_l):
                t[t_l] = self.in_report.read()
        
        make_fig(ells=self.ells,
                 pred_mean=t["pred_mean"],
                 pred_sd=t["pred_sd"],
                 ensemble_chi_sq=t["ensemble_chi_square"],
                 thry=t["thry_mean"],
                 units=self.units,
                 fig_label=self.fig_label,
                 fig_handler=self.out_fig)

def make_fig(ells, pred_mean, pred_sd, thry, ensemble_chi_sq, units, fig_label, fig_handler):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top plot: prediction mean with +/-1 s.d. band
    ax1.plot(ells, pred_mean, label="Prediction (Mean)")
    minus_std = pred_mean - pred_sd
    plus_std  = pred_mean + pred_sd
    ax1.fill_between(ells, minus_std, plus_std, alpha=0.3, label=r"$\pm1\sigma$")
    ax1.plot(ells, thry, color='black', lw=1, label="Theory")

    ax1.set_ylabel(r"$D_\ell$ [" + units.to_string("latex_inline") + "]")
    ax1.legend(loc="best")

    # Bottom plot: Chi Square
    rel_error = ensemble_chi_sq
    ax2.plot(ells, rel_error, color="tab:red", label="Chi Square")

    ax2.set_xlabel(r"Multipole $\ell$")
    ax2.set_ylabel("Chi Square")

    # Title
    fig.suptitle(f"Predicted Spectrum and Chi Square, {fig_label}", fontsize=14)

    plt.tight_layout()
    fig_handler.write(fig=fig)
