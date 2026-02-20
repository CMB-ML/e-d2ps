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


class ShowPetroffSingle(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="fig_petroff_single")

        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: Figure

        self.in_thry: Asset = self.assets_in["ps_thry"]
        self.in_pred: Asset = self.assets_in["ps_pred"]
        self.in_real: Asset = self.assets_in["ps_real"]
        in_thry: CambPowerSpectrum
        in_pred: TextPowerSpectrum
        in_real: NumpyPowerSpectrum
        self.ells = parse_ells(cfg.model.use_ells)

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
        self.fig_label = cfg.fig_model_name
        self._model_epochs = cfg.use_epochs
        self.display_n_sims = cfg.n_show_cap
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
                self.process_epoch(split)

    def process_epoch(self, split: Split) -> None:
        for sim in split.iter_sims():
            if self.display_n_sims and sim > self.display_n_sims:
                break
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(split)

    def process_sim(self, split: Split):
        thry = self.in_thry.read(use_alt_path=split.ps_fidu_fixed)
        pred = self.in_pred.read()
        real = self.in_real.read()

        # Need to convert from implicit ells in the general case to this model's case
        thry = thry[self.ells]
        real = real[self.ells]

        make_fig_single(ells=self.ells,
                 thry=thry,
                 pred=pred,
                 real=real,
                 units=self.units,
                 fig_label=self.fig_label,
                 fig_handler=self.out_fig)

def make_fig_single(ells, thry, pred, real, units, fig_label, fig_handler):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    real_color = "gray"
    pred_color = "blue"
    thry_color = "green"

    ax1.plot(ells, real, color=real_color, lw=1, label="Reference Map")
    ax1.plot(ells, pred, color=pred_color, lw=1, label="Reconstructed Map")
    ax1.plot(ells, thry, color=thry_color, lw=1, linestyle="--", label="Reference Theory")

    ax1.set_ylabel(r"$D_\ell^{TT}$ [" + units.to_string("latex_inline") + "]")
    ax1.set_ylim(0, 7500)
    ax1.legend(loc="best")

    resid_real = real - thry
    resid_pred = pred - thry

    ax2.plot(ells, resid_real, color=real_color)
    ax2.plot(ells, resid_pred, color=pred_color)
    ax2.plot(ells, np.zeros_like(ells), linestyle="--", color=thry_color)

    ax2.set_xlabel(r"Multipole moment ($\ell$)")
    ax2.set_ylabel(r"$\Delta D_\ell^{TT} \; [\mu \mathrm{K}^2]$")
    ax2.set_ylim(-1000, 1000)

    # Title
    fig.suptitle(f"Predicted Spectrum and Cosmic Variance Fractions, {fig_label}", fontsize=14)

    plt.tight_layout()
    fig_handler.write(fig=fig)


class ShowPetroffSet(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="fig_petroff_set")

        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: Figure

        self.in_pred: Asset = self.assets_in["pred_vecs"]
        self.in_real: Asset = self.assets_in["real_vecs"]
        in_report: TextPowerSpectrum
        self.ells = parse_ells(cfg.model.use_ells)

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
        self.fig_label = cfg.fig_model_name
        self._model_epochs = cfg.use_epochs
        self.display_n_sims = cfg.n_show_cap

    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(split)

    def process_epoch(self, split: Split) -> None:
        with self.name_tracker.set_context("metric", "thry_mean"):
            thry_mean = self.in_pred.read()
        with self.name_tracker.set_context("metric", "pred_mean"):
            pred_mean = self.in_pred.read()
        with self.name_tracker.set_context("metric", "pred_sd"):
            pred_sd = self.in_pred.read()
        # It is deliberate that the context is set to "pred_mean" when
        #    in the next lines. These files were produced *as though*
        #    it were the output of a predictive model.
        with self.name_tracker.set_context("metric", "pred_mean"):
            real_mean = self.in_real.read()
        with self.name_tracker.set_context("metric", "pred_sd"):
            real_sd = self.in_real.read()
        
        make_fig_set(ells=self.ells,
                     thry_mean=thry_mean,
                     pred_mean=pred_mean,
                     pred_sd=pred_sd,
                     real_mean=real_mean,
                     real_sd=real_sd,
                     units=self.units,
                     fig_label=self.fig_label,
                     fig_handler=self.out_fig)

def make_fig_set(ells, 
                 thry_mean,
                 pred_mean,
                 pred_sd,
                 real_mean,
                 real_sd,
                 units, fig_label, fig_handler):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    real_color = "gray"
    pred_color = "blue"
    thry_color = "green"

    # Top plot: prediction mean with +/-1 s.d. band
    real_mnus_sd = real_mean - real_sd
    real_plus_sd = real_mean + real_sd
    pred_mnus_sd = pred_mean - pred_sd
    pred_plus_sd = pred_mean + pred_sd

    ax1.fill_between(ells, real_mnus_sd, real_plus_sd, color=real_color, alpha=0.3, edgecolor=None, label=r"Ref $\pm1\sigma$")
    ax1.fill_between(ells, pred_mnus_sd, pred_plus_sd, color=pred_color, alpha=0.3, edgecolor=None, label=r"Rec $\pm1\sigma$")
    ax1.plot(ells, real_mean, color=real_color, lw=1, label="Reference Map")
    ax1.plot(ells, pred_mean, color=pred_color, lw=1, label="Reconstructed Map")

    ax1.plot(ells, thry_mean, color=thry_color, lw=1, linestyle="--", label="Reference Theory")

    ax1.set_ylabel(r"$D_\ell^{TT}$ [" + units.to_string("latex_inline") + "]")
    ax1.set_ylim(0, 7500)
    ax1.legend(loc="best")

    resid_real = real_mean - thry_mean
    resid_pred = pred_mean - thry_mean
    resid_real_mnus_sd = resid_real - real_sd
    resid_real_plus_sd = resid_real + real_sd
    resid_pred_mnus_sd = resid_pred - pred_sd
    resid_pred_plus_sd = resid_pred + pred_sd

    # Bottom plot: Cosmic Variance Fractions
    # rel_error = cosmic_err
    ax2.fill_between(ells, resid_real_mnus_sd, resid_real_plus_sd, color=real_color, alpha=0.3, edgecolor=None)
    ax2.fill_between(ells, resid_pred_mnus_sd, resid_pred_plus_sd, color=pred_color, alpha=0.3, edgecolor=None)
    ax2.plot(ells, resid_real, color=real_color)
    ax2.plot(ells, resid_pred, color=pred_color)
    ax2.plot(ells, np.zeros_like(ells), linestyle="--", color=thry_color)

    ax2.set_xlabel(r"Multipole moment ($\ell$)")
    ax2.set_ylabel(r"$\Delta D_\ell^{TT} \; [\mu \mathrm{K}^2]$")
    ax2.set_ylim(-500, 500)

    # Title
    fig.suptitle(f"Predicted Spectrum and Cosmic Variance Fractions, {fig_label}", fontsize=14)

    plt.tight_layout()
    fig_handler.write(fig=fig)


class ShowPetroffSetBinned(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="fig_petroff_set_bin")

        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: Figure

        self.in_pred: Asset = self.assets_in["pred_vecs"]
        self.in_real: Asset = self.assets_in["real_vecs"]
        in_report: TextPowerSpectrum
        self.ells = parse_ells(cfg.model.use_ells)

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
        self.fig_label = cfg.fig_model_name
        self._model_epochs = cfg.use_epochs
        self.display_n_sims = cfg.n_show_cap

    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(split)

    def process_epoch(self, split: Split) -> None:
        with self.name_tracker.set_context("metric", "thry_mean"):
            thry_mean = self.in_pred.read()
        with self.name_tracker.set_context("metric", "pred_mean"):
            pred_mean = self.in_pred.read()
        with self.name_tracker.set_context("metric", "pred_sd"):
            pred_sd = self.in_pred.read()
        # Binned stuff
        with self.name_tracker.set_context("metric", "bin_centers"):
            bin_centers = self.in_pred.read()
        with self.name_tracker.set_context("metric", "bin_means"):
            pred_bin_mean = self.in_pred.read()
        with self.name_tracker.set_context("metric", "bin_sds"):
            pred_bin_sd = self.in_pred.read()
        with self.name_tracker.set_context("metric", "bin_resid_means"):
            res_pred_bin_mean = self.in_pred.read()
        with self.name_tracker.set_context("metric", "bin_resid_sds"):
            res_pred_bin_sd = self.in_pred.read()

        # It is deliberate that the context is set to "pred_mean" when
        #    in the next lines. These files were produced *as though*
        #    it were the output of a predictive model.
        with self.name_tracker.set_context("metric", "pred_mean"):
            real_mean = self.in_real.read()
        with self.name_tracker.set_context("metric", "pred_sd"):
            real_sd = self.in_real.read()
        with self.name_tracker.set_context("metric", "bin_means"):
            real_bin_mean = self.in_real.read()
        with self.name_tracker.set_context("metric", "bin_sds"):
            real_bin_sd = self.in_real.read()
        with self.name_tracker.set_context("metric", "bin_resid_means"):
            res_real_bin_mean = self.in_real.read()
        with self.name_tracker.set_context("metric", "bin_resid_sds"):
            res_real_bin_sd = self.in_real.read()

        make_fig_bin(ells=self.ells,
                     thry_mean=thry_mean,
                     pred_mean=pred_mean,
                     pred_sd=pred_sd,
                     real_mean=real_mean,
                     real_sd=real_sd,
                     bin_ells=bin_centers,
                     pred_bin_mean=pred_bin_mean,
                     pred_bin_sd=pred_bin_sd,
                     real_bin_mean=real_bin_mean,
                     real_bin_sd=real_bin_sd,
                     res_pred_bin_mean=res_pred_bin_mean,
                     res_pred_bin_sd  =res_pred_bin_sd,
                     res_real_bin_mean=res_real_bin_mean,
                     res_real_bin_sd  =res_real_bin_sd,
                     units=self.units,
                     fig_label=self.fig_label,
                     fig_handler=self.out_fig)

def make_fig_bin(ells, 
                 thry_mean,
                 pred_mean,
                 pred_sd,
                 real_mean,
                 real_sd,
                 bin_ells,
                 pred_bin_mean,
                 pred_bin_sd,
                 real_bin_mean,
                 real_bin_sd,
                 res_pred_bin_mean,
                 res_pred_bin_sd  ,
                 res_real_bin_mean,
                 res_real_bin_sd  ,
                 units, fig_label, fig_handler):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    real_color = "#6D6C6CFF"
    pred_color = "#0E23E4FF"
    thry_color = "#0C6E11FF"
    real_color_a = "#6D6C6C99"
    pred_color_a = "#0E23E499"

    # Top plot: prediction mean with +/-1 s.d. band
    real_mnus_sd = real_mean - real_sd
    real_plus_sd = real_mean + real_sd
    pred_mnus_sd = pred_mean - pred_sd
    pred_plus_sd = pred_mean + pred_sd

    ax1.errorbar(
        x=bin_ells,
        y=real_bin_mean,
        yerr=real_bin_sd,
        fmt='o',
        fillstyle='right',
        markerfacecolor=real_color_a,
        ecolor=real_color_a,
        markeredgecolor='#00000000',
        capsize=4,
        ms=5,
        label="Reference",
        linestyle=None
    )
    ax1.errorbar(
        x=bin_ells,
        y=pred_bin_mean,
        yerr=pred_bin_sd,
        fmt='o',
        fillstyle='left',
        markerfacecolor=pred_color_a,
        ecolor=pred_color_a,
        markeredgecolor='#00000000',
        capsize=4,
        ms=5,
        label="Reconstructed",
        linestyle=None,
    )

    ax1.fill_between(ells, real_mnus_sd, real_plus_sd, color=real_color, alpha=0.3, edgecolor=None, label=r"Ref $\pm1\sigma$")
    ax1.fill_between(ells, pred_mnus_sd, pred_plus_sd, color=pred_color, alpha=0.3, edgecolor=None, label=r"Rec $\pm1\sigma$")
    ax1.plot(ells, real_mean, color=real_color, lw=1, label="Reference Map")
    ax1.plot(ells, pred_mean, color=pred_color, lw=1, label="Reconstructed Map")

    ax1.plot(ells, thry_mean, color=thry_color, lw=1, linestyle="--", label="Reference Theory")

    ax1.set_ylabel(r"$D_\ell^{TT}$ [" + units.to_string("latex_inline") + "]")
    ax1.set_ylim(0, 7500)
    ax1.legend(loc="best")

    resid_real = real_mean - thry_mean
    resid_pred = pred_mean - thry_mean

    # resid_real_mnus_sd = resid_real - real_sd
    # resid_real_plus_sd = resid_real + real_sd
    # resid_pred_mnus_sd = resid_pred - pred_sd
    # resid_pred_plus_sd = resid_pred + pred_sd

    # Bottom plot: Cosmic Variance Fractions
    # rel_error = cosmic_err
    ax2.errorbar(
        x=bin_ells,
        y=res_real_bin_mean,
        yerr=res_real_bin_sd,
        fmt='o',
        fillstyle='right',
        markerfacecolor=real_color_a,
        ecolor=real_color_a,
        markeredgecolor='#00000000',
        capsize=4,
        ms=5,
        label="Reference",
        linestyle=None
    )
    ax2.errorbar(
        x=bin_ells,
        y=res_pred_bin_mean,
        yerr=res_pred_bin_sd,
        fmt='o',
        fillstyle='left',
        markerfacecolor=pred_color_a,
        ecolor=pred_color_a,
        markeredgecolor='#00000000',
        capsize=4,
        ms=5,
        label="Reconstructed",
        linestyle=None,
    )

    # ax2.fill_between(ells, resid_real_mnus_sd, resid_real_plus_sd, color=real_color, alpha=0.3, edgecolor=None)
    # ax2.fill_between(ells, resid_pred_mnus_sd, resid_pred_plus_sd, color=pred_color, alpha=0.3, edgecolor=None)
    # ax2.plot(ells, resid_real, color=real_color)
    # ax2.plot(ells, resid_pred, color=pred_color)
    ax2.plot(ells, np.zeros_like(ells), linestyle="--", color=thry_color)

    ax2.set_xlabel(r"Multipole moment ($\ell$)")
    ax2.set_ylabel(r"$\Delta D_\ell^{TT} \; [\mu \mathrm{K}^2]$")
    ax2.set_ylim(-150, 150)

    # Title
    fig.suptitle(f"Predicted Spectrum and Cosmic Variance Fractions, {fig_label}", fontsize=14)

    plt.tight_layout()
    fig_handler.write(fig=fig)
