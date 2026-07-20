import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from omegaconf import DictConfig
import pysm3.units as u
from cmbml.core import Split, Asset, AssetWithPathAlts
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import MPLFigure, NpyHandler, Config
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.get_epochs import get_epochs_with_best
# from d2ps_nn.utils.find_peaks import find_crits, PeakFitError
from d2ps_nn.utils.spectrum_metrics import (
    Prior, compute_metrics, decomposition,
    score, score_fraction, locked_count, cleared_count,
)


logger = logging.getLogger(__name__)


PLOT_SPEC = {
    "ref":     dict(ylim=(0, 7500)),
    # "A":       dict(ylabel=r"$A_\ell$: $\widehat\sigma_\ell / \sigma_\ell$",
    #                 yscale="linear", ylim=(-1.05, 2.05), target=1.0),
    # "r":       dict(ylabel=r"$r_\ell$: $\mathrm{corr}(\widehat{D}_\ell, D_\ell)$",
    #                 yscale="linear", ylim=(-1.05, 2.05), target=1.0),
    # "b":       dict(ylabel=r"$b_\ell$: $(\widehat\mu_\ell - \mu_\ell)/ \sigma_\ell$",
    #                 yscale="linear", ylim=(-1.05, 2.05), target=0.0),
    # "R2":      dict(ylabel=r"$R^2_\ell$: $1 - \mathrm{MSE}_\ell / \sigma_\ell$",
    #                 yscale="linear", ylim=(-1.05, 2.05), target=1.0),
    # "mse":     dict(ylabel=r"MSE$_\ell$",
    #                 yscale="linear", ylim=(0, 1e4)),
    # "tension": dict(ylabel=r"Tension$_\ell$",
    #                 yscale="linear", ylim=(-1.05, 2.05), target=1.0),
    # "rel_err": dict(ylabel=r"RE$_\ell$: $\langle(\widehat D_\ell-\mathring D_\ell)/\mathring D_\ell\rangle$", 
    #                 yscale="linear", ylim=(-0.2, 0.2), target=0.0),
    # "e_thry":  dict(ylabel=r"$\langle E_\ell\rangle/\sigma_{\mathrm{Th},\ell}$", 
    #                 yscale="linear", ylim=(-2.05, 2.05), target=0.0),
    # "e_cv":    dict(ylabel=r"$\langle E_\ell\rangle/\sigma_{\mathrm{CV},\ell}$", 
    #                 yscale="linear", ylim=(-2.05, 2.05), target=0.0),
}

# PLOT_SPEC = {
#     "A":       dict(ylabel=r"$A_\ell$",   yscale="log",    ylim=(0.5, 2.0),   target=1.0),
#     "r":       dict(ylabel=r"$r_\ell$",   yscale="linear", ylim=(0.0, 1.02),  ref="rho0"),
#     "b":       dict(ylabel=r"$b_\ell$",   yscale="linear", ylim=(-0.5, 0.5),  target=0.0),
#     "R2":      dict(ylabel=r"$R^2_\ell$", yscale="linear", ylim=(-1.0, 1.05), ref="rho0_sq", target=0.0),
#     "mse":     dict(ylabel=r"MSE$_\ell$", yscale="log",    ylim=None),
#     "e_thry":  dict(ylabel=r"$\langle E_\ell\rangle/\sigma_\ell$",                 yscale="linear", ylim=(-1.5, 1.5), target=0.0),
#     "e_cv":    dict(ylabel=r"$\langle E_\ell\rangle/\sigma_{\mathrm{CV},\ell}$",   yscale="linear", ylim=(-1.5, 1.5), target=0.0),
#     "tension": dict(ylabel=r"Tension$_\ell$", yscale="linear", ylim=(0, 2),  target=1.0),
#     # "tension": dict(ylabel=r"Tension$_\ell$", yscale="log", ylim=(1e-3, 3e1),  target=1.0),
#     "rel_err": dict(ylabel=r"RE$_\ell$", yscale="linear", ylim=(0, 0.1),  target=0.0),
#     "ref":     dict(ylim=(0, 7500))
# }


class NewMetricExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str="new_metric") -> None:
        super().__init__(cfg, stage_str)

        self.out_metric: Asset = self.assets_out["metric"]
        self.out_fig: Asset = self.assets_out["fig"]
        out_metric_handler: Config
        out_fig_handler: MPLFigure

        self.in_ps_pred: Asset = self.assets_in["pred"]
        self.in_ps_thry: AssetWithPathAlts = self.assets_in["thry"]
        self.in_ps_bank: Asset = self.assets_in["bank"]
        in_ps_pred_handler: NpyHandler
        in_ps_thry_handler: NpyHandler
        in_ps_bank_handler: NpyHandler

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax
        self.display_n_sims = cfg.n_show_cap
        self.fig_label = cfg.fig_model_name
        self._model_epochs = cfg.use_epochs
        self.fig_type = cfg.fig_type

        self.bank = None

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

        self.bank = self.in_ps_bank.read()
        self.prior = Prior(self.bank, self.ells)

        for split in self.splits:
            # if "F" not in split.name:
            #     continue
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(split, epoch)

    def process_epoch(self, split: Split, epoch) -> None:
        # print(self.in_ps_thry.path)
        # exit()
        thry = self.in_ps_thry.read()
        pred = self.in_ps_pred.read()
        result = compute_metrics(pred, thry, self.prior)

        if "R2" in result:                       # matched, varying target
            scalars = {
                "score":      score(result),
                "score_frac": score_fraction(result),
                "locked_0.5": locked_count(result, 0.5),
                "locked_0.9": locked_count(result, 0.9),
                "cleared":    cleared_count(result),
            }
            logger.info(
                f"[{split.name} | epoch {epoch}]  score={scalars['score']:.1f}  "
                f"frac={scalars['score_frac']:.3f}  locked@0.5={scalars['locked_0.5']}  "
                f"locked@0.9={scalars['locked_0.9']}  cleared={scalars['cleared']}"
            )
        else:                                    # single-spectrum target: tension only
            med_t = float(np.nanmedian(result["tension"]))
            scalars = {"tension_median": med_t}
            logger.info(
                f"[{split.name} | epoch {epoch}]  single-spectrum target — "
                f"tension only, median tension={med_t:.2f}"
            )

        self.out_metric.write(data=scalars)

        thry2 = np.atleast_2d(thry)
        ref_spectra = (thry2[0], pred[0])          # first matched theory, prediction

        title = f"{self.fig_label} · {split.name} · epoch {epoch}"
        for key, spec in PLOT_SPEC.items():
            if key not in result:
                continue
            fig = self._plot_metric(result, key, spec, ref_spectra, title)
            with self.name_tracker.set_context("metric", key):
                self.out_fig.write(fig=fig)
            plt.close(fig)     

        if "R2" in result:                       # matched, varying target
            decomp = decomposition(pred, thry, self.prior)
            fig = self._plot_decomposition(decomp, ref_spectra, title, rho0_sq=result["rho0_sq"])
            with self.name_tracker.set_context("metric", "R2_decomp"):
                self.out_fig.write(fig=fig)
            plt.close(fig)

            fig = self._plot_decomposition_oriented(
                decomp,
                ref_spectra,
                title,
                rho0_sq=result["rho0_sq"],
            )
            with self.name_tracker.set_context("metric", "R2_decomp_terms_sq"):
                self.out_fig.write(fig=fig)
            plt.close(fig)

            fig = self._plot_decomposition_terms(decomp, ref_spectra, title)
            with self.name_tracker.set_context("metric", "R2_decomp_terms_sn"):
                self.out_fig.write(fig=fig)
            plt.close(fig)

    def _plot_metric(self, result, key, spec, ref_spectra, title):
        """Two rows on a shared linear ell axis: top = first theory+pred spectra for
        context, bottom = the metric with fixed limits/scale."""
        ell = result["ell"]
        dl_thry, dl_pred = ref_spectra

        fig, (ax_top, ax) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 2.4]),   # no hspace here
        )

        # context: the first matched (theory, prediction) pair
        ax_top.plot(ell, dl_thry, color="k",  lw=1.0, label=r"theory $\mathring D_\ell$")
        ax_top.plot(ell, dl_pred, color="C0", lw=0.8, label=r"pred $\widehat D_\ell$")
        ax_top.set_ylabel(r"$D_\ell$")
        ax_top.set_ylim(*PLOT_SPEC["ref"]["ylim"])
        ax_top.legend(fontsize=7, loc="upper right")

        # the metric
        ax.plot(ell, result[key], color="C0", lw=0.8)
        ref = spec.get("ref")
        if ref is not None and ref in result:
            ax.plot(ell, result[ref], "k--", lw=1.0,
                    label=r"$\rho_0$" if ref == "rho0" else r"$\rho_0^2$")
            ax.legend(fontsize=8, loc="best")
        if "target" in spec:
            ax.axhline(spec["target"], color="0.5", ls=":", lw=1.0)

        ax.set_yscale(spec["yscale"])
        if spec.get("ylim") is not None:
            ax.set_ylim(spec["ylim"])
        ax.set_xlim(ell.min(), self.display_lmax)     # linear ell
        ax.set(xlabel=r"$\ell$", ylabel=spec["ylabel"])

        fig.suptitle(f"{key}  —  {title}", y=0.97)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.08)
        return fig
    
    def _plot_decomposition(self, decomp, ref_spectra, title, rho0_sq=None):
        ell = decomp["ell"]
        r2  = decomp["ceiling_r2"]
        R2  = decomp["R2_local"]
        mid = r2 - decomp["loss_bias"]            # r^2 - b^2 : boundary between the two losses

        fig, (ax_top, ax) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 2.4]),
        )

        dl_thry, dl_pred = ref_spectra
        ax_top.plot(ell, dl_thry, color="k",  lw=1.0, label=r"theory $\mathring D_\ell$")
        ax_top.plot(ell, dl_pred, color="C0", lw=0.8, label=r"pred $\widehat D_\ell$")
        ax_top.set_ylabel(r"$D_\ell$")
        ax_top.legend(fontsize=7, loc="upper right")
        ax_top.set_ylim(*PLOT_SPEC["ref"]["ylim"])

        ax.fill_between(ell, mid, r2, color="C3", alpha=0.20, label=r"$b^2$  (bias loss)")
        ax.fill_between(ell, R2, mid, color="C1", alpha=0.20, label=r"$(A-r)^2$  (calib. loss)")
        ax.plot(ell, r2, color="0.3", lw=1.0, label=r"$r^2$  (corr. ceiling)")
        ax.plot(ell, R2, color="C0", lw=1.2, label=r"$R^2_{\rm local}$  (achieved)")
        # if rho0_sq is not None:
        #     ax.plot(ell, rho0_sq, "k--", lw=1.0, label=r"$\rho_0^2$")
        ax.axhline(0, color="0.5", ls=":", lw=1.0)

        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(ell.min(), self.display_lmax)
        ax.set(xlabel=r"$\ell$", ylabel=r"$R^2_\ell$ budget")
        ax.legend(fontsize=7, loc="lower right", ncol=2)

        fig.suptitle(f"R2 decomposition  —  {title}", y=0.97)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.08)
        return fig

    def _plot_decomposition_oriented(self, decomp, ref_spectra, title, rho0_sq=None):
        """Decomposition-oriented view without presenting the terms as an additive sum.

        Plots the four local-normalized quantities that matter for interpreting
        R2_local:

            r^2          : correlation ceiling
            (A-r)^2      : calibration loss magnitude
            b^2          : bias loss magnitude
            R2_local     : achieved local R^2

        Unlike _plot_decomposition, this does not use filled regions or stacked
        visual accounting. It is a same-axis comparison of the ingredients.
        """
        ell = decomp["ell"]

        r2 = decomp["ceiling_r2"]
        calib = decomp["loss_calib"]
        bias = decomp["loss_bias"]
        R2 = decomp["R2_local"]

        fig, (ax_top, ax) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 2.4]),
        )

        # context: first matched theory/prediction pair
        dl_thry, dl_pred = ref_spectra
        ax_top.plot(ell, dl_thry, color="k",  lw=1.0, label=r"theory $\mathring D_\ell$")
        ax_top.plot(ell, dl_pred, color="C0", lw=0.8, label=r"pred $\widehat D_\ell$")
        ax_top.set_ylabel(r"$D_\ell$")
        ax_top.set_ylim(*PLOT_SPEC["ref"]["ylim"])
        ax_top.legend(fontsize=7, loc="upper right")

        # decomposition-oriented traces, not a sum/stack
        ax.plot(ell, r2,     color="0.25", lw=1.1, label=r"$r_\ell^2$")
        ax.plot(ell, calib,  color="C1",   lw=1.0, label=r"$(A_\ell-r_\ell)^2$")
        ax.plot(ell, bias,   color="C3",   lw=1.0, label=r"$b_\ell^2$")
        ax.plot(ell, R2,     color="C0",   lw=1.2, label=r"$R^2_{\mathrm{local},\ell}$")

        # if rho0_sq is not None:
        #     ax.plot(ell, rho0_sq, "k--", lw=1.0, label=r"$\rho_{0,\ell}^2$")

        ax.axhline(0, color="0.5", ls=":", lw=1.0)

        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(ell.min(), self.display_lmax)
        ax.set(xlabel=r"$\ell$", ylabel=r"local decomposition terms")
        ax.legend(fontsize=7, loc="best", ncol=2)

        fig.suptitle(f"R2 decomposition terms  —  {title}", y=0.97)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.08)
        return fig

    def _plot_decomposition_terms(self, decomp_terms, ref_spectra, title):
        ell = decomp_terms["ell"]

        fig, (ax_top, ax) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[1, 2.4]),
        )

        dl_thry, dl_pred = ref_spectra
        ax_top.plot(ell, dl_thry, color="k",  lw=1.0, label=r"theory $\mathring D_\ell$")
        ax_top.plot(ell, dl_pred, color="C0", lw=0.8, label=r"pred $\widehat D_\ell$")
        ax_top.set_ylabel(r"$D_\ell$")
        ax_top.set_ylim(*PLOT_SPEC["ref"]["ylim"])
        ax_top.legend(fontsize=7, loc="upper right")

        ax.plot(ell, decomp_terms["r"], color="0.2", lw=1.0, label=r"$r$")
        ax.plot(ell, decomp_terms["A_minus_r"], color="C1", lw=1.0, label=r"$A-r$")
        ax.plot(ell, decomp_terms["b"], color="C3", lw=1.0, label=r"$b$")
        ax.plot(ell, decomp_terms["R2_local"], color="C0", lw=1.2, label=r"$R^2_{\rm local}$")

        ax.axhline(0, color="0.5", ls=":", lw=1.0)
        ax.axhline(1, color="0.85", ls=":", lw=0.8)

        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(ell.min(), self.display_lmax)
        ax.set(xlabel=r"$\ell$", ylabel=r"local decomposition terms")
        ax.legend(fontsize=7, loc="best", ncol=2)

        fig.suptitle(f"R2 decomposition terms  —  {title}", y=0.97)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.08)
        return fig


class NewMetricSmExecutor(NewMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="new_metric_sm") -> None:
        super().__init__(cfg, stage_str)
