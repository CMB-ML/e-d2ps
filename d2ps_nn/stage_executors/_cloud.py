import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from omegaconf import DictConfig
import pysm3.units as u

from cmbml.core import Split, Asset, AssetWithPathAlts
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, NpyHandler
from cmbml.utils.number_report import float_to_latex
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.find_peaks import find_crits, PeakFitError


logger = logging.getLogger(__name__)


class ShowPredictionsCloudsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_predictions_clouds")

        self.out_fig: Asset = self.assets_out["fig"]
        out_lut_handler: Figure

        self.in_ps_pred: Asset = self.assets_in["pred"]
        self.in_ps_thry: AssetWithPathAlts = self.assets_in["thry"]
        # self.in_report: Asset = self.assets_in["report"]
        in_pred_handler: NpyHandler
        in_thry_handler: NpyHandler

        self.units = u.Unit(cfg.scenario.units)
        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax
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
            # if "F" not in split.name:
            #     continue
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(split, epoch)       

    def process_epoch(self, split:Split, epoch) -> None:
        thry = self.in_ps_thry.read()
        thry = np.atleast_2d(thry)
        pred = self.in_ps_pred.read()

        fig = self._plot_prediction_relerr_cloud(
            all_pred_ps=pred,
            all_theory_ps=thry,
            split_name=split.name,
            epoch=epoch,
            show_max_l=self.ells[-1],
        )

        self.out_fig.write(fig=fig)
        plt.close(fig)

    def _prediction_on_used_ells(self, ps_pred, part_ells):
        """Return prediction values on self.ells.

        Handles either:
        * already-cropped prediction with len == len(self.ells)
        * full-width prediction indexed by ell
        """
        ps_pred = np.asarray(ps_pred, dtype=float).squeeze()

        if ps_pred.ndim != 1:
            raise ValueError(f"expected 1-D prediction spectrum, got shape {ps_pred.shape}")

        if ps_pred.shape[0] == part_ells.size:
            return ps_pred

        if ps_pred.shape[0] > int(part_ells.max()):
            return ps_pred[part_ells]

        raise ValueError(
            f"prediction length {ps_pred.shape[0]} is neither len(self.ells)={part_ells.size} "
            f"nor full-width enough to index max ell {part_ells.max()}"
        )

    def _theory_on_used_ells(self, ps_thry, part_ells):
        """Return theory values on self.ells.

        Handles either:
        * already-cropped theory with len == len(self.ells)
        * full-width theory indexed by ell
        """
        ps_thry = np.asarray(ps_thry, dtype=float).squeeze()

        if ps_thry.ndim != 1:
            raise ValueError(f"expected 1-D theory spectrum, got shape {ps_thry.shape}")

        if ps_thry.shape[0] == part_ells.size:
            return ps_thry

        if ps_thry.shape[0] > int(part_ells.max()):
            return ps_thry[part_ells]

        raise ValueError(
            f"theory length {ps_thry.shape[0]} is neither len(self.ells)={part_ells.size} "
            f"nor full-width enough to index max ell {part_ells.max()}"
        )

    def _theory_for_prediction(self, all_theory_ps, i):
        """Return the theory spectrum paired with prediction i.

        Supports either:
        * one fixed theory spectrum for all predictions
        * one matched theory spectrum per prediction
        """
        if len(all_theory_ps) == 1:
            return all_theory_ps[0]

        return all_theory_ps[i]

    def _plot_prediction_relerr_cloud(
        self,
        all_pred_ps,
        all_theory_ps,
        split_name,
        epoch,
        show_max_l,
    ):
        """Plot prediction cloud and relative-error cloud.

        Top:
            cloud of predictions only, no target/theory spectra drawn.

        Bottom:
            cloud of per-simulation relative errors:
                (prediction - theory) / theory

        Peak ticks:
            theory peak locations are marked at the bottom of the top axis.
            prediction peak locations are marked at the top of the top axis.
        """
        part_ells = np.asarray(self.ells)

        if len(all_theory_ps) not in (1, len(all_pred_ps)):
            raise ValueError(
                f"Expected either one fixed theory spectrum or one theory per prediction; "
                f"got {len(all_theory_ps)} theories and {len(all_pred_ps)} predictions."
            )

        fixed_theory = len(all_theory_ps) == 1
        marked_fixed_theory_peaks = False

        trace_color = "C0"
        peak_color = "0.15"

        pred_lw = 0.25
        err_lw = 0.25
        peak_lw = 0.45
        trace_alpha = 0.25
        peak_alpha = 0.35

        fig, (ax_top, ax_err) = plt.subplots(
            2, 1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw=dict(height_ratios=[1, 2.4]),
        )

        fixed_theory = len(all_theory_ps) == 1
        marked_fixed_theory_peaks = False

        for i, ps_pred_raw in enumerate(all_pred_ps):
            ps_thry_raw = self._theory_for_prediction(all_theory_ps, i)

            ps_pred = self._prediction_on_used_ells(ps_pred_raw, part_ells)
            ps_thry_used = self._theory_on_used_ells(ps_thry_raw, part_ells)

            # top: prediction cloud
            ax_top.plot(
                part_ells,
                ps_pred,
                color=trace_color,
                alpha=trace_alpha,
                linewidth=pred_lw,
            )

            # Mark theory peaks once for fixed-theory splits, otherwise once per matched theory.
            should_mark_theory = (not fixed_theory) or (fixed_theory and not marked_fixed_theory_peaks)

            if should_mark_theory:
                try:
                    thry_full = np.asarray(ps_thry_raw, dtype=float).squeeze()
                    thry_ells = np.arange(thry_full.shape[0])

                    keep = (thry_ells >= part_ells.min()) & (thry_ells <= show_max_l)
                    crits_thry = find_crits(thry_full[keep], thry_ells[keep])

                    mark_crits_at_axis_edge(
                        ax_top,
                        crits_thry,
                        color=peak_color,
                        where="bottom",
                        tick_frac=0.025,
                        linewidth=peak_lw,
                        alpha=peak_alpha,
                        ls="-",
                    )

                    marked_fixed_theory_peaks = True

                except PeakFitError as e:
                    logger.warning(
                        f"Could not fit theory peaks for sim {i}! "
                        f"Peak index={e.peak_index}, reason: {e}"
                    )
                except Exception as e:
                    logger.warning(f"Could not mark theory peaks for sim {i}: {e}")

            # Mark prediction peaks for every prediction trace.
            try:
                crits_pred = find_crits(ps_pred, part_ells)

                mark_crits_at_axis_edge(
                    ax_top,
                    crits_pred,
                    color=peak_color,
                    where="top",
                    tick_frac=0.025,
                    linewidth=peak_lw,
                    alpha=peak_alpha,
                    ls="--",
                )

            except PeakFitError as e:
                logger.warning(
                    f"Could not fit prediction peaks for sim {i}! "
                    f"Peak index={e.peak_index}, reason: {e}"
                )
            except Exception as e:
                logger.warning(f"Could not mark prediction peaks for sim {i}: {e}")

            # bottom: relative-error cloud
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.divide(
                    ps_pred - ps_thry_used,
                    ps_thry_used,
                    out=np.full_like(ps_pred, np.nan, dtype=float),
                    where=(ps_thry_used != 0),
                )

            ax_err.plot(
                part_ells,
                rel_err,
                color=trace_color,
                alpha=trace_alpha,
                linewidth=err_lw,
            )

        # Legend proxies: no per-trace labels
        pred_proxy = Line2D(
            [0], [0],
            color=trace_color,
            linewidth=1.0,
            alpha=0.8,
            label="Predictions",
        )
        thry_peak_proxy = Line2D(
            [0], [0],
            color=peak_color,
            linewidth=1.0,
            linestyle="-",
            label="Theory peaks (bottom)",
        )
        pred_peak_proxy = Line2D(
            [0], [0],
            color=peak_color,
            linewidth=1.0,
            linestyle="--",
            label="Prediction peak (top)",
        )

        ax_top.legend(
            handles=[pred_proxy, thry_peak_proxy, pred_peak_proxy],
            fontsize=8,
            loc="upper right",
        )

        unit_str = (self.units**2).to_string("latex_inline")

        ax_top.set_ylabel(r"$\widehat D_\ell$ [" + unit_str + "]")
        ax_top.set_ylim(0, 6700)
        ax_top.set_xlim(part_ells.min(), show_max_l)

        ax_err.axhline(0, color="0.5", linestyle=":", linewidth=1.0)
        ax_err.set_ylabel(r"$\langle(\widehat D_\ell-\mathring D_\ell)/\mathring D_\ell\rangle$")
        ax_err.set_xlabel(r"$\ell$")
        ax_err.set_xlim(part_ells.min(), show_max_l)

        # # Symmetric robust y-limit for relative error, so one bad trace does not
        # # destroy readability.
        # err_lines = [line.get_ydata() for line in ax_err.lines]
        # if err_lines:
        #     err_all = np.concatenate(err_lines)
        #     finite = np.isfinite(err_all)
        #     if np.any(finite):
        #         q = np.nanpercentile(np.abs(err_all[finite]), 99)
        #         y = max(0.05, min(1.0, 1.15 * q))
        #         ax_err.set_ylim(-y, y)
        ax_err.set_ylim(-0.2, 0.2)

        fig.suptitle(
            f"Prediction cloud and relative error — {split_name}, epoch {epoch}, {self.fig_label}",
            y=0.98,
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.08)

        return fig


def mark_crits_at_axis_edge(
    ax: plt.Axes,
    crit_ells,
    color,
    where="bottom",
    tick_frac=0.025,
    linewidth=1.2,
    alpha=1.0,
    ls=None,
):
    """Mark extrema as short ticks at the top or bottom edge of an axis.

    where="bottom" -> ticks rise from bottom edge.
    where="top"    -> ticks drop from top edge.
    """
    if where not in {"bottom", "top"}:
        raise ValueError(f"where must be 'bottom' or 'top', got {where!r}")

    trans = ax.get_xaxis_transform()  # x=data, y=axes fraction

    if where == "bottom":
        y_pos = [0.0, tick_frac]
    else:
        y_pos = [1.0 - tick_frac, 1.0]

    for ell in crit_ells:
        if ell is None:
            continue

        ax.plot(
            [ell, ell],
            y_pos,
            transform=trans,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            ls=ls,
            solid_capstyle="butt",
            clip_on=False,
        )