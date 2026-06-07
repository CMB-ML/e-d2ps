import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 
# from cmbml.analysis.stage_executors._16_loss_plot import make_loss_plot

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from d2ps_nn.utils.clipped_gaussian_smooth import clipped_gaussian_smooth


logger = logging.getLogger(__name__)

STAT_CONFIGS = {
    "R":   {"ylim": (-0.05, 1.05), "label": r"S.D. Ratio($\ell$): $\sigma(D_\ell^\mathrm{pred}) / \sigma(D_\ell^\mathrm{true})$"},
    "rho": {"ylim": (-0.05, 1.05), "label": r"Correlation($\ell$): $\rho_\ell$"},
    "FVE": {"ylim": (-0.05, 1.05), "label": r"FVE($\ell$): $1-\langle (D_\ell^\mathrm{pred} - D_\ell^\mathrm{true} )^2 \rangle / \sigma^2 (D_\ell^\mathrm{true})$"},
}


class PerEllStatsPlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="show_train_stats_plots")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_parquet = self.assets_in["loss_parquet"]

        self.fig_label = cfg.fig_model_name
        self.fig_type = cfg.get("fig_type", None)
        self.n_epochs = cfg.model.train.n_epochs

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        df = self.in_parquet.read()
        df = df[(df["Epoch"] <= self.n_epochs)].reset_index(drop=True)

        for stat, stat_cfg in STAT_CONFIGS.items():
            fig, ax = make_ell_stat_plot(
                df=df,
                this_stat=stat,
                fig_label=self.fig_label,
                ylim=stat_cfg.get("ylim"),
                xlim=(0, self.n_epochs),
                ylabel=stat_cfg.get("label"),
                sm=True
            )
            with self.name_tracker.set_context("stat", stat):
                self.out_fig.write(fig=fig)
            plt.close(fig)


def get_ell_range(df: pd.DataFrame, stat: str) -> list[int]:
    """Extract sorted ell indices from columns named {stat}_ell{n}."""
    prefix = f"{stat}_ell"
    return sorted(
        int(col[len(prefix):])
        for col in df.columns
        if col.startswith(prefix) and col[len(prefix):].isdigit()
    )


def make_ell_stat_plot(
    df: pd.DataFrame,
    this_stat: str,
    fig_label: str,
    figsize=(10, 6),
    fig=None,
    ax=None,
    ylim=None,
    xlim=None,
    ylabel=None,
    sm=None
):
    """Plot one per-ell statistic (R, rho, or FVE) across all ells vs. epoch.

    Parameters
    ----------
    df : DataFrame already loaded from the loss CSV.
    this_stat : Column prefix — one of 'R', 'rho', 'FVE'.
    fig_label : String appended to the plot title (e.g. model name).
    ell_range : Iterable of ell integers whose columns exist in df.
    figsize : Passed to plt.subplots when no existing axes are supplied.
    fig, ax : Optional pre-existing Figure/Axes pair (must be given together).
    ylim, xlim : Optional (lo, hi) axis limits.
    """
    if (fig is None) != (ax is None):
        raise ValueError("fig and ax must be provided together or not at all")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ell_range = get_ell_range(df, this_stat)

    cmap = plt.get_cmap("viridis")
    n = len(list(ell_range))  # total number of traces for colour spacing

    n_epochs = df["Epoch"].iloc[-1]
    if sm == True:
        sm_sigma = min(n_epochs // 20, 100)
    else:
        sm_sigma = sm

    logger.info("Smoothing curves.")
    smoothed = {}
    for ell in ell_range:
        col = f"{this_stat}_ell{ell}"
        if col not in df.columns:
            continue
        if sm:
            smoothed[ell] = clipped_gaussian_smooth(df[col].values, sigma=sm_sigma, end_method="interior")
        else:
            smoothed[ell] = df[col].values

    logger.info("Done smoothing.")

    # Last ell whose final smoothed value exceeds 0.5
    above_half = [ell for ell, s in smoothed.items() if s[-1] > 0.5]
    last_above_half = above_half[-1] if above_half else None

    labelled = {ell_range[0], ell_range[-1]}
    if last_above_half is not None:
        labelled.add(last_above_half)

    # Plot
    cmap = plt.get_cmap("viridis")
    for i, ell in enumerate(ell_range):
        if ell not in smoothed:
            continue
        color = cmap(i / max(n - 1, 1))
        ax.plot(df["Epoch"], df[f"{this_stat}_ell{ell}"],
                color=color, alpha=0.2, linewidth=0.5)

    for i, ell in enumerate(ell_range):
        if ell not in smoothed:
            continue
        color = cmap(i / max(n - 1, 1))
        label = rf"$\ell={ell}$" if ell in labelled else None
        ax.plot(df["Epoch"], smoothed[ell],
                color=color, alpha=0.85, linewidth=0.8, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"{this_stat} per $\ell$ — {fig_label}")

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.legend(
        ncol=1,
        fontsize="small",
        # loc="lower middle",
    )
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    return fig, ax
