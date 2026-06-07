import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from d2ps_nn.utils.clipped_gaussian_smooth import clipped_gaussian_smooth
from d2ps_nn.utils.decimate_for_plotting import decimate_for_plot


STAT_CONFIGS = {
    "score": {"ylim": (-0.05, "N_ell"), "label": r"$\sum_{\ell=2}^{\ell_\mathrm{max}} \left[\mathrm{FVE}_\ell\right]_+$"},
    "locked": {"ylim": (-0.05, "N_ell"), "label": r"# $\mathrm{FVE}_\ell > $ 50%"},
}


class ScorePlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="show_score")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_parquet = self.assets_in["loss_parquet"]

        self.fig_label = cfg.fig_model_name
        self.fig_type = cfg.get("fig_type", None)
        self.n_epochs = cfg.model.train.n_epochs
        self.lmax     = cfg.lmax

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        df = self.in_parquet.read()

        for stat, stat_cfg in STAT_CONFIGS.items():
            ylim = stat_cfg.get("ylim")
            if ylim[1] == "N_ell":
                ylim = (ylim[0], self.lmax)
            fig, ax = make_ell_stat_plot(
                df=df,
                this_stat=stat,
                fig_label=self.fig_label,
                ylim=ylim,
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
    """Plot single tracking metric

    Parameters
    ----------
    df : DataFrame already loaded from the loss CSV.
    this_stat : Column prefix — one of 'score', 'fixed'.
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

    n_epochs = df["Epoch"].iloc[-1]
    if sm == True:
        sm_sigma = min(n_epochs // 20, 100)
    else:
        sm_sigma = sm

    smoothed = clipped_gaussian_smooth(df[this_stat].values, sigma=sm_sigma, end_method="interior")

    # Plot
    ax.plot(df["Epoch"], df[this_stat], alpha=0.2, linewidth=0.5)
    ax.plot(df["Epoch"], smoothed, alpha=0.85, linewidth=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"{this_stat} per $\ell$ — {fig_label}")

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    return fig, ax
