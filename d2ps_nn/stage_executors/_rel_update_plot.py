import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 


class RelativeUpdatePlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="rel_update_plot")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_loss_csv = self.assets_in["loss_record"]

        self.fig_label = cfg.fig_model_name
        
        self.fig_type = cfg.get("fig_type", None)

        self.n_epochs = cfg.model.train.n_epochs

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        fig, ax = make_relative_update_plot(some_path=self.in_loss_csv.path,
                                 fig_label=self.fig_label,
                                 ylim=(1e-7, 1e0),
                                 xlim=(0,self.n_epochs))
        self.out_fig.write(fig=fig)


def make_relative_update_plot(some_path, fig_label, figsize=(10,6), fig=None, ax=None, ylim=None, xlim=None):
        df = pd.read_csv(some_path)

        df = df.dropna(subset=["Epoch", "Relative Update"])
        df = df[df["Relative Update"] > 0]

        if (fig is None and ax is not None) or (fig is not None and ax is None):
             raise ValueError("fig and ax must be provided together")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["Epoch"], df["Relative Update"], label="Relative Update", color="tab:blue", alpha=1.0, linewidth=0.5)

        rel_update = df["Relative Update"].values
        rel_update_log = np.log(rel_update)
        rel_update_smoothed = gaussian_filter1d(rel_update_log, sigma=2)
        ax.plot(df["Epoch"], np.exp(rel_update_smoothed), label="Smoothed Relative Update", linestyle="--", linewidth=2, alpha=0.5, color="tab:blue")

        ax.set_yscale('log')

        ax.set_xlabel("Step")
        ax.set_ylabel("Relative Update")
        ax.set_title(f"{fig_label}")

        if ylim is not None:
            ax.set_ylim(*ylim)

        if xlim is not None:
            ax.set_xlim(*xlim)

        for y in [1e-5, 1e-4, 1e-3]:
            ax.axhline(y, linestyle=":", linewidth=1, alpha=0.6)

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig, ax


def exp_as_ltx(some_val):
    if np.isnan(some_val):
        return np.nan
    val_sci = "{:.1e}".format(some_val)
    base, exp = val_sci.split("e")
    exp = int(exp)
    return f"{base} \\times 10^{{{exp}}}" if exp != 0 else base
