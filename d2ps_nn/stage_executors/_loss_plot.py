import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 
# from cmbml.analysis.stage_executors._16_loss_plot import make_loss_plot

class LossPlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="loss_plot")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_loss_csv = self.assets_in["loss_record"]
        # self.skip_n_values = 10

        self.fig_label = cfg.fig_model_name
        
        self.fig_type = cfg.get("fig_type", None)

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        fig, ax = make_loss_plot(some_path=self.in_loss_csv.path,
                                 fig_label=self.fig_label,
                                 ylim=(1e-8,1e7),
                                 sm_sigma2=20)
        self.out_fig.write(fig=fig)


def make_loss_plot(some_path, fig_label, figsize=(10,6), fig=None, ax=None, ylim=None, sm_sigma2=None):
        df = pd.read_csv(some_path)
        #skipping the first few rows if they are not needed
        # df = df[df["Epoch"] >= self.skip_n_values]

        # best_idx = df["Validation Loss"].idxmin()
        # best_val_loss = df.loc[best_idx, "Validation Loss"]
        # best_epoch = df.loc[best_idx, "Epoch"]

        if (fig is None and ax is not None) or (fig is not None and ax is None):
             raise ValueError("fig and ax must be provided together")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["Epoch"], df["Training Loss"], label="Training Loss", color="tab:blue", alpha=0.6, linewidth=0.5)
        # ax.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", color="tab:orange", alpha=0.8, linewidth=0.5)

        # loss_txt = exp_as_ltx(best_val_loss)
        # ax.annotate(
        #     f"Best validation loss\n${loss_txt}$ at epoch {best_epoch}",
        #     # xy=(best_epoch, best_val_loss),
        #     xytext=(50, 1e6),
        #     arrowprops=dict(arrowstyle="->", color="black"),
        #     fontsize=10,
        #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        # )

        train = df["Training Loss"].values
        # val = df["Validation Loss"].values

        train_log = np.log(train)
        # val_log = np.log(val)

        train_smoothed = gaussian_filter1d(train_log, sigma=5)
        # val_smoothed   = gaussian_filter1d(val_log, sigma=5)

        train_end_loss_txt = exp_as_ltx(np.exp(train_smoothed[-1]))
        # valid_end_loss_txt = exp_as_ltx(np.exp(val_smoothed[-1]))
        
        ax.plot(df["Epoch"], np.exp(train_smoothed), label="Smoothed Training Loss", linestyle="--", linewidth=2, color="tab:blue")
        # ax.plot(df["Epoch"], np.exp(val_smoothed), label="Smoothed Validation Loss", linestyle="--", linewidth=2, color="tab:orange")

        if sm_sigma2 is not None:
            train_smoothed = gaussian_filter1d(train_log, sigma=sm_sigma2)
            # val_smoothed   = gaussian_filter1d(val_log, sigma=sm_sigma2)

            ax.plot(df["Epoch"], np.exp(train_smoothed), 
                    label="Smoothed Training Loss", 
                    linestyle="--", linewidth=4, alpha=0.5, 
                    color="tab:blue")
            # ax.plot(df["Epoch"], np.exp(val_smoothed), 
            #         label="Smoothed Validation Loss", 
            #         linestyle="--", linewidth=4, alpha=0.5, 
            #         color="tab:orange")

            ax.annotate(
                f"Approx Train loss\n${train_end_loss_txt}$",
                xy=(df["Epoch"].iloc[-1], np.exp(train_smoothed[-1])),
                xytext=(150, 5e5),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
            # ax.annotate(
            #     f"Approx Valid loss\n${valid_end_loss_txt}$",
            #     xy=(df["Epoch"].iloc[-1], np.exp(val_smoothed[-1])),
            #     xytext=(200, 1.5e6),
            #     arrowprops=dict(arrowstyle="->", color="black"),
            #     fontsize=10,
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            # )

        ax.set_yscale('log')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training vs Validation Loss, {fig_label}")

        if ylim is not None:
            ax.set_ylim(*ylim)

        custom_legend = [
            Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2, label="Training Loss"),
            # Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=2, label="Validation Loss")
        ]
        ax.legend(handles=custom_legend)
        ax.grid(True)
        return fig, ax

def exp_as_ltx(some_val):
    val_sci = "{:.1e}".format(some_val)
    base, exp = val_sci.split("e")
    exp = int(exp)
    return f"{base} \\times 10^{{{exp}}}" if exp != 0 else base

class LRPlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="lr_plot")
        self.out_fig1: Figure = self.assets_out["fig1"]
        self.out_fig2: Figure = self.assets_out["fig2"]
        self.in_loss_csv = self.assets_in["loss_record"]
        # self.skip_n_values = 10

        self.fig_label = cfg.fig_model_name
        
        self.fig_type = cfg.get("fig_type", None)

    def execute(self):
        self.out_fig1.handler.set_fig_types(self.fig_type)
        self.out_fig2.handler.set_fig_types(self.fig_type)
        fig, ax = make_lr_plot(some_path=self.in_loss_csv.path,
                                 fig_label=self.fig_label,
                                 ylim=(1e2,1e8),
                                 sm_sigma2=20)
        
        self.out_fig1.write(fig=fig)

        fig, ax1, ax2 = make_dldlr_plot(some_path=self.in_loss_csv.path,
                                        fig_label=self.fig_label,
                                        ylim=(1e2,1e8))
        
        self.out_fig2.write(fig=fig)


def make_lr_plot(some_path, fig_label, figsize=(10,6), fig=None, ax=None, ylim=None, sm_sigma2=None):
        df = pd.read_csv(some_path)
        df = df.iloc[50:].reset_index(drop=True)

        if (fig is None and ax is not None) or (fig is not None and ax is None):
             raise ValueError("fig and ax must be provided together")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["Learning Rate"], df["Training Loss"], label="Training Loss", color="tab:blue", alpha=0.6, linewidth=0.5)

        train = df["Training Loss"].values
        train_log = np.log(train)

        train_smoothed = gaussian_filter1d(train_log, sigma=5)

        # train_end_loss_txt = exp_as_ltx(np.exp(train_smoothed[-1]))
        
        ax.plot(df["Learning Rate"], np.exp(train_smoothed), label="Smoothed Training Loss", linestyle="--", linewidth=2, color="tab:blue")
        # # ax.plot(df["Epoch"], np.exp(val_smoothed), label="Smoothed Validation Loss", linestyle="--", linewidth=2, color="tab:orange")

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss vs Learning Rate, {fig_label}")

        if ylim is not None:
            ax.set_ylim(*ylim)

        custom_legend = [
            Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2, label="Training Loss"),
            # Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=2, label="Validation Loss")
        ]
        ax.legend(handles=custom_legend)
        ax.grid(True)
        return fig, ax


def make_dldlr_plot(
    some_path,
    fig_label,
    figsize=(10, 6),
    fig=None,
    ax=None,
    ylim=None,
    sm_sigma=5,
    deriv_sigma=5,
    plot_deriv=True,
    deriv_ylim=None,
):
    df = pd.read_csv(some_path)
    df = df.iloc[50:].reset_index(drop=True)

    if (fig is None) ^ (ax is None):
        raise ValueError("fig and ax must be provided together")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Extract and clean
    lr = df["Learning Rate"].to_numpy(dtype=float)
    loss = df["Training Loss"].to_numpy(dtype=float)

    m = np.isfinite(lr) & np.isfinite(loss) & (lr > 0) & (loss > 0)
    lr = lr[m]
    loss = loss[m]

    # Make sure lr is strictly increasing for the derivative logic
    order = np.argsort(lr)
    lr = lr[order]
    loss = loss[order]

    x = np.log(lr)       # log-lr
    y = np.log(loss)     # log-loss

    # Smooth log-loss (your current approach, but in log-space)
    y_s = gaussian_filter1d(y, sigma=sm_sigma)

    # Colors
    loss_color = "tab:blue"
    smooth_color = "tab:orange"
    deriv_color = "tab:green"

    # Plot loss + smoothed loss
    ax.plot(lr, loss, 
            label="Training Loss", 
            alpha=0.6, 
            linewidth=0.5, 
            color=loss_color
            )
    ax.plot(lr, np.exp(y_s), 
            label="Smoothed Training Loss", 
            linestyle="--", 
            linewidth=2, 
            color=smooth_color
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss vs Learning Rate, {fig_label}")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)

    # Derivative plot: d log(loss) / d log(lr)
    ax2 = None
    if plot_deriv:
        # mask out repeated lr (burn-in plateau)
        ramp_mask = np.concatenate([[False], np.diff(lr) > 0])

        lr_r = lr[ramp_mask]
        x_r = np.log(lr_r)
        y_r = np.log(loss[ramp_mask])

        y_r_s = gaussian_filter1d(y_r, sigma=deriv_sigma)
        d = np.gradient(y_r_s, x_r)

        ax2 = ax.twinx()
        ax2.plot(lr_r, d, linewidth=1.5, alpha=0.8, label="dlog(loss)/dlog(lr)")
        ax2.set_xscale("log")
        ax2.set_ylabel("d log(loss) / d log(lr)")
        if deriv_ylim is not None:
            ax2.set_ylim(*deriv_ylim)
        ax2.axhline(0.0, linewidth=1.0, alpha=0.5)

        spike_idx = np.argmax(d > 1)
        if spike_idx == 0:
            spike_idx = len(d)
        search_region = d[:spike_idx]
        best_idx = np.argmin(search_region)
        best_lr = lr_r[best_idx]
        ax.axvline(best_lr, color="k", linestyle=":", linewidth=1)
        ax.text(
            best_lr*1.1,
            0.7,
            f"best LR ≈ {best_lr:.1e}",
            rotation=90,
            color="k",
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )

    # ---- Legend ----
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(lines1 + lines2, labels1 + labels2)
    return fig, ax, ax2