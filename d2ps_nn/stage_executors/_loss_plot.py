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

        self.n_epochs = cfg.model.train.n_epochs

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        fig, ax = make_loss_plot(some_path=self.in_loss_csv.path,
                                 fig_label=self.fig_label,
                                 ylim=(1e3,1e6),
                                 xlim=(0,self.n_epochs),
                                 sm_sigma2=None)
        self.out_fig.write(fig=fig)


class LossDeltaPlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="loss_delta_plot")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_loss_csv = self.assets_in["loss_record"]
        # self.skip_n_values = 10

        self.fig_label = cfg.fig_model_name
        
        self.fig_type = cfg.get("fig_type", None)

        self.n_epochs = cfg.model.train.n_epochs

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        fig, ax = make_loss_delta_ratio_plot(some_path=self.in_loss_csv.path,
                                             fig_label=self.fig_label,
                                            #  ylim=(1e3,1e6),
                                             xlim=(0,self.n_epochs),
                                            #  sm_sigma2=None
                                             )
        self.out_fig.write(fig=fig)


def make_loss_plot(some_path, fig_label, figsize=(12,7), fig=None, ax=None, ylim=None, xlim=None, sm_sigma2=None, show_sigma2=False):
        df = pd.read_csv(some_path)
        #skipping the first few rows if they are not needed
        # df = df[df["Epoch"] >= self.skip_n_values]

        sm_sigma = 5

        # df = df.dropna()

        if (fig is None and ax is not None) or (fig is not None and ax is None):
             raise ValueError("fig and ax must be provided together")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(df["Epoch"], df["Training Loss"], label="Training Loss", color="tab:blue", alpha=1.0, linewidth=0.5)

        train = df["Training Loss"].values
        train_log = np.log(train)
        train_smoothed = gaussian_filter1d(train_log, sigma=sm_sigma)
        train_end_loss_txt = exp_as_ltx(np.exp(train_smoothed[-1]))
        ax.plot(df["Epoch"], np.exp(train_smoothed), label="Smoothed Training Loss", linestyle="--", linewidth=2, alpha=0.5, color="tab:blue")

        if sm_sigma2 is not None:
            train_smoothed = gaussian_filter1d(train_log, sigma=sm_sigma2)
            train_end_loss_txt = exp_as_ltx(np.exp(train_smoothed[-1]))

            if show_sigma2:
                ax.plot(df["Epoch"], np.exp(train_smoothed), 
                        label=f"Smoothed Training Loss (sigma={sm_sigma2})", 
                        linestyle="--", linewidth=4, alpha=0.5, 
                        color="tab:blue")

        ax.annotate(
            f"Approx Train loss\n${train_end_loss_txt}$",
            xy=(df["Epoch"].iloc[-1], np.exp(train_smoothed[-1])),
            xytext=(0.8, 0.3),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        include_valid = "Validation Loss" in df.columns

        if include_valid:
            best_idx = df["Validation Loss"].idxmin()
            best_val_loss = df.loc[best_idx, "Validation Loss"]
            best_epoch = df.loc[best_idx, "Epoch"]

            ax.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", color="tab:orange", alpha=1.0, linewidth=0.5)

            val = df["Validation Loss"].values
            val_log = np.log(val)
            val_smoothed   = gaussian_filter1d(val_log, sigma=sm_sigma)
            valid_end_loss_txt = exp_as_ltx(np.exp(val_smoothed[-1]))
            ax.plot(df["Epoch"], np.exp(val_smoothed), label="Smoothed Validation Loss", linestyle="--", linewidth=2, alpha=0.5, color="tab:orange")

            loss_txt = exp_as_ltx(best_val_loss)
            ax.annotate(
                f"Best validation loss\n${loss_txt}$ at epoch {best_epoch}",
                xy=(best_epoch, best_val_loss),
                xytext=(0.65, 0.65),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            if sm_sigma2 is not None:
                val_smoothed   = gaussian_filter1d(val_log, sigma=sm_sigma2)
                valid_end_loss_txt = exp_as_ltx(np.exp(val_smoothed[-1]))
                if show_sigma2:
                    ax.plot(df["Epoch"], np.exp(val_smoothed), 
                            label=f"Smoothed Validation Loss (sigma={sm_sigma2})", 
                            linestyle="--", linewidth=4, alpha=0.5, 
                            color="tab:orange")

            ax.annotate(
                f"Approx Valid loss\n${valid_end_loss_txt}$",
                xy=(df["Epoch"].iloc[-1], np.exp(val_smoothed[-1])),
                xytext=(0.8, 0.5),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

        ax.set_yscale('log')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training vs Validation Loss, {fig_label}")

        if ylim is not None:
            ax.set_ylim(*ylim)

        if xlim is not None:
            ax.set_xlim(*xlim)

        custom_legend = [
            Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2, label="Training Loss")
        ]
        if include_valid:
            custom_legend.append(Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=2, label="Validation Loss"))
        ax.legend(handles=custom_legend)
        ax.grid(True)
        fig.tight_layout()
        return fig, ax


def make_loss_delta_ratio_plot(some_path, fig_label, figsize=(12,7), fig=None, ax=None,
                               xlim=None, delta_ylim=None, 
                            #    ratio_ylim=None,
                               sm_sigma=5, delta_yscale="linear"):
    df = pd.read_csv(some_path)

    if "Validation Loss" not in df.columns:
        raise ValueError("Delta/ratio plot requires a 'Validation Loss' column")

    # both must exist per-row; one NaN poisons the whole gaussian_filter1d output
    df = df.dropna(subset=["Training Loss", "Validation Loss"])

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError("fig and ax must be provided together")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epoch = df["Epoch"].values
    train = df["Training Loss"].values
    val   = df["Validation Loss"].values

    delta_c = "k"   # left axis
    # delta_c = "tab:purple"   # left axis
    # ratio_c = "tab:green"  # right axis

    # log-space smoothing, consistent with make_loss_plot, so these curves
    # equal the gap/ratio between the smoothed loss curves you already plot
    train_sm = np.exp(gaussian_filter1d(np.log(train), sigma=sm_sigma))
    val_sm   = np.exp(gaussian_filter1d(np.log(val),   sigma=sm_sigma))

    # ---- left axis: delta = val - train ----
    delta_raw = val - train
    delta_sm  = val_sm - train_sm
    ax.axhline(0, color=delta_c, linewidth=0.8, alpha=0.4)
    ax.plot(epoch, delta_raw, color=delta_c, alpha=1.0, linewidth=0.5)
    ax.plot(epoch, delta_sm, linestyle="--", linewidth=2, alpha=0.8, color=delta_c)
    ax.set_ylabel("Validation - Training  (delta)", color=delta_c)
    ax.tick_params(axis="y", labelcolor=delta_c)
    ax.set_yscale(delta_yscale)  # "symlog" if the early transient dominates

    # # ---- right axis: ratio = val / train ----
    # ax2 = ax.twinx()
    # ratio_raw = val / train
    # ratio_sm  = val_sm / train_sm
    # ax2.axhline(1.0, color=ratio_c, linewidth=0.8, alpha=0.4)
    # ax2.plot(epoch, ratio_raw, color=ratio_c, alpha=1.0, linewidth=0.5)
    # ax2.plot(epoch, ratio_sm, linestyle="--", linewidth=2, alpha=0.8, color=ratio_c)
    # ax2.set_ylabel("Validation / Training  (ratio)", color=ratio_c)
    # ax2.tick_params(axis="y", labelcolor=ratio_c)
    # ax2.set_yscale("log")  # ratio sits near 1.0; log keeps it symmetric

    # ---- best epoch = lowest validation loss ----
    best_idx   = df["Validation Loss"].idxmin()
    best_epoch = df.loc[best_idx, "Epoch"]
    best_val   = df.loc[best_idx, "Validation Loss"]
    best_train = df.loc[best_idx, "Training Loss"]
    best_gap   = best_val - best_train
    best_ratio = best_val / best_train

    ax.axvline(best_epoch, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.7)
    gap_txt = f"${exp_as_ltx(best_gap)}$" if abs(best_gap) >= 100 else f"{best_gap:.1f}"
    ax.annotate(
        f"Best val epoch {best_epoch}\ndelta = {gap_txt}",  #\nratio = {best_ratio:.3f}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Epoch")
    ax.set_title(f"Generalization Gap, {fig_label}")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if delta_ylim is not None:
        ax.set_ylim(*delta_ylim)
    # if ratio_ylim is not None:
    #     ax2.set_ylim(*ratio_ylim)

    custom_legend = [
        Line2D([0], [0], color=delta_c,  linestyle="-", linewidth=2, label="Val - Train (left)"),
        # Line2D([0], [0], color=ratio_c,  linestyle="-", linewidth=2, label="Val / Train (right)"),
    ]
    ax.legend(handles=custom_legend, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


class LearningRatePlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="lr_plot")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_loss_csv = self.assets_in["loss_record"]
        # self.skip_n_values = 10

        self.fig_label = cfg.fig_model_name
        
        self.fig_type = cfg.get("fig_type", None)

        self.n_epochs = cfg.model.train.n_epochs

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        fig, ax = make_learning_rate_plot("Training Loss",
                                          some_path=self.in_loss_csv.path,
                                          fig_label=self.fig_label
                                          )
        with self.name_tracker.set_context("split", "train"):
            self.out_fig.write(fig=fig)
        fig, ax = make_learning_rate_plot("Validation Loss",
                                          some_path=self.in_loss_csv.path,
                                          fig_label=self.fig_label
                                          )
        with self.name_tracker.set_context("split", "valid"):
            self.out_fig.write(fig=fig)


def make_learning_rate_plot(signal, some_path, fig_label, figsize=(10,6), fig=None, ax=None, ylim=None, xlim=None):
        df = pd.read_csv(some_path)

        df = df.dropna(subset=["Step", "Learning Rate", signal])

        if (fig is None and ax is not None) or (fig is not None and ax is None):
             raise ValueError("fig and ax must be provided together")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["Learning Rate"], df[signal], label="Validation Loss", color="tab:blue", alpha=1.0, linewidth=0.5)

        valid = df[signal].values
        valid_log = np.log(valid)
        valid_smoothed = gaussian_filter1d(valid_log, sigma=5)
        valid_end_loss_txt = exp_as_ltx(np.exp(valid_smoothed[-1]))
        ax.plot(df["Learning Rate"], np.exp(valid_smoothed), label="Smoothed Validation Loss", linestyle="--", linewidth=2, alpha=0.5, color="tab:blue")

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title(f"Validation Loss vs Learning Rate, {fig_label}")

        if ylim is not None:
            ax.set_ylim(*ylim)

        if xlim is not None:
            ax.set_xlim(*xlim)

        custom_legend = [
            Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2, label="Validation Loss")
        ]
        ax.legend(handles=custom_legend)
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
