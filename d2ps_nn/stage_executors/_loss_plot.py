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


def make_loss_plot(some_path, fig_label, figsize=(10,6), fig=None, ax=None, ylim=None, xlim=None, sm_sigma2=None, show_sigma2=False):
        df = pd.read_csv(some_path)
        #skipping the first few rows if they are not needed
        # df = df[df["Epoch"] >= self.skip_n_values]

        sm_sigma = 100

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

            ax.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", color="tab:orange", alpha=0.2, linewidth=0.5)

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
