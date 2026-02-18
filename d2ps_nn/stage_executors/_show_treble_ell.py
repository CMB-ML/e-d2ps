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


class ShowTrebleEll(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_treble_ell")

        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: Figure

        self.in_thry: Asset = self.assets_in["thry"]
        self.in_pred: Asset = self.assets_in["pred"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        in_pv_report: NumpyPowerSpectrum

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

        self.fsky = get_fsky(self.in_mask)

        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split {split.name}")
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch()

    def process_epoch(self) -> None:
        pred = self.in_pred.read()
        thry = self.in_thry.read()
        if len(thry.shape) == 1:
            thry = thry[np.newaxis, :]
        make_fig(ells=self.ells,
                 thry=thry,
                 pred=pred,
                 fsky=self.fsky,
                 units=self.units,
                 fig_label=self.fig_label,
                 fig_handler=self.out_fig)

def get_fsky(mask_asset: Asset):
    if mask_asset is None:
        return 1
    mask = mask_asset.read()
    return np.mean(mask**2)

def make_fig(ells, thry, pred, fsky, units, fig_label, fig_handler):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top plot: prediction mean with +/-1 s.d. band
    ex_cosmic_stdev = dl2cosm_stdev(ells, thry[0], fsky)
    mnus_cv = thry[0] - ex_cosmic_stdev
    plus_cv = thry[0] + ex_cosmic_stdev
    ax1.fill_between(ells, mnus_cv, plus_cv, 
                     color="k", 
                     edgecolor=None,
                     alpha=0.2, 
                     label=r"$\pm1\sigma$")
    ax1.plot(ells, thry[0], color='black', lw=1, label="Theory")
    ax1.plot(ells, pred[0], label="Prediction, sim0000")

    ax1.set_ylabel(r"$D_\ell$ [" + units.to_string("latex_inline") + "]")
    ax1.legend(loc="best")

    # Bottom plot: relative error
    n_show = 200
    # Individual "shadows"
    for i in range(n_show):
        use_thry = 0 if thry.shape[0] == 1 else i
        ax2.plot(ells, 
                 np.abs(treble_ell(ells, thry[use_thry], pred[i], fsky)),
                 lw=0.2,
                 alpha=0.5)
    ax2.plot(ells, 
             np.abs(treble_ell(ells, thry[0], pred[0], fsky)),
             color="tab:blue",
             lw=1)
    ax2.plot(ells, treble_rms(ells, thry, pred, fsky), 
             color="k", 
             label="RMS")

    ax2.set_xlabel(r"Multipole $\ell$")
    ax2.set_ylabel("Relative Error")

    ax1.set_xlim(0, ells[-1])
    ax2.set_xlim(0, ells[-1])
    ax1.set_ylim(0, 7000)
    ax2.set_ylim(0, 4)

    fig.suptitle(f"Predicted Spectrum and Relative Error, {fig_label}", fontsize=14)

    plt.tight_layout()
    fig_handler.write(fig=fig)


def dl2cosm_stdev(ells, dl, fsky):
    if fsky > 1 or fsky<=0:
        raise ValueError("f_sky must be between 0 and 1.")
    sigma = dl * np.sqrt(2 / (2*ells + 1)*fsky)
    return sigma


def treble_ell(ells, thry_dl, pred_dl, fsky):
    norm = np.sqrt((2 * ells + 1) * fsky / 2)
    thry_dl = thry_dl.copy()
    thry_dl[thry_dl==0] = 1
    return norm * (pred_dl - thry_dl) / thry_dl


def treble_rms(ells, trues, preds, fsky):
    if trues.shape == preds.shape:
        pass
    elif trues.shape[0] == 1 and trues.shape[1] == preds.shape[1]:
        pass
    else:
        raise NotImplementedError("Mismatch between shapes of true spectra and predictions")
    n_instances = preds.shape[0]
    trues = trues.copy()
    preds = preds.copy()
    trues[:, :2], preds[:, :2] = 1, 1
    res = np.sqrt((2*ells + 1) * fsky / (2 * n_instances) * np.sum(((preds - trues) / trues)**2, axis=0))
    return res
