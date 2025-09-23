import sys
import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from itertools import product
import yaml
import csv

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import (
    TextPowerSpectrum, 
    CambPowerSpectrum, 
    PandasCsvHandler, 
    # NumpyPowerSpectrum
)
from cmbml.utils.cosmic_variance import get_var_per_ell
from cmbml.utils.dl_norm import get_cl2dl_norm_per_ell
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class PerSplitAnalysis(BaseStageExecutor):
    """
    Goal: Calculate various error metrics for power spectra predictions.
    """
    def __init__(self, cfg: DictConfig, stage_str="per_split_analysis") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_vectors: Asset = self.assets_out["per_vector_report"]
        out_report_handler: TextPowerSpectrum
        self.out_scalars: Asset = self.assets_out["per_scalar_report"]
        out_report_handler: PandasCsvHandler

        self.in_ps_pred: Asset = self.assets_in["ps_pred"]
        self.in_ps_thry: Asset = self.assets_in["ps_thry"]
        in_ps_pred_handler: TextPowerSpectrum
        in_ps_thry_handler: CambPowerSpectrum

        self.ells = parse_ells(cfg.model.use_ells)
        self.num_cosmo_params = cfg.model.analysis.metrics.num_cosmo_params
        for split in self.splits:
            num_samples = split.n_sims
            if num_samples - self.num_cosmo_params <= 0:
                raise ValueError(f"Number of simulations ({num_samples}) must be greater than number of cosmological parameters ({self.num_cosmo_params}) for valid chi-squared calculation.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            print(f"Processing split: {split.name}")
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split: {split.name}")
                self.process_split(split)

    def process_split(self, split):
        if split.ps_fidu_fixed:
            thry_ps = self.in_ps_thry.read(use_alt_path=True)
            thry_ps = np.array(thry_ps)
            thry_ps = thry_ps[self.ells]
        else:
            thry_ps = []
            for sim in split.iter_sims():
                with self.name_tracker.set_context("sim_num", sim):
                    one_thry_ps = self.in_ps_thry.read(use_alt_path=False)
                thry_ps.append(one_thry_ps)
            thry_ps = np.array(thry_ps)
            thry_ps = thry_ps[:, self.ells]

        for model_epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", model_epoch):
                self.process_epoch(split, model_epoch, thry_ps)

    def process_epoch(self, split: Split, epoch, thry_ps) -> pd.DataFrame:
        """
        Process a single epoch for a given split to calculate standard deviations.
        """
        logger.info(f"Processing epoch {epoch} for split {split.name}")

        # Iterate through predictions for this epoch in the split
        pred_ps = []
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                one_pred_ps = self.in_ps_pred.read()
                pred_ps.append(one_pred_ps)
        pred_ps = np.array(pred_ps)

        # TODO: Fix this. Patch exists to handle Realization Power Spectra,
        #       which are produced using a different way of defining ells
        #       (thye use lmax; this later method uses use_ells)
        if pred_ps.shape[1] == 1281:
            pred_ps = pred_ps[:, 2:]

        metrics = get_split_analysis(ells=self.ells, 
                                     thry_ps=thry_ps, 
                                     pred_ps=pred_ps)

        for metric in metrics['vectors'].keys():
            with self.name_tracker.set_context("metric", metric):
                self.out_vectors.write(data=metrics['vectors'][metric])

        df = pd.DataFrame(metrics['scalars']).T
        df = df.reset_index()
        df.columns = ['metric','proj','norm','weighting'] + df.columns[4:].tolist()
        self.out_scalars.write(data=df)


def get_split_analysis(ells, thry_ps, pred_ps):
        res_v = res_vector = {}
        res_s = res_scalar = {}
        fixed_fidu = len(thry_ps.shape) == 1

        cosmic_var = get_var_per_ell(ells)

        # norm = get_cl2dl_norm_per_ell(ells)
        # thry_ps = thry_ps / norm
        # pred_ps = pred_ps / norm

        # Calculate statistics (all vectors--per ell)
        pred_mean       = np.mean(pred_ps, axis=0)
        pred_sd         = np.std(pred_ps, axis=0)
        cosmic_var      = cosmic_var              # Ratio: (Delta D) / D

        # Ensemble chi-square 
        # Reference vs. mean prediction, normalized by prediction scatter
        # Single Thry:  "Is the mean of the prediction ensemble consistent with the fixed
        #                reference, given the ensemble’s scatter?"
        # Many Thry:    Doesn't really apply cleanly (since there isn't one reference to
        #                compare to the mean across sims).
        chi2_ens        = ((thry_ps - pred_mean)**2) / (pred_sd**2)

        # For each instance
        # Residual is the difference between prediction and target
        residual        = pred_ps - thry_ps

        # Absolute error 
        # How far is a prediction from the target?
        abs_err         = np.abs(residual)

        # Relative error
        # How large is the fractional error of predictions compared to the target?
        rel_err         = abs_err / np.abs(thry_ps)

        # Cosmic-variance–normalized error
        # Predictions normalized by (an estimate of) the *data* variance
        cv_err          = abs_err / (cosmic_var * thry_ps)

        # Signed Z-score error 
        # Predictions normalized by the *empirical* variance, above or below the target
        z_sign_err      = residual / pred_sd

        # Absolute Z-score error
        # Predictions normalized by the *empirical* variance, distance from the target
        z_abs_err       = abs_err / pred_sd

        ############################
        # Single scalar scores
        ells            = ells
        weights         = 2 * ells + 1

        metrics = {
            "absolute_error"       : abs_err,
            "relative_error"       : rel_err,
            "cosmic_relative_error": cv_err,
            "z_abs_relative_error" : z_abs_err,
            "z_sign_relative_error": z_sign_err,
            "residual"             : residual
        }
        if not fixed_fidu:
            metrics["ensemble_chi_square"] = chi2_ens
            res_v["thry_mean"] = np.mean(thry_ps, axis=0)
            res_v["thry_std"] = np.std(thry_ps, axis=0)

        nw = "uniform"
        yw = "alm"
        l1 = "mean"
        l2 = "rms"

        for m_label, m in metrics.items():
            # Project over all simulations; this indicates performance per-ell
            m_sims = np.mean(m, axis=0)
            res_v[m_label] = m_sims
            res_s[(m_label, "per_ell", l1, nw)] = accumulate_over(m_sims)
            res_s[(m_label, "per_ell", l1, yw)] = {"mean": np.average(m_sims, weights=weights)}

            # Ells first, no weighting
            m_ells = np.mean(m, axis=1)
            res_s[(m_label, "per_sim", l1, nw)] = accumulate_over(m_ells)
            
            # Ells first, weight by modes
            m_ells = np.average(m, axis=1, weights=weights)
            res_s[(m_label, "per_sim", l1, yw)] = accumulate_over(m_ells)

            # Ells first, highlight trouble using L2
            m_ells = np.sqrt(np.mean(m ** 2, axis=1))
            res_s[(m_label, "per_sim", l2, nw)] = accumulate_over(m_ells)

            # Ells first, highlight trouble using L2, weight by modes
            m_ells = np.sqrt(np.average(m ** 2, axis=1, weights=weights))
            res_s[(m_label, "per_sim", l2, yw)] = accumulate_over(m_ells)

        if fixed_fidu:
            chi2_lbl = 'ensemble_chi_square'
            res_v[chi2_lbl] = chi2_ens
            res_s[(chi2_lbl, "per_ell", l1, nw)] = accumulate_over(chi2_ens)
            res_s[(chi2_lbl, "per_ell", l1, nw)] = {'mean': np.average(chi2_ens, weights=weights)}
            res_v["thry_mean"] = thry_ps

        res_v['pred_mean'] = pred_mean
        res_v['pred_sd']   = pred_sd
        return {'vectors': res_v, 'scalars': res_s}


def accumulate_over(arr):
    """Compute summary stats over a 1D array."""
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25":    float(np.quantile(arr, 0.25)),
        "q75":    float(np.quantile(arr, 0.75)),
        "max":    float(np.max(arr)),
        "min":    float(np.min(arr)),
    }


class PerSplitSurrogateAnalysis(PerSplitAnalysis):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="per_split_surrogate_analysis")
