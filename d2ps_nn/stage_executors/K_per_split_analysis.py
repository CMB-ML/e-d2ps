import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from itertools import product

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import (
    TextPowerSpectrum, 
    CambPowerSpectrum, 
    PandasCsvHandler, 
    NumpyPowerSpectrum
)
from cmbml.utils.cosmic_variance import get_var_per_ell
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class PerSplitPerEllAnalysis(BaseStageExecutor):
    """
    Goal: Calculate various error metrics for power spectra predictions.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="per_split_per_ell_analysis")

        self.out_per_ell_vector: Asset = self.assets_out["report"]
        out_report_handler: NumpyPowerSpectrum
        self.over_all_ell_scalars: Asset = self.assets_out["report"]
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
        self.cosmic_var = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            print(f"Processing split: {split.name}")
            if not split.ps_fidu_fixed:
                continue  # Skip if no fixed fiducial power spectrum
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split: {split.name}")
                self.process_split(split)

    def get_cosmic_var(self) -> None:
        # Cosmic variance is a relative measure of variance, reporting 
        #   Delta Cl / Cl (or Delta Dl / Dl)
        self.cosmic_var = get_var_per_ell(self.ells)

    def process_split(self, split):
        if split.ps_fidu_fixed:
            thry_ps = self.in_ps_thry.read(use_alt_path=True)
            thry_ps = np.array(thry_ps)
        else:
            thry_ps = []
            for sim in split.iter_sim():
                with self.name_tracker.set_context("sim_num", sim):
                    one_thry_ps = self.in_ps_thry.read()
                thry_ps.append(one_thry_ps)
            thry_ps = np.array(thry_ps)
        thry_ps = thry_ps[self.ells]

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

        # Calculate statistics (all vectors--per ell)
        pred_mean       = np.mean(pred_ps, axis=0)
        pred_sd         = np.std(pred_ps, axis=0)
        cosmic_var      = self.cosmic_var              # Ratio: (Delta D) / D

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
        ells            = self.ells
        weights         = 2 * ells + 1

        metrics = {
            "absolute_error":        abs_err,
            "relative_error":        rel_err,
            "cosmic_relative_error": cv_err,
            "z_abs_relative_error":  z_abs_err,
            "z_sign_relative_error": z_sign_err,
            "ensemble_chi_square":   chi2_ens
        }

        res_v = res_vector = {}
        res_s = res_scalar = {}
        for m_label, m in metrics.items():
            # Project over all simulations; this indicates performance per-ell
            m_sims = np.mean(m, axis=0)
            res_s[(m_label, "per_ell")] = accumulate_over(m_sims)
            res_v[(m_label, "per_ell")] = m_sims

            # Project over all simulations, account for weighting at each ell
            # m_sims_w = np.average()  #????

            # Project over all ells first
            m_ells = np.mean(m, axis=1)
            res_s[(m_label, "per_sim", "L1", "nw")] = accumulate_over(m_ells)
            
            # Ells first, weight by modes
            m_ells = np.average(m, axis=1, weights=weights)
            res_s[(m_label, "per_sim", "L1", "wt")] = accumulate_over(m_ells)

            # Ells first, highlight trouble using L2
            m_ells = np.mean(m ** 2, axis=1)
            res_s[(m_label, "per_sim", "L2", "nw")]["mean"] = np.mean(np.sqrt(m_ells))

            # Ells first, highlight trouble using L2, weight by modes
            m_ells = np.average(m ** 2, axis=1, weights=weights)
            res_s[(m_label, "per_sim", "L2", "nw")]["mean"] = np.mean(np.sqrt(m_ells))

        df = pd.DataFrame(res_s)
        self.over_all_ell_scalars.write(df)

        self.out_per_ell_vector.write(res_v)
        # self.out_per_ell_vector: Asset = self.assets_out["report"]
        # out_report_handler: NumpyPowerSpectrum
        # self.over_all_ell_scalars: Asset = self.assets_out["report"]
        # out_report_handler: PandasCsvHandler


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
