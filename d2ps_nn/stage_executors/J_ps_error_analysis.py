import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, CambPowerSpectrum, PandasCsvHandler
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class ErrorAnalysisExecutor(BaseStageExecutor):
    """
    Goal: Calculate various error metrics for power spectra predictions.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="error_analysis")

        self.out_report: Asset = self.assets_out["report"]
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
        report = []
        for split in self.splits:
            print(f"Processing split: {split.name}")
            if "F" not in split.name:
                continue  # Skip if no fixed fiducial power spectrum
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split: {split.name}")
                self.process_split(split, report)
        report = pd.DataFrame(report)
        self.out_report.write(data=report)
            
    def process_split(self, split, report):
        for model_epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", model_epoch):
                self.process_epoch(split, model_epoch, report)

    def process_epoch(self, split: Split, epoch, report) -> pd.DataFrame:
        """
        Process a single epoch for a given split to calculate standard deviations.
        """
        logger.info(f"Processing epoch {epoch} for split {split.name}")
        use_alt_path = True if split.ps_fidu_fixed else False
        # Prepare to collect results
        all_pred_ps = []
        thry_ps = self.in_ps_thry.read(use_alt_path=use_alt_path)
        thry_ps = thry_ps[self.ells]  # Ensure we only use the relevant ell range

        # Iterate through simulations in the split
        for sim in tqdm(split.iter_sims(), desc=f"Processing simulations for epoch {epoch}"):
            with self.name_tracker.set_context("sim_num", sim):
                pred_ps = self.in_ps_pred.read()
                all_pred_ps.append(pred_ps)
                
        all_pred_ps = np.array(all_pred_ps)
        thry_ps = np.array(thry_ps)

        # Calculate statistics
        std_dev = np.std(all_pred_ps, axis=0)
        pred_mean = np.mean(all_pred_ps, axis=0)
        chi_square = np.sum(((thry_ps - pred_mean) ** 2) / std_dev**2)
        # dof = number of Cls - number of cosmological parameters
        chi_square_per_dof = chi_square / (len(thry_ps) - self.num_cosmo_params)

        # Collect results
        report.append({
            "split": split.name,
            "epoch": epoch,
            # TODO: Save mean, std_dev, power_spectra, etc.?
            #"std_dev": std_dev,
            #"mean": mean,
            "chi_square_per_dof": chi_square_per_dof,
        })