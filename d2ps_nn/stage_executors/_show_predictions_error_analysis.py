import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure, TextPowerSpectrum, NumpyPowerSpectrum, CambPowerSpectrum, PandasCsvHandler
from d2ps_nn.utils.ps_scale import get_ps_scale
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class ShowPredictionsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_predictions_error_analysis")

        self.out_fig: Asset = self.assets_out["fig"]
        out_lut_handler: Figure

        self.in_ps_pred: Asset = self.assets_in["ps_pred"]
        self.in_ps_thry: Asset = self.assets_in["ps_thry"]
        self.in_report: Asset = self.assets_in["report"]
        in_ps_prep_handler: TextPowerSpectrum
        in_ps_thry_handler: CambPowerSpectrum
        in_report_handler: PandasCsvHandler

        self.ells = parse_ells(cfg.model.use_ells)
        self.display_lmax = cfg.lmax
        self.display_n_sims = cfg.n_show_cap

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        try:
            report_data = self.in_report.read()
        except FileNotFoundError:
            logger.warning(f"Report file {self.in_report.path} not found. Skipping report data.")
            report_data = None

        for split in self.splits:
            print(f"Processing split: {split.name}")
            if "F" not in split.name:
                continue  # Skip if no fixed fiducial power spectrum
            with self.name_tracker.set_context("split", split.name):
                logger.info(f"Processing split: {split.name}")
                self.process_split(split, report_data)

    def process_split(self, split, report_data):
        for model_epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", model_epoch):
                self.process_epoch(split, model_epoch, report_data)

    def process_epoch(self, split: Split, epoch, report_data):
        """
        Process a single epoch for a given split to calculate standard deviations.
        """
        #mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
        pred_mean = np.mean(all_pred_ps, axis=0)
        std_dev = np.std(all_pred_ps, axis=0)
        thry_ps = np.array(thry_ps)

        full_ells = np.arange(self.display_lmax)
        #part_ells = self.ells

        logger.info(f"There are {len(self.ells)} ells chosen.")

        fig, ax1 = plt.subplots(1, 1, figsize=(24, 12))
        ax2 = ax1

        ax1.plot(self.ells, thry_ps, 
                    alpha=1, linewidth=0.5,
                    label=f'Theory', 
                    color= 'black',
                    linestyle='--',
                    zorder=10)
                    
        if len(self.ells) < 20:
            ax2.scatter(self.ells, pred_mean, 
                        alpha=0.7, zorder=10,
                        label=f'Prediction Mean',
                        color='blue')
        else:
            ax2.plot(self.ells, pred_mean, 
                    alpha=0.5, linewidth=2, 
                    label=f'Prediction Mean', 
                    color='blue')
        
        ax2.fill_between(self.ells, pred_mean - std_dev, pred_mean + std_dev, 
                 color='red', alpha=0.2, label='$\pm 1 \sigma$')
        ax1.set_xlim([0, self.display_lmax])
        ax1.set_ylim([0, 6500])
        ax1.set_xlabel(r"$\ell$")
        ax1.set_ylabel(r"$D_\ell$")

        ax2.set_xlim([0, self.display_lmax])
        # ax2.set_ylim([0, 1.5])

        ax1.set_title(f"Prediction Mean and Theory, Epoch {epoch}")


        if report_data is not None:
            key = ('chi_square_per_dof')
            mask = (
                (report_data['split'] == split.name) &
                (report_data['epoch'] == epoch)
            )
            subset = report_data.loc[mask]
            if subset.empty:
                raise ValueError(f"No matching rows for split={split.name}, epoch={epoch}")
            else:
                chi_square_per_dof = subset[key].values[0]
            ax1.text(0.9, 0.7, f"$\Delta \chi ^2$: {chi_square_per_dof:.1f}")

        ax1.legend()

        fn = self.out_fig.path
        logger.info(f"Saving figure to {fn}")
        self.out_fig.write()  # Make directory only
        # fn = self.out_fig.path.name  # This may be more convenient
        plt.savefig(fn)
        plt.close(fig)