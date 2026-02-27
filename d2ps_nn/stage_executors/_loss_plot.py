import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 
from cmbml.analysis.stage_executors._16_loss_plot import make_loss_plot

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
                                 ylim=(1e3,1e7),
                                 sm_sigma2=20)
        self.out_fig.write(fig=fig)
