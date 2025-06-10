import matplotlib.pyplot as plt
import pandas as pd
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 

class LossPlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="loss_plot")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_loss_csv = self.assets_in["loss_record"]

    def execute(self):
        df = pd.read_csv(self.in_loss_csv.path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["Epoch"], df["Training Loss"], label="Training Loss")
        ax.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        ax.grid(True)
        self.out_fig.write()
        plt.savefig(self.out_fig.path)
        plt.close()
