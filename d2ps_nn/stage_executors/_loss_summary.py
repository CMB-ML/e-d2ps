import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import TextHandler 

class LossSummaryExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="loss_summary")
        self.out_summary: TextHandler = self.assets_out["summary"]
        self.in_loss_csv = self.assets_in["loss_record"]

    def execute(self):
        df = pd.read_csv(self.in_loss_csv.path)
        #skipping the first few rows if they are not needed
        # df = df[df["Epoch"] >= self.skip_n_values]

        best_idx = df["Validation Loss"].idxmin()
        best_val_loss = df.loc[best_idx, "Validation Loss"]
        best_epoch = df.loc[best_idx, "Epoch"]
        
        best_train_idx = df["Training Loss"].idxmin()
        best_train_loss = df.loc[best_train_idx, "Training Loss"]

        row_1 = f"Iteration\tBest Valid Loss\tBest Valid Epoch\tBest Train Loss"
        row_2 = f"\t" \
                f"{best_val_loss:.2e}\t" \
                f"{best_epoch}\t" \
                f"{best_train_loss:.2e}\t"
        res = f"{row_1}\n{row_2}"
        self.out_summary.write(data=res)

def exp_as_ltx(some_val):
    val_sci = "{:.1e}".format(some_val)
    base, exp = val_sci.split("e")
    exp = int(exp)
    return f"{base} \\times 10^{{{exp}}}" if exp != 0 else base
