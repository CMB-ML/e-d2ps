import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from d2ps_fcn.scaler import ArrayScaler


logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
        self,
        n_sims,
        label_path,
        label_handler,
        feature_path,
        feature_handler,
        feature_scaler: Optional[ArrayScaler] = None,
        label_scaler: Optional[ArrayScaler] = None,
        dtype=torch.float32,
        fit_scalers: bool = True,
        device: Optional[torch.device] = None
    ):
        self.n_sims = n_sims
        self.feature_scaler = feature_scaler or ArrayScaler()
        self.label_scaler = label_scaler or ArrayScaler()

        labels = label_handler.read(label_path)
        feats = feature_handler.read(feature_path)

        if n_sims is not None:
            labels = labels[:n_sims]
            feats = feats[:n_sims]

        if fit_scalers:
            self.feature_scaler.fit(feats)
            self.label_scaler.fit(labels)

        feats = self.feature_scaler.transform(feats)
        labels = self.label_scaler.transform(labels)

        self.all_feats = torch.as_tensor(feats, dtype=dtype, device=device)
        self.all_labels = torch.as_tensor(labels, dtype=dtype, device=device)

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        return self.all_feats[sim_idx], self.all_labels[sim_idx]
