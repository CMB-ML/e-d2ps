import os
import logging

import numpy as np
from torch.utils.data import Dataset
import torch


logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    """
    For use with pre-processed maps.
    """
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields: str,
                 label_path_template,
                 label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        label = _get_label_idx(path_template=self.label_path_template,
                               handler=self.label_handler,
                               n_fields=self.n_map_fields,
                               sim_idx=sim_idx)

        label_tensor = torch.as_tensor(label)

        features = _get_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.feature_handler,
                                     n_fields=self.n_map_fields,
                                     sim_idx=sim_idx)

        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.stack(features_tensor, dim=0)

        return features_tensor, label_tensor


class TestDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features = _get_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.feature_handler,
                                     n_fields=self.n_map_fields,
                                     sim_idx=sim_idx)

        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.stack(features_tensor, dim=0)

        return features_tensor, sim_idx


def _get_label_idx(path_template, handler, n_fields, sim_idx):
    # TODO: Implement multiple fields
    if n_fields > 1:
        raise NotImplementedError("This function only supports one field at a time.")

    label_path = path_template.format(sim_idx=sim_idx)
    label = handler.read(label_path)
    # label = label[0]
    return label


def _get_features_idx(freqs, path_template, handler, n_fields, sim_idx):
    # TODO: Implement multiple fields
    if n_fields > 1:
        raise NotImplementedError("This function only supports one field at a time.")

    features = []
    for freq in freqs:
        feature_path = path_template.format(sim_idx=sim_idx, freq=freq)
        feature_data = handler.read(feature_path)
        feature_data = feature_data[0]  # TODO: Implement multiple fields
        features.append(feature_data)
    return features
