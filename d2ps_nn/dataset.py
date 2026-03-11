import os
import logging
from tqdm import tqdm

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
                 pos_enc_lut=None
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        self.pos_enc_lut = torch.as_tensor(pos_enc_lut) if pos_enc_lut is not None else None

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        label = _get_label_idx(path_template=self.label_path_template,
                               handler=self.label_handler,
                               n_fields=self.n_map_fields,
                               sim_idx=sim_idx)

        label_tensor = torch.as_tensor(label)

        features_tensor = _build_feature_tensor(freqs=self.freqs,
                                                path_template=self.feature_path_template,
                                                handler=self.feature_handler,
                                                n_fields=self.n_map_fields,
                                                sim_idx=sim_idx)
        
        if self.pos_enc_lut is not None:
            features_tensor = torch.cat([features_tensor, self.pos_enc_lut], dim=1)

        return features_tensor, label_tensor


def _build_feature_tensor(freqs, path_template, handler, n_fields, sim_idx):
    features = _get_features_idx(
        freqs=freqs,
        path_template=path_template,
        handler=handler,
        n_fields=n_fields,
        sim_idx=sim_idx,
    )

    features_tensor = tuple(torch.as_tensor(f) for f in features)
    features_tensor = torch.stack(features_tensor, dim=0)

    if features_tensor.ndim == 4:  # likely (C,F,H,W) or (F,C,H,W)
        if 12 in features_tensor.shape:
            if features_tensor.shape[0] == 12:
                pass
            elif features_tensor.shape[1] == 12:
                features_tensor = features_tensor.permute(1, 0, 2, 3)
            else:
                raise ValueError(f"Unexpected feature shape {features_tensor.shape}")

    return features_tensor


class PreloadTrainDataset(Dataset):
    """
    For use with pre-processed maps.
    Preloads all labels and features into RAM.
    """
    def __init__(
        self,
        n_sims,
        freqs,
        map_fields: str,
        label_path_template,
        label_handler,
        feature_path_template,
        feature_handler,
        pos_enc_lut=None
    ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields: int = len(map_fields)

        logger.info("Creating label set")
        all_labels = [
            _get_label_idx(
                path_template=self.label_path_template,
                handler=self.label_handler,
                n_fields=self.n_map_fields,
                sim_idx=sim_idx,
            )
            for sim_idx in range(self.n_sims)
        ]
        self.all_labels = np.stack(all_labels, axis=0)

        logger.info("Creating feature set (number in set is 1 more than tqdm says)")
        first_feat = _build_feature_tensor(
            freqs=self.freqs,
            path_template=self.feature_path_template,
            handler=self.feature_handler,
            n_fields=self.n_map_fields,
            sim_idx=0,
        ).numpy()

        all_feats = np.empty((self.n_sims, *first_feat.shape), dtype=np.float32)
        all_feats[0] = first_feat

        for sim_idx in tqdm(range(1, self.n_sims)):
            this_feat = _build_feature_tensor(
                freqs=self.freqs,
                path_template=self.feature_path_template,
                handler=self.feature_handler,
                n_fields=self.n_map_fields,
                sim_idx=sim_idx,
            ).numpy()
            all_feats[sim_idx] = this_feat

        self.pos_enc_lut = torch.as_tensor(pos_enc_lut) if pos_enc_lut is not None else None
        self.all_feats = all_feats

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        xt = torch.from_numpy(self.all_feats[sim_idx])
        yt = torch.from_numpy(self.all_labels[sim_idx])
        if self.pos_enc_lut is not None:
            xt = torch.cat([xt, self.pos_enc_lut], axis=1)
        return xt, yt


class TestDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields,
                 feature_path_template,
                 feature_handler,
                 pos_enc_lut=None
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        self.pos_enc_lut = torch.as_tensor(pos_enc_lut) if pos_enc_lut is not None else None

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features_tensor = _build_feature_tensor(freqs=self.freqs,
                                                path_template=self.feature_path_template,
                                                handler=self.feature_handler,
                                                n_fields=self.n_map_fields,
                                                sim_idx=sim_idx)
        
        if self.pos_enc_lut is not None:
            features_tensor = torch.cat([features_tensor, self.pos_enc_lut], dim=1)

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
