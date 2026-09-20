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
                 px_arrange=None,
                 pos_enc_lut=None,
                 mask=None
                 ):
        self.feature_assembler, self.cat_axis = get_feature_assembler(px_arrange)
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        self.pos_enc_lut = torch.as_tensor(pos_enc_lut) if pos_enc_lut is not None else None
        self.mask = torch.as_tensor(mask) if mask is not None else None

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        label = _get_label_idx(path_template=self.label_path_template,
                               handler=self.label_handler,
                               n_fields=self.n_map_fields,
                               sim_idx=sim_idx)

        label_tensor = torch.as_tensor(label)

        features_tensor = self.feature_assembler(freqs=self.freqs,
                                                 path_template=self.feature_path_template,
                                                 handler=self.feature_handler,
                                                 n_fields=self.n_map_fields,
                                                 sim_idx=sim_idx)

        if self.mask is not None:
            features_tensor = self.mask * features_tensor
        
        if self.pos_enc_lut is not None:
            features_tensor = torch.cat([features_tensor, self.pos_enc_lut], dim=self.cat_axis)

        return features_tensor, label_tensor


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
        px_arrange=None,
        pos_enc_lut=None
    ):
        raise NotImplementedError("""
1) PreloadTrainDataset has a silent bug. first_feat uses self.feature_assembler, but the loop body calls _build_feature_tensor_channel directly. With px_arrange="cmbNNCS" and 4D data, sim 0 and sims 1..N get different treatment. Should be self.feature_assembler.
2) The 12 in shape check is ambiguous if you ever hit 12 frequencies. Unlikely with Planck's 9, but a named axis or explicit config would be safer.
""")

        self.feature_assembler, self.cat_axis = get_feature_assembler(px_arrange)
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
        first_feat = self.feature_assembler(
            freqs=self.freqs,
            path_template=self.feature_path_template,
            handler=self.feature_handler,
            n_fields=self.n_map_fields,
            sim_idx=0,
        ).numpy()

        all_feats = np.empty((self.n_sims, *first_feat.shape), dtype=np.float32)
        all_feats[0] = first_feat

        for sim_idx in tqdm(range(1, self.n_sims)):
            this_feat = _build_feature_tensor_channel(
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
            xt = torch.cat([xt, self.pos_enc_lut], axis=self.cat_axis)
        return xt, yt


class TestDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields,
                 feature_path_template,
                 feature_handler,
                 px_arrange=None,
                 pos_enc_lut=None,
                 mask=None
                 ):
        self.feature_assembler, self.cat_axis = get_feature_assembler(px_arrange)
        self.n_sims = n_sims
        self.freqs = freqs
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        self.pos_enc_lut = torch.as_tensor(pos_enc_lut) if pos_enc_lut is not None else None
        self.mask = torch.as_tensor(mask) if mask is not None else None

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features_tensor = self.feature_assembler(freqs=self.freqs,
                                                 path_template=self.feature_path_template,
                                                 handler=self.feature_handler,
                                                 n_fields=self.n_map_fields,
                                                 sim_idx=sim_idx)

        if self.mask is not None:
            features_tensor = self.mask * features_tensor

        if self.pos_enc_lut is not None:
            features_tensor = torch.cat([features_tensor, self.pos_enc_lut], dim=self.cat_axis)

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


def _stack_channel(arrays):
    t = torch.stack([torch.as_tensor(a) for a in arrays], dim=0)
    if t.ndim == 4 and 12 in t.shape:  # likely (C,F,H,W) or (F,C,H,W)
        if t.shape[0] == 12:
            pass
        elif t.shape[1] == 12:
            t = t.permute(1, 0, 2, 3)
        else:
            raise ValueError(f"Unexpected feature shape {t.shape}")
    return t


def _stack_cmbNNCS(arrays):
    return torch.stack([torch.as_tensor(a) for a in arrays], dim=0)


def get_stacker(px_arrange):
    if px_arrange == "channel":
        return _stack_channel, 1
    if px_arrange == "cmbNNCS":
        return _stack_cmbNNCS, 0
    raise ValueError("Pixel arrangement must be set. 'channel' and 'cmbNNCS' are accepted")


def get_feature_assembler(px_arrange):
    stacker, concat_axis = get_stacker(px_arrange)

    def assemble(freqs, path_template, handler, n_fields, sim_idx):
        features = _get_features_idx(freqs=freqs, path_template=path_template,
                                     handler=handler, n_fields=n_fields, sim_idx=sim_idx)
        return stacker(features)

    return assemble, concat_axis


def stack_masks(masks, px_arrange):
    """Stack per-channel arranged masks exactly as features are stacked."""
    stacker, _ = get_stacker(px_arrange)
    return stacker(masks)


def summarize_params(params: dict) -> dict:
    """Readable summary of dataset parameters for logging."""
    out = {}
    for k, v in params.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif hasattr(v, "shape"):
            out[k] = f"<{type(v).__name__} shape {tuple(v.shape)}>"
        elif isinstance(v, (list, tuple, type({}.keys()))):
            out[k] = list(v)
        else:
            out[k] = f"<{type(v).__name__}>"
    return out
