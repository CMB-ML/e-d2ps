import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from d2ps_nn.dataset import TrainDataset, PreloadTrainDataset
from d2ps_nn.encdec import SimpleEncDec
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset
import numpy as np


logger = logging.getLogger(__name__)


class TrainingExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        self.in_trgt_asset: Asset = self.assets_in["cmb_ps"]
        self.in_feat_asset = FeatureAsset.select(self.assets_in, 
                                                 cfg.features_type, 
                                                 self.instrument)
        self.in_model: Asset = self.assets_in["model"]
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.dtype         = self.dtype_mapping[cfg.model.dtype]  # TODO: Ensure this is used

        self.choose_device(cfg.model.train.device)  # See parent class
        self.batch_size    = cfg.model.train.batch_size
        self.num_workers   = cfg.model.train.num_loader_workers
        self.learning_rate = cfg.model.train.learning_rate
        self.n_epochs      = cfg.model.train.n_epochs
        self.restart_epoch = cfg.model.train.restart_epoch
        self.checkpoint    = cfg.model.train.checkpoint_every
        self.extra_check   = cfg.model.train.extra_check

        self.features_type = cfg.features_type

        self.norm          = None
        # self.lmax          = cfg.model.lmax
        self.ells          = parse_ells(cfg.model.use_ells)
        self.loss_key      = cfg.model.train.loss

        # When using Wishart loss, we need to skip the monopole and dipole
        # self.skip_mono_dipole = cfg.model.train.loss == "wishart"
        self.final_activation = cfg.model.final_activation
        self.use_batch_norm_enc = cfg.model.use_batch_norm_enc
        self.use_batch_norm_dec = cfg.model.use_batch_norm_dec

        self.dropout_rate = cfg.model.dropout_rate
        self.activation = cfg.model.activation
        if self.activation is None:
            raise ValueError("Activation cannot be None. Please specify 'relu' or gelu' in config.")
        self.pooling = cfg.model.pooling
        if self.pooling is None:
            raise ValueError("Please specify 'avg' or 'max'.")
        self.model_n_init_features = cfg.model.n_init_features
        self.model_n_levels = cfg.model.n_levels
        self.model_note = cfg.model.note
        self.earliest_best = cfg.model.train.earliest_best
        self.features_type = cfg.features_type
        self.use_preload = cfg.model.train.get("use_preload", False)
        logger.info(f"{'U' if self.use_preload else 'Not u'}sing dataset with preload. This eats RAM! BE CAREFUL!")
        self.freeze_bn = cfg.freeze_bn

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        # When using Wishart loss, we need to skip the monopole and dipole
        # use_lmax = self.lmax - 2 if self.skip_mono_dipole else self.lmax

        # if self.final_activation == "softplus":
        #     final_activation = torch.nn.Softplus()
        # elif not self.final_activation:
        #     final_activation = None
        # else:
        #     raise ValueError(f"Unknown final activation: {self.final_activation}")

        logger.info(f"Creating model")
        n_channels = len(self.instrument.dets.keys()) if self.features_type == "obs" else 1

        model = SimpleEncDec(
                            #  l_max=use_lmax,
                             n_ells=1279,
                             n_in_channels=1,
                             final_activation=None,
                             n_init_features=16,
                             n_levels=7,
                             use_batch_norm_enc=False,
                             use_batch_norm_dec=False,
                             activation="relu",
                             dropout_rate=0,
                             pooling="max",
                             note=""
                             )

        model = model.to(self.device)
        # logger.info(f"Model created!")

        if self.freeze_bn:
            model.apply(freeze_bn)
            logger.info("Batch Norm Frozen")

        # Potentially confusing... the order is determined by the order of
        #   splits for this stage in the pipeline yaml.
        train_split = self.splits.get("train")
        # valid_split = self.splits.get("valid")

        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
            )

        # valid_dataset = self.set_up_dataset(valid_split)
        # valid_dataloader = DataLoader(
        #     valid_dataset, 
        #     batch_size=self.batch_size, 
        #     shuffle=False,
        #     num_workers=self.num_workers
        #     )

        # logger.info(f"Training on {len(train_dataloader)} and Validation on {len(valid_dataloader)} batches, batch size {self.batch_size}.")

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)

        burn_in_steps = 50
        burn_in_lr    = 1e-9
        ramp_start_lr = self.learning_rate
        ramp_end_lr   = ramp_start_lr * 1e3
        total_steps   = len(train_dataloader) * self.n_epochs
        ramp_steps    = total_steps - burn_in_steps

        optimizer = torch.optim.Adam(model.parameters(), lr=burn_in_lr)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 
                                                        factor=1.0, 
                                                        total_iters=burn_in_steps)

        logger.info(f"Starting new model.")
        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")
        start_epoch = 0
        loss_record_headers = ['Epoch', 'Training Loss', 'Learning Rate']
        self.out_loss_record.write(data=loss_record_headers)

        n_epoch_digits = len(str(self.n_epochs))
        step_counter = 0

        for epoch in range(start_epoch, self.n_epochs):
            with tqdm(train_dataloader, desc=f"Ep {epoch + 1:<{n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
                model.train()
                for train_features, train_label in pbar:
                    if step_counter == burn_in_steps:
                        optimizer = torch.optim.Adam(model.parameters(), 
                                                    lr=ramp_start_lr)
                        gamma = (ramp_end_lr / ramp_start_lr) ** (1.0 / (ramp_steps))
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            optimizer,
                            gamma=gamma
                        )
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)
                    optimizer.zero_grad()
                    output = model(train_features)
                    batch_train_loss = loss_function(output, train_label)
                    batch_train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    step_counter += 1
                    pbar.set_postfix({'Loss': batch_train_loss.item(), 'LR': optimizer.param_groups[0]["lr"]})
                    lr = optimizer.param_groups[0]["lr"]
                    self.out_loss_record.append([step_counter, batch_train_loss.item(), lr])  # , train_loss])

    def set_up_dataset(self, split: Split) -> None:
        # TODO: Unsure if this change is needed
        # label_path_template = self.make_fn_template(split, self.in_ps_asset)
        # feature_path_template = self.make_fn_template(split, self.in_map_asset)
        trgt_path_template = self.make_fn_template(split, self.in_trgt_asset)
        feat_path_template = self.make_fn_template(split, self.in_feat_asset)

        freqs = self.instrument.dets.keys() if self.features_type == "obs" else [None]

        dataset_class = PreloadTrainDataset if self.use_preload else TrainDataset

        dataset = dataset_class(
            n_sims = split.n_sims,
            freqs = freqs,
            map_fields=self.map_fields,
            label_path_template=trgt_path_template,
            # label_path_template=label_path_template,
            label_handler=self.in_trgt_asset.handler,
            feature_path_template=feat_path_template,
            # feature_path_template=feature_path_template,
            feature_handler=self.in_feat_asset.handler
            )
        return dataset


class SingleDWishartLoss(torch.nn.Module):
    def __init__(self, ells):
        super().__init__()

        ells = torch.tensor(ells, dtype=torch.float32)
        # ells = torch.arange(0, lmax + 1, dtype=torch.float32)
        # ells = ells[2:]
        ratio = (2 * ells + 1) / (2 * ells - 1)
        self.register_buffer("ratio", ratio)

    def forward(self, output, target):
        t1 = output / target
        t2 = torch.log(target)
        t3 = torch.log(output)

        per_ell = t1 + t2 - self.ratio * t3

        return torch.sum(per_ell)

def get_loss(loss_key, ells):
    if loss_key == "l1":
        loss_function = torch.nn.L1Loss(reduction='mean')
    elif loss_key == "l2":
        loss_function = torch.nn.MSELoss(reduction='mean')
    elif loss_key == "wishart":
        loss_function = SingleDWishartLoss(ells=ells)
    else:
        raise ValueError(f"Unknown loss function: {loss_key}")
    return loss_function


def freeze_bn(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)
