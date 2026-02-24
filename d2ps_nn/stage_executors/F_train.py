import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor
from d2ps_nn.dataset import TrainDataset
from d2ps_nn.encdec import SimpleEncDec
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset


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
        self.in_mean_ps: Asset = self.assets_in["mean_ps"]
        self.in_stdv_ps: Asset = self.assets_in["stdv_ps"]
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

        self.norm          = None
        # self.lmax          = cfg.model.lmax
        self.ells          = parse_ells(cfg.model.use_ells)
        self.loss_key      = cfg.model.train.loss

        # When using Wishart loss, we need to skip the monopole and dipole
        # self.skip_mono_dipole = cfg.model.train.loss == "wishart"
        self.final_activation = cfg.model.final_activation
        self.norm_kind_enc = "gn"
        self.norm_kind_dec = None

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

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        # When using Wishart loss, we need to skip the monopole and dipole
        # use_lmax = self.lmax - 2 if self.skip_mono_dipole else self.lmax

        if self.final_activation == "softplus":
            final_activation = torch.nn.Softplus()
        elif not self.final_activation:
            final_activation = None
        else:
            raise ValueError(f"Unknown final activation: {self.final_activation}")

        mean_ps = self.in_mean_ps.read()
        mean_ps = torch.tensor(mean_ps, device=self.device, dtype=self.dtype)
        stdv_ps = self.in_stdv_ps.read()
        stdv_ps = torch.tensor(stdv_ps, device=self.device, dtype=self.dtype)

        logger.info(f"Creating model")
        model = SimpleEncDec(
                            #  l_max=use_lmax,
                             n_ells=len(self.ells),
                             n_in_channels=len(self.instrument.dets),
                             final_activation=final_activation,
                             n_init_features=self.model_n_init_features,
                             n_levels=self.model_n_levels,
                             norm_kind_enc=self.norm_kind_enc,
                             norm_kind_dec=self.norm_kind_dec,
                             activation=self.activation,
                             dropout_rate=self.dropout_rate,
                             pooling=self.pooling,
                             note=self.model_note
                             )

        model = model.to(self.device)
        logger.info(f"Model created!")

        # Potentially confusing... the order is determined by the order of
        #   splits for this stage in the pipeline yaml.
        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
            )

        valid_dataset = self.set_up_dataset(valid_split)
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
            )

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                start_epoch = self.in_model.read(model=model, 
                                                 epoch=self.restart_epoch, 
                                                 optimizer=optimizer, 
                                                #  scheduler=scheduler
                                                 )
            if start_epoch == "init":
                start_epoch = 0
            self.out_loss_record.start()  # New lines will be added, naively
                                          #  Naively = appended; if resuming, manual cleaning may be needed.
        else:
            logger.info(f"Starting new model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
            start_epoch = 0
            loss_record_headers = ['Epoch', 
                                   'Batch Loss', 
                                   'Validation Loss', 
                                   'Training Loss', 
                                   'Valid Loss Unstd',
                                   'Train Loss Unstd',
                                   'Valid Loss Baseline',
                                   'Train Loss Baseline',
                                   ]
            self.out_loss_record.write(data=loss_record_headers)

        n_epoch_digits = len(str(self.n_epochs))

        # all_batch_loss = []
        # all_train_loss = []
        all_valid_loss = []
        # all_train_loss_unstd = []
        # all_valid_loss_unstd = []
        # all_train_loss_base = []
        # all_valid_loss_base = []

        # step = 0

        for epoch in range(start_epoch, self.n_epochs):
            # Training
            with tqdm(train_dataloader, desc=f"Ep {epoch + 1:<{n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
                model.train()
                # model.apply(freeze_bn)
                batch_loss = 0
                for train_features, train_label in pbar:
                # for train_features, train_label in train_dataloader:
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)
                    optimizer.zero_grad()
                    output = model(train_features)
                    batch_train_loss = loss_function(output, train_label)
                    batch_train_loss.backward()
                    # with torch.no_grad():
                    #     # grad norm
                    #     g2 = 0.0
                    #     badg = 0
                    #     for p in model.parameters():
                    #         if p.grad is None:
                    #             continue
                    #         if not torch.isfinite(p.grad).all():
                    #             badg += 1
                    #         g2 += p.grad.norm().item()**2
                    #     grad_norm = g2**0.5

                    #     # param norm + nonfinite params
                    #     w2 = 0.0
                    #     badw = 0
                    #     for p in model.parameters():
                    #         if not torch.isfinite(p).all():
                    #             badw += 1
                    #         w2 += p.norm().item()**2
                    #     weight_norm = w2**0.5
                    #     delta = (output[0]-output[1]).abs().max().item()
                    #     torch_same = torch.equal(output[0], output[1])
                    # step +=1
                    # print(f"iter {step: 2d} loss {batch_train_loss.item():.4e} "
                    #       f"grad_norm {grad_norm:.4e} weight_norm {weight_norm:.4e} "
                    #       f"badg {badg} badw {badw} max output delta {delta:.4e} torch same {torch_same}")
                    
                    optimizer.step()
                    pbar.set_postfix({'Loss': batch_train_loss.item()})
                    batch_loss += batch_train_loss.item()
                batch_loss /= len(train_dataloader)
                # all_batch_loss.append(batch_loss)

            # Validation
            model.eval()
            train_loss = 0
            valid_loss = 0
            train_loss_unstd = 0
            valid_loss_unstd = 0
            train_loss0 = 0
            valid_loss0 = 0
            with torch.no_grad():
                for train_features, train_label in tqdm(train_dataloader, leave=False):
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)
                    baseline = torch.zeros_like(train_label)
                    train_loss0 += loss_function(baseline, train_label).item()
                    output = model(train_features)
                    train_loss += loss_function(output, train_label).item()
                    output_unstd = unstandardize(output, mean_ps, stdv_ps)
                    label_unstd = unstandardize(train_label, mean_ps, stdv_ps)
                    train_loss_unstd += loss_function(output_unstd, label_unstd).item()
                train_loss /= len(train_dataloader)
                train_loss0 /= len(train_dataloader)
                train_loss_unstd /= len(train_dataloader)
                logger.info(f"Epoch {epoch + 1:<{n_epoch_digits}} Training loss: {train_loss:.02e}")

                for valid_features, valid_label in tqdm(valid_dataloader, leave=False):
                    valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                    valid_label = valid_label.to(device=self.device, dtype=self.dtype)
                    baseline = torch.zeros_like(valid_label)
                    valid_loss0 += loss_function(baseline, valid_label).item()
                    output = model(valid_features)
                    output_unstd = unstandardize(output, mean_ps, stdv_ps)
                    label_unstd = unstandardize(valid_label, mean_ps, stdv_ps)
                    valid_loss += loss_function(output, valid_label).item()
                    valid_loss_unstd += loss_function(output_unstd, label_unstd).item()
                valid_loss /= len(valid_dataloader)
                valid_loss0 /= len(valid_dataloader)
                valid_loss_unstd /= len(valid_dataloader)
                all_valid_loss.append(valid_loss)
                note_min_loss = " *" if valid_loss == min(all_valid_loss) else ""
                logger.info(f"Epoch {epoch + 1:<{n_epoch_digits}} Validation loss: {valid_loss:.02e}{note_min_loss}")

            self.out_loss_record.append([epoch + 1, batch_loss, valid_loss, train_loss, valid_loss_unstd, train_loss_unstd, valid_loss0, train_loss0])

            is_best_condition_a = valid_loss == min(all_valid_loss)
            is_best_condition_b = epoch >= self.earliest_best
            if is_best_condition_a and is_best_condition_b:
                with self.name_tracker.set_context("epoch", "best"):
                    self.out_model.write(model=model,
                                        epoch="best"
                                        )

            # Checkpoint every so many epochs
            condition_a = (epoch + 1) in self.extra_check
            condition_b = (epoch + 1) % self.checkpoint == 0
            if (condition_a or condition_b):
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         epoch=epoch + 1)


        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

    def set_up_dataset(self, split: Split) -> None:
        # TODO: Unsure if this change is needed
        # label_path_template = self.make_fn_template(split, self.in_ps_asset)
        # feature_path_template = self.make_fn_template(split, self.in_map_asset)
        trgt_path_template = self.make_fn_template(split, self.in_trgt_asset)
        feat_path_template = self.make_fn_template(split, self.in_feat_asset)

        dataset = TrainDataset(
            n_sims = split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=trgt_path_template,
            # label_path_template=label_path_template,
            label_handler=self.in_trgt_asset.handler,
            feature_path_template=feat_path_template,
            # feature_path_template=feature_path_template,
            feature_handler=self.in_feat_asset.handler
            )
        return dataset


# def freeze_bn(m):
#     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#         m.eval()

def unstandardize(x_std: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # mu, sigma: shape [L]
    view_shape = [1] * x_std.ndim
    view_shape[-1] = -1
    return x_std * sigma.view(*view_shape) + mu.view(*view_shape)


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