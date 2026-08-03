import logging
from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from d2ps_nn.utils.loss import get_loss
from d2ps_nn.dataset import TrainDataset, PreloadTrainDataset
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.ps_scale import make_ps_processor
from d2ps_nn.models.attenuated_bn import set_bn_alpha


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainMetrics:
    batch_loss: float
    mean_grad_norm: Optional[float]
    max_grad_norm: Optional[float]


@dataclass(frozen=True)
class ValidMetrics:
    train_pre_loss:float
    valid_pre_loss:float
    train_pst_loss:float
    valid_pst_loss:float


class TrainingExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train") -> None:
        super().__init__(cfg, stage_str=stage_str)

        # What comes out
        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        # What goes in
        self.in_trgt_asset: Asset = self.assets_in["cmb_ps"]
        self.in_feat_asset = FeatureAsset.select(self.assets_in, 
                                                 cfg.features_type, 
                                                 self.instrument)
        self.in_resume_model: Asset = self.assets_in["resume_model"]
        self.in_initial_model: Asset = self.assets_in["initial_model"]
        self.in_pos_enc_lut: Asset = self.assets_in["pos_enc_lut"]
        in_trgt_handler: NumpyMap
        in_feat_handler: NumpyMap
        in_pos_enc_lut_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.lut_style = cfg.model.patches.get("lut_style", None)

        # How data was processed ()
        self.scale_ps = make_ps_processor(cfg.model.scale_ps, cfg)
        self.scale_px = cfg.model.scale_px

        # How model is evaluated
        self.loss_key      = cfg.model.train.loss
        self.n_sims_train_infer = cfg.get("n_sims_train_infer", None)

        # How model is updated
        self.lr_start      = cfg.model.train.learning_rate_start
        self.lr_final      = cfg.model.train.learning_rate_final
        self.lr_rmp_e      = cfg.model.train.learning_rate_ramp_epochs
        self.optimizer     = cfg.model.train.optimizer
        self.adamw_wd      = cfg.model.train.get("adamw_wd", None)

        # How long it runs
        self.n_epochs      = cfg.model.train.n_epochs
        self.restart_epoch = cfg.model.train.restart_epoch

        # How it handles data
        self.batch_size    = cfg.model.train.batch_size
        self.num_workers   = cfg.model.train.num_loader_workers
        self.use_preload   = False  # Only for Overfit... I think 1000 sims will fill RAM of computer.?

        # Checkpointing
        self.model_num     = cfg.get("model_num", None)
        self.checkpoint    = cfg.model.train.checkpoint_every
        self.extra_check   = cfg.model.train.extra_check
        self.earliest_best = cfg.model.train.earliest_best
        self.random_init   = cfg.model.get("random_init", False)

        self.track_grad_norm = cfg.get("track_grad_norm", False)
        self.n_epoch_digits = len(str(self.n_epochs))

    def get_pos_enc_lut(self):
        if not self.use_pos_enc_lut:
            return None
        with self.name_tracker.set_context("style", self.lut_style):
            pos_enc_lut = self.in_pos_enc_lut.read() 
        return pos_enc_lut

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = self.init_model()
        # TEMP: alpha=0 warm-start probe — remove before commit
        set_bn_alpha(model, 0.0)

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)
        logger.info(f"Loss: {self.loss_key}")

        train_dataloader, train_infer_dataloader, valid_dataloader = self.build_dataloaders()
        optimizer, scheduler = self.build_optimization(model, len(train_dataloader))

        # Initialize for training
        all_batch_loss = []
        all_train_pre_loss = []
        all_valid_pre_loss = []

        post = self.scale_ps.postprocess
        all_train_pst_loss = []
        all_valid_pst_loss = []

        is_continuation, start_epoch, initial_best_loss = self.prep_training_state(
            model, optimizer, scheduler
        )
        if initial_best_loss is not None:
            all_valid_pre_loss.append(initial_best_loss)
        self.set_up_loss_record(is_continuation)

        if start_epoch >= self.n_epochs:
            logger.warning(f"start_epoch {start_epoch} >= n_epochs {self.n_epochs}; no training will run.")

        vm0 = self.validate(model, train_infer_dataloader, valid_dataloader, loss_function, post, epoch=-1)
        logger.info(f"WARM-START GATE pre-train valid_pre: {vm0.valid_pre_loss:.02e}")
        # exit()

        best_valid_loss = initial_best_loss
        for epoch in range(start_epoch, self.n_epochs):
            tm:TrainMetrics = self.train_one_epoch(
                model, train_dataloader, loss_function, optimizer, scheduler, epoch)
            all_batch_loss.append(tm.batch_loss)

            vm: ValidMetrics = self.validate(
                model, train_infer_dataloader, valid_dataloader, loss_function, post, epoch)
            all_train_pre_loss.append(vm.train_pre_loss) 
            all_train_pst_loss.append(vm.train_pst_loss)
            all_valid_pre_loss.append(vm.valid_pre_loss)
            all_valid_pst_loss.append(vm.valid_pst_loss)

            best_valid_loss = min(all_valid_pre_loss)
            self.record_and_checkpoint(model, optimizer, epoch, tm, vm, best_valid_loss)

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, optimizer=optimizer, best_loss=best_valid_loss, epoch="final")

    def build_dataloaders(self):
        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

        train_dl = self._make_dataloader(
            self.set_up_dataset(train_split), 
            shuffle=True, 
            label="train")
        train_infer_dl = self._make_dataloader(
            self.set_up_dataset(train_split, early_stop=self.n_sims_train_infer), 
            shuffle=False, 
            label="train infer")
        valid_dl = self._make_dataloader(
            self.set_up_dataset(valid_split), 
            shuffle=False, 
            label="valid")

        logger.info(f"Training on {len(train_dl)} and Validation on {len(valid_dl)} "
                    f"batches, batch size {self.batch_size}.")
        return train_dl, train_infer_dl, valid_dl

    def set_up_dataset(self, split: Split, early_stop: int=None) -> None:
        context = dict(px_arrange=self.px_arrange, scale=self.scale_px)
        with self.name_tracker.set_contexts(context):
            trgt_path_template = self.make_fn_template(split, self.in_trgt_asset)
            feat_path_template = self.make_fn_template(split, self.in_feat_asset)

        freqs = self.instrument.dets.keys() if self.features_type == "obs" else [None]

        if self.use_preload:
            logger.info(f"Using dataset with preload. This eats RAM! BE CAREFUL!")

        dataset_class = PreloadTrainDataset if self.use_preload else TrainDataset

        dataset_params = dict(
            n_sims = split.n_sims if early_stop is None else early_stop,
            freqs = freqs,
            map_fields=self.map_fields,
            label_path_template=trgt_path_template,
            label_handler=self.in_trgt_asset.handler,
            feature_path_template=feat_path_template,
            feature_handler=self.in_feat_asset.handler,
            px_arrange=self.px_arrange,
            pos_enc_lut=self.get_pos_enc_lut()
        )
        logger.info(f"Dataset: {dataset_params}")

        dataset = dataset_class(**dataset_params)
        return dataset

    def _make_dataloader(self, dataset, shuffle, label):
        params = dict(batch_size=self.batch_size, shuffle=shuffle,
                    num_workers=self.num_workers)
        logger.info(f"Dataloader {label} (shuffle={shuffle}): {params}")
        return DataLoader(dataset, **params)

    def build_optimization(self, model, steps_per_epoch):
        total_warmup_steps = self.lr_rmp_e * steps_per_epoch

        optimizer = self.get_optimizer(model)
        if self.lr_rmp_e > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                        start_factor=self.lr_start / self.lr_final,
                                                        end_factor=1.0,
                                                        total_iters=total_warmup_steps
                                                        )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                            factor=1,
                                                            total_iters=1)
        return optimizer, scheduler
    
    def get_optimizer(self, model):
        if self.optimizer == "Adam":
            optim_dict = dict(lr=self.lr_final)
            optimizer = torch.optim.Adam(model.parameters(), **optim_dict)

        elif self.optimizer == "AdamW":
            if self.adamw_wd is None:
                raise ValueError(
                    "Must set a value for AdamW weight decay in config, "
                    "even if forcing it to 0.0"
                )

            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 1 or name.endswith(".bias"):
                    no_decay.append(p)
                else:
                    decay.append(p)

            optim_dict = dict(
                lr=self.lr_final,
                weight_decay=self.adamw_wd,
                n_decay_params=len(decay),
                n_no_decay_params=len(no_decay),
            )

            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.adamw_wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr_final,
            )

        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        logger.info(f"Optimizer: {optimizer.__class__.__name__} with {optim_dict}")
        return optimizer

    def prep_training_state(self, model, optimizer, scheduler):
        is_continuation = False
        start_epoch = 0
        initial_best_loss = None
        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                model_record = self.in_resume_model.read(model=model,
                                                         epoch=self.restart_epoch,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler)
            if model_record['epoch'] != "init" and int(model_record['epoch']) != int(self.restart_epoch):
                raise ValueError(
                    f"Requested restart at epoch {self.restart_epoch} but loaded checkpoint "
                    f"reports epoch {model_record['epoch']}; no matching checkpoint on disk."
                )
            start_epoch = 0 if model_record['epoch'] == "init" else int(model_record['epoch'])
            if model_record['best_loss'] is not None:
                initial_best_loss = model_record['best_loss']
                logger.info(f"Resuming with stored best loss: {initial_best_loss:.02e}")
            else:
                initial_best_loss = None
                logger.warning("Restarting without a stored best loss; 'best' checkpoints "
                            "will be judged only against epochs from this run.")
            is_continuation = True
        elif self.random_init:
            logger.info(f"Starting new, randomly initialized model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
        else:
            logger.info(f"Loading transfer-initialization model.")
            context = dict(model_num=self.model_num,
                           pos_enc=self.use_pos_enc_lut)
            with self.name_tracker.set_contexts(context):
                self.in_initial_model.read(model=model, strict=False)
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
        
        return is_continuation, start_epoch, initial_best_loss

    def set_up_loss_record(self, is_continuation):
        if is_continuation:
            self.out_loss_record.start()  # New lines will be added, naively
                                          #  Naively = appended; if resuming, manual cleaning may be needed.
        else:
            loss_record_headers = ['Epoch', 
                                   'Batch Loss', 
                                   'Validation Pre Loss', 
                                   'Training Pre Loss', 
                                   'Validation Loss', 
                                   'Training Loss', 
                                   'Learning Rate']
            if self.track_grad_norm:
                loss_record_headers.append('Mean Grad Norm')
                loss_record_headers.append('Max Grad Norm')
            self.out_loss_record.write(data=loss_record_headers)

    def train_one_epoch(self, model, train_dataloader, loss_function, optimizer, scheduler, epoch):
        epoch_grad_norm = []
        with tqdm(train_dataloader, desc=f"Ep {epoch + 1:<{self.n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
            model.train()
            batch_loss = 0
            for train_features, train_label in pbar:
                train_features = train_features.to(device=self.device, dtype=self.dtype)
                train_label = train_label.to(device=self.device, dtype=self.dtype)

                optimizer.zero_grad()
                
                output = model(train_features)
                batch_train_loss = loss_function(output, train_label)
                batch_train_loss.backward()

                if self.track_grad_norm:
                    epoch_grad_norm.append(_get_global_grad_norm(model))

                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'Loss': batch_train_loss.item()})
                batch_loss += batch_train_loss.item()
        batch_loss /= len(train_dataloader)
        if self.track_grad_norm:
            mean_gn = np.mean(epoch_grad_norm)
            max_gn = np.max(epoch_grad_norm)
        else:
            mean_gn = None
            max_gn = None
        return TrainMetrics(batch_loss=batch_loss, mean_grad_norm=mean_gn, max_grad_norm=max_gn)

    def validate(self, model, train_infer_dataloader, valid_dataloader, loss_function, post, epoch):
        model.eval()
        with torch.no_grad():
            train_pre, train_pst = evaluate(
                model=model,
                dataloader=train_infer_dataloader,
                loss_function=loss_function,
                post=post,
                device=self.device,
                dtype=self.dtype,
            )
            logger.info(
                f"Epoch {epoch + 1:<{self.n_epoch_digits}} "
                f"train_{self.scale_ps.pre_name}: {train_pre:.02e} | "
                f"train_{self.scale_ps.post_name}: {train_pst:.02e}"
            )

            valid_pre, valid_pst = evaluate(
                model=model,
                dataloader=valid_dataloader,
                loss_function=loss_function,
                post=post,
                device=self.device,
                dtype=self.dtype,
            )
            # This is logged in record_and_checkpoint

        return ValidMetrics(
            train_pre_loss=train_pre, train_pst_loss=train_pst,
            valid_pre_loss=valid_pre, valid_pst_loss=valid_pst
        )

    # record_and_checkpoint(model, optimizer, epoch, tm, vm, all_valid_pre_loss)
    def record_and_checkpoint(self, model, optimizer, epoch, 
                              tm: TrainMetrics, vm: ValidMetrics,
                              best_valid_loss):
            note_min_loss = " *" if vm.valid_pre_loss == best_valid_loss else ""
            logger.info(
                f"Epoch {epoch + 1:<{self.n_epoch_digits}} "
                f"valid_{self.scale_ps.pre_name}: {vm.valid_pre_loss:.02e} | "
                f"valid_{self.scale_ps.post_name}: {vm.valid_pst_loss:.02e}"
                f"{note_min_loss}"
            )

            record_line = [
                epoch + 1, 
                tm.batch_loss, 
                vm.valid_pre_loss, 
                vm.train_pre_loss, 
                vm.valid_pst_loss, 
                vm.train_pst_loss,
                optimizer.param_groups[0]["lr"]            
                ]
            if self.track_grad_norm:
                record_line.append(tm.mean_grad_norm)
                record_line.append(tm.max_grad_norm)
            self.out_loss_record.append(record_line)

            is_best_condition_a = vm.valid_pre_loss == best_valid_loss
            is_best_condition_b = epoch >= self.earliest_best
            if is_best_condition_a and is_best_condition_b:
                with self.name_tracker.set_context("epoch", "best"):
                    self.out_model.write(model=model,
                                         epoch="best",
                                         best_loss=best_valid_loss,
                                         optimizer=optimizer
                                         )

            # Checkpoint every so many epochs
            condition_a = (epoch + 1) in self.extra_check
            condition_b = (epoch + 1) % self.checkpoint == 0
            if (condition_a or condition_b):
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         best_loss=best_valid_loss,
                                         epoch=epoch + 1)

def evaluate(model, dataloader, loss_function, post, device, dtype):
    pre_loss = 0.0
    pst_loss = 0.0

    model.eval()

    with torch.inference_mode():
        for features, label in tqdm(dataloader, leave=False):
            features = features.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=dtype)

            output = model(features)

            pre_loss += loss_function(output, label).item()
            pst_loss += loss_function(
                post(output),
                post(label),
            ).item()

    n_batches = len(dataloader)
    return pre_loss / n_batches, pst_loss / n_batches


def _get_global_grad_norm(model, norm_type=2.0):
    vals = [
        p.grad.detach().norm(norm_type)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not vals:
        return 0.0
    return torch.norm(torch.stack(vals), norm_type).item()
