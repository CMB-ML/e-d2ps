import logging
import time

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


logger = logging.getLogger(__name__)


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

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = self.init_model()

        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

        train_infer_dataset = self.set_up_dataset(train_split, early_stop=self.n_sims_train_infer)
        train_infer_dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        logger.info(f"Train dataloader: {train_infer_dataloader_params}")
        train_infer_dataloader = DataLoader(train_infer_dataset, **train_infer_dataloader_params)

        train_dataset = self.set_up_dataset(train_split)
        train_dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        logger.info(f"Train dataloader: {train_dataloader_params}")
        train_dataloader = DataLoader(train_dataset, **train_dataloader_params)

        valid_dataset = self.set_up_dataset(valid_split)
        valid_dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        logger.info(f"Valid dataloader: {valid_dataloader_params}")        
        valid_dataloader = DataLoader(valid_dataset, **valid_dataloader_params)

        logger.info(f"Training on {len(train_dataloader)} and Validation on {len(valid_dataloader)} batches, batch size {self.batch_size}.")

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)
        logger.info(f"Loss: {self.loss_key}")

        steps_per_epoch = len(train_dataloader)
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
        # scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimizer)
        # scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_0, scheduler_1])

        # Initialize for training
        all_batch_loss = []
        all_avg_grad_norm = []
        all_max_grad_norm = []
        all_train_pre_loss = []
        all_valid_pre_loss = []

        post = self.scale_ps.postprocess
        all_train_pst_loss = []
        all_valid_pst_loss = []

        do_loss_record = "new"
        start_epoch = 0
        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                model_record = self.in_resume_model.read(model=model, 
                                                         epoch=self.restart_epoch, 
                                                         optimizer=optimizer, 
                                                         scheduler=scheduler
                                                         )
            start_epoch = 0 if model_record['epoch'] == "init" else int(model_record['epoch']) + 1
            if model_record['best_loss'] is not None:
                all_valid_pre_loss.append(model_record['best_loss'])
            do_loss_record = "continue"
            logger.warning("Restarting training without finding previous best loss!")
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

        if do_loss_record == "continue":
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

        n_epoch_digits = len(str(self.n_epochs))

        for epoch in range(start_epoch, self.n_epochs):
            # Training
            epoch_grad_norm = []
            with tqdm(train_dataloader, desc=f"Ep {epoch + 1:<{n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
                model.train()
                batch_loss = 0
                for train_features, train_label in pbar:
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    optimizer.zero_grad()
                    
                    output = model(train_features)
                    # logger.info(f"Memory after forward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                    # loss = loss_function(output, train_label)
                    # loss.backward()
                    # logger.info(f"Memory after backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                    # exit()

                    batch_train_loss = loss_function(output, train_label)
                    batch_train_loss.backward()

                    if self.track_grad_norm:
                        epoch_grad_norm.append(_get_global_grad_norm(model))

                    optimizer.step()
                    scheduler.step()
                    pbar.set_postfix({'Loss': batch_train_loss.item()})
                    batch_loss += batch_train_loss.item()
            batch_loss /= len(train_dataloader)
            all_batch_loss.append(batch_loss)
            if self.track_grad_norm:
                mean_grad_norm = np.mean(epoch_grad_norm)
                all_avg_grad_norm.append(mean_grad_norm)
                max_grad_norm = np.max(epoch_grad_norm)
                all_max_grad_norm.append(max_grad_norm)

            # Validation
            model.eval()
            train_pre_loss = 0
            valid_pre_loss = 0
            train_pst_loss = 0
            valid_pst_loss = 0
            with torch.no_grad():
                train_pre_loss, train_pst_loss = evaluate(
                    model=model,
                    dataloader=train_infer_dataloader,
                    loss_function=loss_function,
                    post=post,
                    device=self.device,
                    dtype=self.dtype,
                )
                all_train_pre_loss.append(train_pre_loss)
                all_train_pst_loss.append(train_pst_loss)
                logger.info(
                    f"Epoch {epoch + 1:<{n_epoch_digits}} "
                    f"train_{self.scale_ps.pre_name}: {train_pre_loss:.02e} | "
                    f"train_{self.scale_ps.post_name}: {train_pst_loss:.02e}"
                )

                valid_pre_loss, valid_pst_loss = evaluate(
                    model=model,
                    dataloader=valid_dataloader,
                    loss_function=loss_function,
                    post=post,
                    device=self.device,
                    dtype=self.dtype,
                )
                all_valid_pre_loss.append(valid_pre_loss)
                all_valid_pst_loss.append(valid_pst_loss)
                note_min_loss = " *" if valid_pre_loss == min(all_valid_pre_loss) else ""
                logger.info(
                    f"Epoch {epoch + 1:<{n_epoch_digits}} "
                    f"valid_{self.scale_ps.pre_name}: {valid_pre_loss:.02e} | "
                    f"valid_{self.scale_ps.post_name}: {valid_pst_loss:.02e}"
                    f"{note_min_loss}"
                )
            record_line = [
                epoch + 1, 
                batch_loss, 
                valid_pre_loss, 
                train_pre_loss, 
                valid_pst_loss, 
                train_pst_loss,
                optimizer.param_groups[0]["lr"]            
                ]
            if self.track_grad_norm:
                record_line.append(mean_grad_norm)
                record_line.append(max_grad_norm)
            self.out_loss_record.append(record_line)

            is_best_condition_a = valid_pre_loss == min(all_valid_pre_loss)
            is_best_condition_b = epoch >= self.earliest_best
            if is_best_condition_a and is_best_condition_b:
                with self.name_tracker.set_context("epoch", "best"):
                    self.out_model.write(model=model,
                                         epoch="best",
                                         optimizer=optimizer
                                         )

            # Checkpoint every so many epochs
            condition_a = (epoch + 1) in self.extra_check
            condition_b = (epoch + 1) % self.checkpoint == 0
            if (condition_a or condition_b):
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         best_loss=min(all_valid_pre_loss),
                                         epoch=epoch + 1)

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

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

    def get_pos_enc_lut(self):
        if not self.use_pos_enc_lut:
            return None
        with self.name_tracker.set_context("style", self.lut_style):
            pos_enc_lut = self.in_pos_enc_lut.read() 
        return pos_enc_lut

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
