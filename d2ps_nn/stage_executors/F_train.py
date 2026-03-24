import logging

from tqdm import tqdm

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
        self.in_model: Asset = self.assets_in["model"]
        self.in_pos_enc_lut: Asset = self.assets_in["pos_enc_lut"]
        in_trgt_handler: NumpyMap
        in_feat_handler: NumpyMap
        in_pos_enc_lut_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.scale_ps = make_ps_processor(cfg.model.scale_ps, cfg)

        # How model is evaluated
        self.loss_key      = cfg.model.train.loss

        # How model is updated
        self.learning_rate = cfg.model.train.learning_rate
        self.optimizer     = cfg.model.train.optimizer

        # How long it runs
        self.n_epochs      = cfg.model.train.n_epochs
        self.restart_epoch = cfg.model.train.restart_epoch

        # How it handles data
        self.batch_size    = cfg.model.train.batch_size
        self.num_workers   = cfg.model.train.num_loader_workers
        self.use_preload   = False  # Only for Overfit... I think 1000 sims will fill RAM of computer.?

        # Checkpointing
        self.checkpoint    = cfg.model.train.checkpoint_every
        self.extra_check   = cfg.model.train.extra_check
        self.earliest_best = cfg.model.train.earliest_best

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = self.init_model()

        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

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

        optimizer = self.get_optimizer(model)

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
                                   'Validation Pre Loss', 
                                   'Training Pre Loss', 
                                   'Validation Loss', 
                                   'Training Loss', ]
            self.out_loss_record.write(data=loss_record_headers)

        n_epoch_digits = len(str(self.n_epochs))

        all_batch_loss = []
        all_train_pre_loss = []
        all_valid_pre_loss = []

        post = self.scale_ps.postprocess
        all_train_pst_loss = []
        all_valid_pst_loss = []

        for epoch in range(start_epoch, self.n_epochs):
            # Training
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
                    optimizer.step()
                    pbar.set_postfix({'Loss': batch_train_loss.item()})
                    batch_loss += batch_train_loss.item()
                batch_loss /= len(train_dataloader)
                all_batch_loss.append(batch_loss)

            # Validation
            model.eval()
            train_pre_loss = 0
            valid_pre_loss = 0
            train_pst_loss = 0
            valid_pst_loss = 0
            with torch.no_grad():
                for train_features, train_label in tqdm(train_dataloader, leave=False):
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)
                    output = model(train_features)
                    train_pre_loss += loss_function(output, train_label).item()
                    train_pst_loss += loss_function(post(output), post(train_label)).item()
                train_pre_loss /= len(train_dataloader)
                train_pst_loss /= len(train_dataloader)
                all_train_pre_loss.append(train_pre_loss)
                all_train_pst_loss.append(train_pst_loss)
                logger.info(
                    f"Epoch {epoch + 1:<{n_epoch_digits}} "
                    f"tra_{self.scale_ps.pre_name}: {train_pre_loss:.02e} | "
                    f"tra_{self.scale_ps.post_name}: {train_pst_loss:.02e}"
                    f"{note_min_loss}"
                )

                for valid_features, valid_label in tqdm(valid_dataloader, leave=False):
                    valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                    valid_label = valid_label.to(device=self.device, dtype=self.dtype)
                    output = model(valid_features)
                    valid_pre_loss += loss_function(output, valid_label).item()
                    valid_pst_loss += loss_function(post(output), post(valid_label)).item()
                valid_pre_loss /= len(valid_dataloader)
                valid_pst_loss /= len(valid_dataloader)
                all_valid_pre_loss.append(valid_pre_loss)
                all_valid_pst_loss.append(valid_pst_loss)
                note_min_loss = " *" if valid_pre_loss == min(all_valid_pre_loss) else ""
                logger.info(
                    f"Epoch {epoch + 1:<{n_epoch_digits}} "
                    f"val_{self.scale_ps.pre_name}: {valid_pre_loss:.02e} | "
                    f"val_{self.scale_ps.post_name}: {valid_pst_loss:.02e}"
                    f"{note_min_loss}"
                )
            self.out_loss_record.append([epoch + 1, batch_loss, valid_pre_loss, train_pre_loss, train_pst_loss, valid_pst_loss])

            is_best_condition_a = valid_pre_loss == min(all_valid_pre_loss)
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
        with self.name_tracker.set_context("style", self.lut_style):
            trgt_path_template = self.make_fn_template(split, self.in_trgt_asset)
            feat_path_template = self.make_fn_template(split, self.in_feat_asset)

        freqs = self.instrument.dets.keys() if self.features_type == "obs" else [None]

        if self.use_preload:
            logger.info(f"Using dataset with preload. This eats RAM! BE CAREFUL!")

        dataset_class = PreloadTrainDataset if self.use_preload else TrainDataset

        dataset_params = dict(
            n_sims = split.n_sims,
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
            optim_dict = dict(lr=self.learning_rate)
            optimizer = torch.optim.Adam(model.parameters(), **optim_dict)
        elif self.optimizer == "AdamW":
            optim_dict = dict(lr=self.learning_rate, weight_decay=1e-4)
            optimizer = torch.optim.AdamW(model.parameters(), **optim_dict)
        logger.info(f"Optimizer: {optimizer.__class__} with {optim_dict}")
        return optimizer


class OverfitTrainingExecutor(TrainingExecutor):
    def __init__(self, cfg, stage_str="train"):
        super().__init__(cfg, stage_str=stage_str)
        self.use_preload   = cfg.model.train.get("use_preload", False)
        self.batch_size    = cfg.splits.Train.n_sims

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = self.init_model()

        train_split = self.splits.get("train")

        train_dataset = self.set_up_dataset(train_split)
        dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        train_dataloader = DataLoader(train_dataset, **dataloader_params)
        logger.info(f"Dataloader: {dataloader_params}")

        logger.info(f"Training on {len(train_dataloader)}batches, batch size {self.batch_size}.")

        loss_function = get_loss(self.loss_key, self.ells)
        loss_function = loss_function.to(self.device)
        logger.info(f"Loss: {self.loss_key}")

        optimizer = self.get_optimizer(model)

        logger.info(f"Starting new model.")
        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")
        start_epoch = 0
        loss_record_headers = [
            'Epoch',
            'Training Loss',          # scale/report-space epoch average, as before
            'Original Training Loss', # raw epoch average, as before
            'Loss Before Step',
            'Loss After Step',
            'Grad Norm',
            'Param Norm',
            'Update Norm',
            'Relative Update',
        ]
        self.out_loss_record.write(data=loss_record_headers)

        n_epoch_digits = len(str(self.n_epochs))

        all_batch_loss = []
        all_scale_loss = []

        for epoch in range(start_epoch, self.n_epochs):
            with tqdm(
                train_dataloader,
                desc=f"Ep {epoch + 1:<{n_epoch_digits}}",
                postfix={'Loss': 0}
            ) as pbar:
                model.train()

                batch_loss = 0.0
                scale_loss = 0.0

                epoch_loss_before_step = 0.0
                epoch_loss_after_step = 0.0
                epoch_grad_norm = 0.0
                epoch_param_norm = 0.0
                epoch_update_norm = 0.0
                epoch_relative_update = 0.0

                for train_features, lbl in pbar:
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    lbl = lbl.to(device=self.device, dtype=self.dtype)

                    optimizer.zero_grad()

                    # Forward before step
                    out = model(train_features)
                    loss = loss_function(out, lbl)
                    loss_before_step = loss.item()

                    with torch.no_grad():
                        out_sc = self.scale_ps.postprocess(out)
                        lbl_sc = self.scale_ps.postprocess(lbl)
                        loss_sc = loss_function(out_sc, lbl_sc).item()

                    # Save params before backward/step so we can measure update size
                    old_params = _clone_trainable_params(model)

                    # Backward
                    loss.backward()

                    # Diagnostics before step
                    grad_norm = _get_global_grad_norm(model.parameters(), max_norm=float("inf")).item()
                    param_norm = _get_param_norm(model)

                    # Optimizer step
                    optimizer.step()

                    # Forward after step on the exact same batch
                    with torch.no_grad():
                        out_after = model(train_features)
                        loss_after = loss_function(out_after, lbl).item()

                    update_norm = _get_update_norm(model, old_params)
                    relative_update = update_norm / (param_norm + 1e-12)

                    rep_loss = loss.item() if self.scale_ps.report_space == "pre" else loss_sc
                    pbar.set_postfix({'Loss': rep_loss})

                    batch_loss += loss.item()
                    scale_loss += loss_sc

                    epoch_loss_before_step += loss_before_step
                    epoch_loss_after_step += loss_after
                    epoch_grad_norm += grad_norm
                    epoch_param_norm += param_norm
                    epoch_update_norm += update_norm
                    epoch_relative_update += relative_update

                n_batches = len(train_dataloader)

                batch_loss /= n_batches
                scale_loss /= n_batches

                epoch_loss_before_step /= n_batches
                epoch_loss_after_step /= n_batches
                epoch_grad_norm /= n_batches
                epoch_param_norm /= n_batches
                epoch_update_norm /= n_batches
                epoch_relative_update /= n_batches

                all_batch_loss.append(batch_loss)
                all_scale_loss.append(scale_loss)

            self.out_loss_record.append([
                epoch + 1,
                scale_loss,
                batch_loss,
                epoch_loss_before_step,
                epoch_loss_after_step,
                epoch_grad_norm,
                epoch_param_norm,
                epoch_update_norm,
                epoch_relative_update,
            ])

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")


def _get_param_norm(model, norm_type=2.0):
    vals = []
    for p in model.parameters():
        if p.requires_grad:
            vals.append(p.detach().norm(norm_type))
    if not vals:
        return 0.0
    return torch.norm(torch.stack(vals), norm_type).item()


def _clone_trainable_params(model):
    return [
        p.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    ]


def _get_update_norm(model, old_params, norm_type=2.0):
    vals = []
    old_idx = 0
    for p in model.parameters():
        if p.requires_grad:
            vals.append((p.detach() - old_params[old_idx]).norm(norm_type))
            old_idx += 1
    if not vals:
        return 0.0
    return torch.norm(torch.stack(vals), norm_type).item()


def _get_global_grad_norm(model, norm_type=2.0):
    vals = [
        p.grad.detach().norm(norm_type)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not vals:
        return 0.0
    return torch.norm(torch.stack(vals), norm_type).item()
