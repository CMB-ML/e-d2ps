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
from d2ps_nn.utils.gpu_stats import TorchGPUStats
from d2ps_nn.stage_executors.F_train import TrainingExecutor


logger = logging.getLogger(__name__)


class LRFindingTrainingExecutor(TrainingExecutor):
    def __init__(self, cfg, stage_str="train"):
        super().__init__(cfg, stage_str=stage_str)
        self.final_lr = cfg.model.train.get("final_learning_rate", 1e-3)
    
    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = self.init_model()

        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        valid_dataset = self.set_up_dataset(valid_split)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loss_function = get_loss(self.loss_key, self.ells).to(self.device)
        post = self.scale_ps.postprocess

        optimizer = self.get_optimizer(model)

        # Load/init model exactly as in normal training.
        if self.random_init:
            logger.info("Starting new, randomly initialized model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
        else:
            logger.info("Loading transfer-initialization model.")
            context = dict(model_num=self.model_num, pos_enc=self.use_pos_enc_lut)
            with self.name_tracker.set_contexts(context):
                self.in_initial_model.read(model=model, strict=False)
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")

        lr_start = self.lr_start
        lr_end = self.final_lr
        total_steps = self.n_epochs * len(train_dataloader)
        gamma = (lr_end / lr_start) ** (1.0 / max(total_steps - 1, 1))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )

        headers = [
            "Step",
            "Epoch",
            "Learning Rate",
            "Batch Loss",
            "Validation Pre Loss",
            "Validation Loss",
            # "Training Pre Loss",
            # "Training Loss",
        ]
        self.out_loss_record.write(data=headers)

        step = 0
        n_epoch_digits = len(str(self.n_epochs))

        for epoch in range(self.n_epochs):
            model.train()

            with tqdm(
                train_dataloader,
                desc=f"Ep {epoch + 1:<{n_epoch_digits}}",
                postfix={"Loss": 0.0, "LR": optimizer.param_groups[0]["lr"]},
            ) as pbar:

                for train_features, train_label in pbar:
                    lr = optimizer.param_groups[0]["lr"]

                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    optimizer.zero_grad()

                    output = model(train_features)
                    batch_loss = loss_function(output, train_label)
                    batch_loss.backward()
                    optimizer.step()

                    # Validate every optimizer step.
                    model.eval()
                    valid_pre_loss = 0.0
                    valid_pst_loss = 0.0

                    with torch.no_grad():
                        for valid_features, valid_label in valid_dataloader:
                            valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                            valid_label = valid_label.to(device=self.device, dtype=self.dtype)

                            valid_output = model(valid_features)
                            valid_pre_loss += loss_function(valid_output, valid_label).item()
                            valid_pst_loss += loss_function(
                                post(valid_output),
                                post(valid_label),
                            ).item()

                    valid_pre_loss /= len(valid_dataloader)
                    valid_pst_loss /= len(valid_dataloader)

                    self.out_loss_record.append([
                        step + 1,
                        epoch + 1,
                        lr,
                        batch_loss.item(),
                        valid_pre_loss,
                        valid_pst_loss,
                        # train_pre_loss,
                        # train_pst_loss,
                    ])

                    pbar.set_postfix({
                        "Loss": batch_loss.item(),
                        "Val": valid_pre_loss,
                        "LR": lr,
                    })

                    scheduler.step()
                    step += 1

                    model.train()


class OverfitTrainingExecutor(TrainingExecutor):
    def __init__(self, cfg, stage_str="train"):
        super().__init__(cfg, stage_str=stage_str)
        self.use_preload   = cfg.model.train.get("use_preload", False)
        self.batch_size    = cfg.splits.Train.n_sims
        self.gpu_stats     = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.gpu_stats = TorchGPUStats(self.device)

        model = self.init_model()
        context = dict(model_num=self.model_num,
                    pos_enc=self.use_pos_enc_lut)
        with self.name_tracker.set_contexts(context):
            self.in_initial_model.read(model=model, strict=False)

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

        train_features, lbl = next(iter(train_dataloader))
        train_features = train_features.to(device=self.device, dtype=self.dtype)
        lbl = lbl.to(device=self.device, dtype=self.dtype)

        header_printed = False

        loss_record_lines = []

        for epoch in range(start_epoch, self.n_epochs):
            t0 = time.time()

            model.train()
            optimizer.zero_grad()

            # Forward before step
            out = model(train_features)
            loss = loss_function(out, lbl)
            loss_before = loss.item()

            with torch.no_grad():
                out_sc = self.scale_ps.postprocess(out)
                lbl_sc = self.scale_ps.postprocess(lbl)
                loss_sc = loss_function(out_sc, lbl_sc).item()

            # Save params before backward/step so we can measure update size
            old_params = _clone_trainable_params(model)

            # Backward
            loss.backward()

            # Diagnostics before step
            grad_norm = _get_global_grad_norm(model)
            param_norm = _get_param_norm(model)

            # Optimizer step
            optimizer.step()

            # Forward after step on the exact same batch
            with torch.no_grad():
                out_after = model(train_features)
                loss_after = loss_function(out_after, lbl).item()

            update_norm = _get_update_norm(model, old_params)
            relative_update = update_norm / (param_norm + 1e-12)

            batch_loss = loss.item()
            rep_loss = loss.item() if self.scale_ps.report_space == "pre" else loss_sc
            dt = time.time() - t0

            if not header_printed:
                print(f"{'Ep':>4} | {'Loss':>10} | {'rate (1/s)':>10}")
                print("-" * 44)
                header_printed = True

            dt = time.time() - t0
            ep_per_sec = 1.0 / dt if dt > 0 else float("inf")
            gpu = self.gpu_stats.get_stats()

            logger.info(
                f"{epoch+1:4d} | "
                f"{rep_loss:10.3e} | "
                f"{ep_per_sec:7.2f} ep/s | "
                f"{gpu['gpu_util']:3d}% util | "
                f"{gpu['temp_C']:2d} C | "
                f"{gpu['power_W']:6.1f} W | "
                f"{gpu['sm_clock_MHz']:4d} MHz"
            )

            loss_record_lines.append([
                epoch + 1,
                loss_sc,
                batch_loss,
                loss_before,
                loss_after,
                grad_norm,
                param_norm,
                update_norm,
                relative_update,
            ])

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

        for l in loss_record_lines:
            self.out_loss_record.append(l)


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
