import logging
from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler, NpzHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from d2ps_nn.utils.loss import get_loss
from d2ps_nn.dataset import TrainDataset, PreloadTrainDataset
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.ps_scale import make_ps_processor
from d2ps_nn.models.bn_recompute import recompute_bn_stats
from d2ps_nn.utils.metrics_record import ColumnSpec, RecordSchema, metric, specs_from
from d2ps_nn.utils.ell_metrics_recorder import EllMetricsRecorder
from d2ps_nn.utils.planck_recorder import PlanckRecorder
from d2ps_nn.utils.metrics_record import EllStatsAccumulator


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainMetrics:
    batch_loss:     float           = metric("Batch Loss")
    mean_grad_norm: Optional[float] = metric("Mean Grad Norm", enabled_by="track_grad_norm")
    max_grad_norm:  Optional[float] = metric("Max Grad Norm",  enabled_by="track_grad_norm")
  
@dataclass(frozen=True)
class ValidMetrics:
    valid_pre_loss: float = metric("Validation Pre Loss")
    train_pre_loss: float = metric("Training Pre Loss")
    valid_pst_loss: float = metric("Validation Loss")
    train_pst_loss: float = metric("Training Loss")

# Two columns that come from neither of the Metrics dataclass. Declared here so 
# they live in the same ordered list as everything else rather than being
# spliced in by hand at two separate call sites.
EPOCH_COLUMN = ColumnSpec(name="epoch", header="Epoch")
LR_COLUMN = ColumnSpec(name="learning_rate", header="Learning Rate")


class TrainingExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.cf_planck = cfg.get("cf_planck", False)

        # What comes out
        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        self.out_ell_record: Asset = self.assets_out["ell_record"]
        self.out_trgt_stats: Asset = self.assets_out["trgt_stats"]
        self.out_val_preds: Asset = self.assets_out["predictions"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler
        out_ell_record: AppendingCsvHandler
        out_trgt_stats: NpzHandler

        if self.cf_planck:
            self.out_pl_record: Asset = self.assets_out["pl_record"]

        # What goes in
        self.in_trgt_asset: Asset = self.assets_in["cmb_ps"]
        self.in_feat_asset = FeatureAsset.select(self.assets_in, 
                                                 cfg.features_type, 
                                                 self.instrument)
        self.pl_obs = None
        if self.cf_planck:
            self.in_planck_obs: Asset = self.assets_in["pl_obs"]
        self.in_resume_model: Asset = self.assets_in["resume_model"]
        self.in_initial_model: Asset = self.assets_in["initial_model"]
        self.in_pos_enc_lut: Asset = self.assets_in["pos_enc_lut"]
        self.in_mask: Asset = self.assets_in["mask"]
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

        self.ell_recorder = EllMetricsRecorder(
            enabled=cfg.model.train.get("track_ell_metrics", False),
            ells=self.ells,
            record_asset=self.out_ell_record,
            ref_asset=self.out_trgt_stats,
            device=self.device,
        )
        self.pl_recorder = PlanckRecorder(
            enabled=self.cf_planck,
            ells=self.ells,
            record_asset=self.out_pl_record if self.cf_planck else None,
        )

        self.keep_valid_preds = cfg.model.train.get("track_val_preds", False)
        if self.keep_valid_preds:
            logger.info("Validation predictions will be recorded per epoch.")

        raw_momentum = cfg.model.train.get("bn_momentum", 0.1)
        if raw_momentum is None or (
            isinstance(raw_momentum, str) and raw_momentum.strip().lower() == "none"
        ):
            self.bn_momentum = None
        else:
            self.bn_momentum = float(raw_momentum)
 
        # Fully recompute BN running stats over the training set after each of
        # the first N epochs. Needed from scratch, where a low momentum leaves
        # buffers uncalibrated for many epochs. 0 disables.
        self.bn_recompute_epochs = int(cfg.model.train.get("bn_recompute_epochs", 0))
        self.bn_recompute_max_batches = cfg.model.train.get(
            "bn_recompute_max_batches", None
        )

        self.use_mask = cfg.model.mask_features

    def get_pos_enc_lut(self):
        if not self.use_pos_enc_lut:
            logger.info("No positional encoding will be used.")
            return None
        with self.name_tracker.set_context("style", self.lut_style):
            pos_enc_lut = self.in_pos_enc_lut.read() 
            logger.info("Positional encoding will be used.")
        return pos_enc_lut

    def get_mask(self):
        if not self.use_mask:
            logger.info("Features will not be masked.")
            return None
        with self.name_tracker.set_context("px_arrange", self.px_arrange):
            mask = self.in_mask.read()
            logger.info("Features will be masked.")
        return mask

    def set_up_planck(self):
            n_dets = len(self.instrument.dets)
            for i, freq in enumerate(self.instrument.dets.keys()):
                with self.name_tracker.set_context("freq", freq):
                    this_map = self.in_planck_obs.read()
                    if self.pl_obs is None:
                        self.pl_obs = np.ndarray((n_dets, this_map.shape[1], this_map.shape[2]))
                    self.pl_obs[i] = this_map[0]

            # (n_dets, H, W) is ONE example, so add the batch dim here.
            #
            # WARNING: the training dataloader applies px_arrange and appends the
            # positional-encoding LUT before the model sees a feature. The raw stack
            # below does neither. If your model expects either, feed it this and you
            # get a shape error at best, plausible-looking garbage at worst.
            if self.use_pos_enc_lut:
                raise NotImplementedError
            self.pl_obs = torch.from_numpy(self.pl_obs).unsqueeze(0)
            logger.info(f"Loaded Planck comparison maps: feature tensor {tuple(self.pl_obs.shape)}")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        if self.cf_planck:
            self.set_up_planck()
        else:
            logger.warning("Skipped Planck comparison maps. Kinda goes against the point, but you do you.")

        model = self.init_model()
        self._prepare_model(model)

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
            self.record_and_checkpoint(model, optimizer, epoch, tm, vm, best_valid_loss, post)

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, optimizer=optimizer, best_loss=best_valid_loss, epoch="final")

    def _prepare_model(self, model):
        """
        Structural model prep that must precede optimizer construction
        (get_optimizer reads requires_grad). Subclasses should call super().
        """
        bns = [m for m in model.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if bns:
            for m in bns:
                m.momentum = self.bn_momentum
            logger.info(
                f"set bn momentum to {self.bn_momentum!r} (None = cumulative) "
                f"on {len(bns)} BatchNorm layer(s)"
            )

    def _log_bn_stat_norms(self, model, tag):
        """Summed norms of BN running stats; identical values across epochs
        prove the buffers are frozen."""
        bns = [m for m in model.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not bns:
            return
        rm = sum(float(m.running_mean.norm()) for m in bns)
        rv = sum(float(m.running_var.norm()) for m in bns)
        logger.info(
            f"BN stat norms [{tag}]: sum|running_mean|={rm:.6g}  "
            f"sum|running_var|={rv:.6g}"
        )

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
            pos_enc_lut=self.get_pos_enc_lut(),
            mask=self.get_mask()
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
            trainable = [p for p in model.parameters() if p.requires_grad]
            n_total = sum(1 for _ in model.parameters())
            optim_dict = dict(lr=self.lr_final, n_trainable=len(trainable), n_frozen=n_total - len(trainable))
            optimizer = torch.optim.Adam(trainable, lr=self.lr_final)
            
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

    train_metrics_cls = TrainMetrics
    valid_metrics_cls = ValidMetrics

    def build_loss_schema(self) -> RecordSchema:
        """The scalar, one-row-per-epoch record.
 
        enabled_by is resolved against self, so track_grad_norm drops those
        columns from the header and the row together.
        """
        specs = ([EPOCH_COLUMN]
                 + specs_from(self.train_metrics_cls, self.valid_metrics_cls)
                 + [LR_COLUMN])
        return RecordSchema(specs, flags=self)
 
    def set_up_loss_record(self, is_continuation):
        self.loss_schema = self.build_loss_schema()
        if is_continuation:
            self.out_loss_record.start()  # New lines will be added, naively
                                          #  Naively = appended; if resuming, manual cleaning may be needed.
        else:
            self.out_loss_record.write(data=self.loss_schema.headers())
        self.ell_recorder.begin_run(is_continuation)  # no-op when disabled
        self.pl_recorder.begin_run(is_continuation)

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
 
        if epoch < self.bn_recompute_epochs:
            self._log_bn_stat_norms(model, f"ep {epoch + 1} pre-recompute")
            n_used = recompute_bn_stats(
                model,
                train_dataloader,
                device=self.device,
                dtype=self.dtype,
                max_batches=self.bn_recompute_max_batches,
            )
            logger.info(
                f"recomputed BN running stats over {n_used} batches "
                f"(epoch {epoch + 1} of first {self.bn_recompute_epochs})"
            )
            self._log_bn_stat_norms(model, f"ep {epoch + 1} post-recompute")
 
        return TrainMetrics(batch_loss=batch_loss,
                            mean_grad_norm=mean_gn,
                            max_grad_norm=max_gn)

    def validate(self, model, train_infer_dataloader, valid_dataloader, loss_function, post, epoch):
        model.eval()
 
        # None when per-ell tracking is off, which evaluate() reads as
        # "don't accumulate".
        ell_acc = self.ell_recorder.new_accumulator()
        pred_collector = _PredCollector() if self.keep_valid_preds else None

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
 
            # Per-ell diagnostics come from the validation pass only: it is the
            # fixed set the reference sidecar describes. train_infer is also
            # fixed, but it is a different set, and mixing the two would make
            # the drift check meaningless.
            valid_pre, valid_pst = evaluate(
                model=model,
                dataloader=valid_dataloader,
                loss_function=loss_function,
                post=post,
                device=self.device,
                dtype=self.dtype,
                ell_accumulator=ell_acc,
                pred_collector=pred_collector
            )

            val_preds = pred_collector.result()
            if val_preds is not None:
                with self.name_tracker.set_context("epoch", f"{epoch+1:0{self.n_epoch_digits}d}"):
                    self.out_val_preds.write(data=val_preds)
 
        return ValidMetrics(
            train_pre_loss=train_pre, train_pst_loss=train_pst,
            valid_pre_loss=valid_pre, valid_pst_loss=valid_pst
        )

    def record_and_checkpoint(self, model, optimizer, epoch, 
                              tm: TrainMetrics, vm: ValidMetrics,
                              best_valid_loss, post):
            note_min_loss = " *" if vm.valid_pre_loss == best_valid_loss else ""
            logger.info(
                f"Epoch {epoch + 1:<{self.n_epoch_digits}} "
                f"valid_{self.scale_ps.pre_name}: {vm.valid_pre_loss:.02e} | "
                f"valid_{self.scale_ps.post_name}: {vm.valid_pst_loss:.02e}"
                f"{note_min_loss}"
            )
 
            extras = {"epoch": epoch + 1,
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            self.out_loss_record.append(self.loss_schema.row(tm, vm, extras))
 
            # Writes the epoch row, and on the first pass establishes the
            # reference sidecar; on later passes checks against it.
            self.ell_recorder.record(epoch)
            self.pl_recorder.record(model, self.pl_obs, post, epoch)

            is_best_condition_a = vm.valid_pre_loss == best_valid_loss
            is_best_condition_b = epoch >= self.earliest_best
            if is_best_condition_a and is_best_condition_b:
                with self.name_tracker.set_context("epoch", f"best_ep{epoch+1}"):
                    self.out_model.write(model=model,
                                         epoch=f"best_ep{epoch+1}",
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


class _PredCollector:
    """Retain per-batch validation predictions for one pass, cat at the end."""
    def __init__(self):
        self._chunks = []

    def update(self, output_pst):
        # .cpu() per batch so GPU memory stays flat -- the ell=1280 cat blowup
        # you flagged in the OTF code, avoided.
        self._chunks.append(output_pst.detach().to("cpu"))

    def result(self):
        if not self._chunks:
            return None
        return torch.cat(self._chunks, dim=0).numpy()   # (n_valid, n_ells)


def evaluate(model, dataloader, loss_function, post, device, dtype,
             ell_accumulator: Optional[EllStatsAccumulator] = None,
             pred_collector: Optional[_PredCollector] = None):
    pre_loss = 0.0
    pst_loss = 0.0
 
    model.eval()
 
    # inference_mode tags its outputs as inference tensors, which cannot be
    # read back out afterwards; fall back to no_grad when accumulating.
    retain = ell_accumulator is not None or pred_collector is not None
    ctx = torch.no_grad() if retain else torch.inference_mode()

    with ctx:
        for features, label in tqdm(dataloader, leave=False):
            features = features.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=dtype)
 
            output = model(features)
            output_pst = post(output)
            label_pst = post(label)
 
            pre_loss += loss_function(output, label).item()
            pst_loss += loss_function(output_pst, label_pst).item()
 
            if ell_accumulator is not None:
                ell_accumulator.update(output_pst, label_pst)
            if pred_collector is not None:
                pred_collector.update(output_pst)
 
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
 
