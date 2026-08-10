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
import copy
import numpy as np
import torch
import torch.nn.modules.batchnorm as _bn


logger = logging.getLogger(__name__)


def _bn_modules(model):
    return [m for m in model.modules() if isinstance(m, _bn._BatchNorm)]


def set_only_bn_train(model):
    """Whole model to eval (dropout OFF), then BN layers back to train
    so they update running stats on the forward pass."""
    model.eval()
    for m in _bn_modules(model):
        m.train()


def reset_bn(model, cumulative=True):
    """Zero running stats. cumulative=True -> momentum=None -> BN accumulates a
    simple cumulative average over the pass (order-independent, exact population
    stats). cumulative=False keeps the EMA (order-dependent)."""
    for m in _bn_modules(model):
        m.reset_running_stats()          # mean->0, var->1, num_batches_tracked->0
        if cumulative:
            m.momentum = None


@torch.no_grad()   # NOT inference_mode — buffer mutation is cleaner under no_grad
def recompute_bn(model, dataloader, device, dtype, n_batches=None):
    set_only_bn_train(model)
    for i, (features, _label) in enumerate(dataloader):
        if n_batches is not None and i >= n_batches:
            break
        model(features.to(device=device, dtype=dtype))
    model.eval()


def redraw_spread(base_model, dataset, eval_loader, loss_function, post,
                  device, dtype, n_draws=6, n_batches=None, batch_size=None,
                  num_workers=0, cumulative=False):
    pre_losses = []
    pst_losses = []
    for d in range(n_draws):
        logger.info(f"Running on redraw {d} of {n_draws}")
        model = copy.deepcopy(base_model)          # frozen weights, fresh buffers
        g = torch.Generator().manual_seed(1000 + d)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, generator=g,
        )
        reset_bn(model, cumulative=cumulative)
        recompute_bn(model, loader, device, dtype, n_batches=n_batches)
        pre, pst = evaluate(model, eval_loader, loss_function, post, device, dtype)
        pre_losses.append(pre)
        pst_losses.append(pst)
    return np.array(pre_losses), np.array(pst_losses)


class BatchNormCheckExecutor(BaseD2PSExecutor):
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

        # model = self.init_model()

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

        post = self.scale_ps.postprocess

        # Test 1: recompute buffers over full train, weights frozen
        logger.info("\n\n\nRunning Test 1...")
        model = self.init_model()
        with self.name_tracker.set_context("epoch", "best"):
            model_path = self.in_resume_model.path
            logger.info(f"Loaded model: {model_path}")
            record = self.in_resume_model.read(model=model, epoch="best")
        if isinstance(record, dict):
            logger.info(f"Loaded checkpoint: epoch={record.get('epoch')} "
                        f"best_loss={record.get('best_loss')}")
            
        # baseline: online EMA buffers as saved at "best" (epoch 75)
        model.eval()
        base_val_pre, base_val_pst = evaluate(model, valid_dataloader,       loss_function, post, self.device, self.dtype)
        base_trn_pre, base_trn_pst = evaluate(model, train_infer_dataloader, loss_function, post, self.device, self.dtype)

        # deployable buffers: recompute over full training data, weights frozen
        reset_bn(model, cumulative=True)
        recompute_bn(model, train_dataloader, self.device, self.dtype)
        rcmp_val_pre, rcmp_val_pst = evaluate(model, valid_dataloader,       loss_function, post, self.device, self.dtype)
        rcmp_trn_pre, rcmp_trn_pst = evaluate(model, train_infer_dataloader, loss_function, post, self.device, self.dtype)

        logger.info(
            "Test 1 (recompute buffers, weights frozen):\n"
            f"  val   pre : base={base_val_pre:.4e}  recomp={rcmp_val_pre:.4e}\n"
            f"  val   post: base={base_val_pst:.4e}  recomp={rcmp_val_pst:.4e}\n"
            f"  train pre : base={base_trn_pre:.4e}  recomp={rcmp_trn_pre:.4e}\n"
            f"  train post: base={base_trn_pst:.4e}  recomp={rcmp_trn_pst:.4e}"
        )

        # Test 2: buffer redraw spread (error bar; keep none of these models)
        logger.info("\n\n\nRunning Test 2...")
        base_model = self.init_model()
        with self.name_tracker.set_context("epoch", "best"):
            self.in_resume_model.read(model=base_model, epoch="best")
            model_path = self.in_resume_model.path
            logger.info(f"Loaded model: {model_path}")

        draws_pre, draws_pst = redraw_spread(
            base_model, train_dataset, valid_dataloader,
            loss_function, post, self.device, self.dtype,
            n_draws=6, n_batches=32, batch_size=self.batch_size,
            cumulative=True,          # population average, matches Test 1's estimator
        )
        logger.info(f"redraw val pre  per-draw: {np.array2string(draws_pre, precision=4)}")
        logger.info(f"redraw val post per-draw: {np.array2string(draws_pst, precision=4)}")
        logger.info(f"redraw val pre : mean={draws_pre.mean():.4e} std={draws_pre.std():.4e} "
                    f"min={draws_pre.min():.4e} max={draws_pre.max():.4e}")
        logger.info(f"redraw val post: mean={draws_pst.mean():.4e} std={draws_pst.std():.4e} "
                    f"min={draws_pst.min():.4e} max={draws_pst.max():.4e}")


        # Test 3: 
        logger.info("\n\n\nRunning Test 3...")
        model3 = self.init_model()
        with self.name_tracker.set_context("epoch", self.restart_epoch):
            model_path = self.in_resume_model.path
            logger.info(f"Loaded model: {model_path}")
            rec3 = self.in_resume_model.read(model=model3, epoch=self.restart_epoch)
        logger.info(f"Test 3 loaded checkpoint: epoch={rec3.get('epoch')}")

        # Give it good buffers first, so eval mode is not handicapped by
        # staleness -- we want the mode difference, not a buffer artifact.
        reset_bn(model3, cumulative=True)
        recompute_bn(model3, train_dataloader, self.device, self.dtype)
        rm_before, rv_before = bn_buffer_checksum(model3)

        # (a) eval mode: running stats (dataset-wide)
        ev_trn_pre, ev_trn_pst = evaluate(
            model3, train_infer_dataloader, loss_function, post,
            self.device, self.dtype)
        ev_val_pre, ev_val_pst = evaluate(
            model3, valid_dataloader, loss_function, post,
            self.device, self.dtype)

        # (b) train mode: per-batch stats, same weights, buffers untouched
        tm_trn_pre, tm_trn_pst = evaluate_train_mode(
            model3, train_infer_dataloader, loss_function, post,
            self.device, self.dtype)
        tm_val_pre, tm_val_pst = evaluate_train_mode(
            model3, valid_dataloader, loss_function, post,
            self.device, self.dtype)

        rm_after, rv_after = bn_buffer_checksum(model3)
        logger.info(
            f"buffer checksum before={rm_before:.6g}/{rv_before:.6g}  "
            f"after={rm_after:.6g}/{rv_after:.6g}  "
            f"(must match -- train-mode eval must not update buffers)"
        )

        logger.info(
            "Test 3 (same weights, recomputed buffers; only BN mode differs):\n"
            f"  train  eval-mode (running stats): {ev_trn_pre:.4e}\n"
            f"  train train-mode (batch stats)  : {tm_trn_pre:.4e}"
            f"   -> ratio {ev_trn_pre / tm_trn_pre:.2f}x\n"
            f"  valid  eval-mode (running stats): {ev_val_pre:.4e}\n"
            f"  valid train-mode (batch stats)  : {tm_val_pre:.4e}"
            f"   -> ratio {ev_val_pre / tm_val_pre:.2f}x"
        )

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


# =============================================================================
# Test 3: train-mode (batch stats) vs eval-mode (running stats)
#
# Same weights, same data, no parameter updates. The ONLY difference is which
# statistics normalize the activations. This isolates the functional gap
# between the two modes from every other confound:
#   - no within-epoch weight movement (weights are frozen)
#   - no stale buffers (recomputed first)
#   - same dataset, same batches, same order
#
# Add these two functions at module level, then the block inside execute().
# =============================================================================


@torch.no_grad()
def evaluate_train_mode(model, dataloader, loss_function, post, device, dtype):
    """
    Loss with BN normalizing by CURRENT-BATCH statistics, weights frozen.

    Mirrors how `Batch Loss` is computed during training, but at fixed weights.
    BN running stats would normally be updated by a train-mode forward pass, so
    momentum is forced to 0.0 for the duration to leave the buffers untouched.
    Dropout is kept OFF (model.eval()) so only the BN mode differs.
    """
    bns = _bn_modules(model)
    saved = [(m.momentum, m.training) for m in bns]

    model.eval()                 # dropout off
    for m in bns:
        m.momentum = 0.0         # train-mode forward, but do NOT touch buffers
        m.train()                # use batch statistics

    pre_loss = 0.0
    pst_loss = 0.0
    n_batches = 0
    try:
        for features, label in tqdm(dataloader, leave=False):
            features = features.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=dtype)
            output = model(features)
            pre_loss += loss_function(output, label).item()
            pst_loss += loss_function(post(output), post(label)).item()
            n_batches += 1
    finally:
        for m, (mom, was_training) in zip(bns, saved):
            m.momentum = mom
            m.train(was_training)
        model.eval()

    return pre_loss / n_batches, pst_loss / n_batches


@torch.no_grad()
def bn_buffer_checksum(model):
    """Summed norms of BN running stats -- to verify train-mode eval left them
    unchanged."""
    bns = _bn_modules(model)
    rm = sum(float(m.running_mean.norm()) for m in bns)
    rv = sum(float(m.running_var.norm()) for m in bns)
    return rm, rv
