import logging

import torch
from omegaconf import DictConfig

from d2ps_nn.stage_executors.F_train import TrainingExecutor, get_loss, evaluate
from d2ps_nn.models.attenuated_bn import set_bn_alpha, _AttenuatedBNMixin


logger = logging.getLogger(__name__)


class AttBNTrainingExecutor(TrainingExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.use_warmstart = cfg.use_warmstart
        self.warmstart_path = cfg.get("warmstart_path", None)
        self.bn_alpha = cfg.model.train.get("bn_alpha", 0.0)
        self.freeze_bn_affine = cfg.model.train.get("freeze_bn_affine", False)

        raw = cfg.model.train.get("bn_momentum", 0.1)
        if raw is None or (isinstance(raw, str) and raw.strip().lower() == "none"):
            self.bn_momentum = None          # PyTorch cumulative average
        else:
            self.bn_momentum = float(raw)

    def _prepare_model(self, model):
        set_bn_alpha(model, self.bn_alpha)

        for m in model.modules():
            # if isinstance(m, _AttenuatedBNMixin):
            #     m.momentum = self.bn_momentum
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.momentum = self.bn_momentum
        logger.info(f"set bn momentum to {self.bn_momentum!r} (None = cumulative) "
                    f"on attenuated BN layers")

        if self.freeze_bn_affine:
            n = 0
            for m in model.modules():
                if isinstance(m, _AttenuatedBNMixin):
                    if m.weight is not None:
                        m.weight.requires_grad_(False)
                    if m.bias is not None:
                        m.bias.requires_grad_(False)
                    n += 1
            logger.info(f"froze affine (gamma/beta) on {n} attenuated BN layers")
        self._audit_norms(model)
        self._log_bn_stat_norms(model, "at init, pre-load")        

    def prep_training_state(self, model, optimizer, scheduler):
        if not self.use_warmstart:
            logger.warning("Using warmstart executor without use warmstart!")
            return super().prep_training_state(model, optimizer, scheduler)

        logger.info(f"Warm-starting from '{self.warmstart_path}' with bn_alpha={self.bn_alpha:.3g}")

        if self.warmstart_path is None: 
            raise ValueError("Must provide a warmstart path.")
        
        warmstart_path = self.name_tracker.path(path_template=self.warmstart_path)
        self.in_resume_model.handler.read(path=warmstart_path, model=model, strict=False)
        self._log_bn_stat_norms(model, "after warmstart load")

        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")

        initial_best_loss = self._warmstart_gate(model)

        is_continuation, start_epoch = False, 0

        return is_continuation, start_epoch, initial_best_loss

    def _warmstart_gate(self, model):
        loss_function = get_loss(self.loss_key, self.ells).to(self.device)
        valid_dl = self._make_dataloader(
            self.set_up_dataset(self.splits.get("valid")), shuffle=False, label="gate")
        pre, _ = evaluate(model=model, dataloader=valid_dl, loss_function=loss_function,
                          post=self.scale_ps.postprocess, device=self.device, dtype=self.dtype)
        logger.info(f"WARM-START GATE pre-train valid_pre: {pre:.02e}")
        return pre

    def _audit_norms(self, model):
        abns = [m for m in model.modules() if isinstance(m, _AttenuatedBNMixin)]
        n_bn  = sum(1 for m in model.modules()
                if isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, _AttenuatedBNMixin))
        logger.info(
            f"norm audit: {len(abns)} attenuated BN "
            f"(alpha={[m.alpha for m in abns][:3]}, momentum={[m.momentum for m in abns][:3]}), "
            f"{n_bn} plain BN"
        )

    def _log_bn_stat_norms(self, model, tag):
        abns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not abns:
            return
        rm = sum(float(m.running_mean.norm()) for m in abns)
        rv = sum(float(m.running_var.norm()) for m in abns)
        logger.info(f"BN stat norms [{tag}]: sum|running_mean|={rm:.6g}  sum|running_var|={rv:.6g}")

    def record_and_checkpoint(self, model, optimizer, epoch, tm, vm, best_valid_loss):
        if (epoch + 1) % 10 == 0:
            self._log_bn_stat_norms(model, f"epoch {epoch + 1}")
        super().record_and_checkpoint(model, optimizer, epoch, tm, vm, best_valid_loss)
