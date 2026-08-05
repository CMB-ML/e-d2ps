import logging

import torch
from omegaconf import DictConfig

from d2ps_nn.stage_executors.F_train import TrainingExecutor
from d2ps_nn.models.attenuated_bn import set_bn_alpha, _AttenuatedBNMixin


logger = logging.getLogger(__name__)


class AttBNTrainingExecutor(TrainingExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.use_warmstart = cfg.use_warmstart
        self.warmstart_path = cfg.get("warmstart_path", None)
        self.bn_alpha = cfg.model.train.get("bn_alpha", 0.0)

    def prep_training_state(self, model, optimizer, scheduler):
        if not self.use_warmstart:
            logger.warning("Using warmstart executor without use warmstart!")
            return super().prep_training_state(model, optimizer, scheduler)

        logger.info(f"Warm-starting from '{self.warmstart_path}' with bn_alpha={self.bn_alpha:.3g}")

        if self.warmstart_path is None: 
            raise ValueError("Must provide a warmstart path.")
        
        warmstart_path = self.name_tracker.path(path_template=self.warmstart_path)
        self.in_resume_model.handler.read(path=warmstart_path, model=model, strict=False)

        set_bn_alpha(model, self.bn_alpha)
        self._audit_norms(model)

        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")

        is_continuation, start_epoch, initial_best_loss = False, 0, None

        return is_continuation, start_epoch, initial_best_loss

    def _audit_norms(model):
        n_abn = sum(1 for m in model.modules() if isinstance(m, _AttenuatedBNMixin))
        n_bn  = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, _AttenuatedBNMixin))
        logger.info(f"norm audit: {n_abn} attenuated BN (alpha={[m.alpha for m in model.modules() if isinstance(m, _AttenuatedBNMixin)][:3]}), {n_bn} plain BN")
