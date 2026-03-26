import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from d2ps_nn.models import BaseD2PSExecutor
from d2ps_nn.stage_executors.F_train import get_loss
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset


logger = logging.getLogger(__name__)


class CheckGPULoadExecutor(BaseD2PSExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.in_features = FeatureAsset.select(self.assets_in, 
                                               cfg.features_type, 
                                               self.instrument)
        self.batch_size    = cfg.model.train.batch_size
        self.loss_key      = cfg.model.train.loss

    # def execute(self) -> None:
    #     logger.debug(f"Running {self.__class__.__name__} execute()")

    #     # Set up model and loss function
    #     model = self.init_model()
    #     loss_function = get_loss(self.loss_key, self.ells)
    #     loss_function = loss_function.to(self.device)

    #     # Make random tensor to check model size
    #     input_shape = (self.batch_size, *model.example_in_shape)
    #     output_shape = (self.batch_size, len(self.ells))
    #     train_features = torch.randn(input_shape).to(self.device)
    #     train_label = torch.randn(output_shape).to(self.device)
    #     logger.info(f"{train_features.shape}, {train_label.shape}")

    #     # Check model size
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    #     output = model(train_features)
    #     logger.info(f"Output shape: {output.shape}")

    #     logger.info(f"Memory after forward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    #     loss = loss_function(output, train_label)
    #     loss.backward()

    #     logger.info(f"Memory after backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise RuntimeError("GPU VRAM checker requires a CUDA device.")

        # Set up model and loss function
        model = self.init_model().to(self.device)
        model.train()

        loss_function = get_loss(self.loss_key, self.ells).to(self.device)

        use_amp = getattr(self, "use_amp", True)
        amp_dtype = torch.float16   # or torch.bfloat16 if you prefer stability
        scaler = GradScaler("cuda", enabled=use_amp)

        # Make random tensors to check model size
        input_shape = (self.batch_size, *model.example_in_shape)
        output_shape = (self.batch_size, len(self.ells))

        train_features = torch.randn(
            input_shape, device=self.device, dtype=torch.float32
        )
        train_label = torch.randn(
            output_shape, device=self.device, dtype=torch.float32
        )

        logger.info(f"Input shape : {train_features.shape}")
        logger.info(f"Label shape : {train_label.shape}")

        # Warm-up pass can help stabilize CUDA kernel/caching effects
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)

        # Reset stats right before the measured region
        torch.cuda.reset_peak_memory_stats(self.device)

        # Forward
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            output = model(train_features)
            loss = loss_function(output, train_label)

        logger.info(f"Output shape: {output.shape}")
        logger.info(
            f"Peak memory after forward: "
            f"{torch.cuda.max_memory_allocated(self.device) / 1024**2:.2f} MB"
        )

        # Backward
        scaler.scale(loss).backward()

        logger.info(
            f"Peak memory after backward: "
            f"{torch.cuda.max_memory_allocated(self.device) / 1024**2:.2f} MB"
        )

        logger.info(
            f"Peak reserved memory: "
            f"{torch.cuda.max_memory_reserved(self.device) / 1024**2:.2f} MB"
        )

        # Optional cleanup
        model.zero_grad(set_to_none=True)
        del output, loss, train_features, train_label
        torch.cuda.empty_cache()