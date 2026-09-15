from dataclasses import dataclass
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import NpyHandler, MPLFigure
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.split_metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    working_dir: str
    cleaning_model: str
    fig_model_name: str
    use_epoch: str


class BaseFigsExecutor(BaseStageExecutor):
    """
    Goal: Calculate metrics to be used in the paper. Some metrics depend on 
    more information and are currently disabled.
    """
    def __init__(self, cfg: DictConfig, stage_str="plot_base") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: MPLFigure
        self.in_pred: Asset = self.assets_in["ps_pred"]
        self.in_thry: Asset = self.assets_in["ps_thry"]
        self.in_per_ell_metric: Asset = self.assets_in["per_ell_metric"]
        in_pred_handler: NpyHandler
        in_thry_handler: NpyHandler
        in_metric_handler: NpyHandler

        self.models: dict[str: ModelConfig] = self.set_up_models(cfg.models)

        self.ells = parse_ells(cfg.model.use_ells)

    def set_up_models(self, cfg_models: DictConfig) -> dict[str, ModelConfig]:
        models = {}
        for model, model_cfg in cfg_models.items():
            models[model] = ModelConfig(
                working_dir=model_cfg["working_dir"],
                cleaning_model=model_cfg["cleaning_model"],
                fig_model_name=model_cfg["fig_model_name"],
                use_epoch=model_cfg["use_epoch"]
            )
        return models

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            thry = self.get_thry(split)

            # Load data
            model_data = {}
            for model, model_cfg in self.models.items():
                context = dict(split=split.name, epoch=model_cfg.use_epoch)
                with self.name_tracker.set_contexts(context):
                    model_data[model] = self.process(split, thry, model_cfg)
            # Make figure with data

    def process(self, split:Split, thry: np.ndarray, model: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
        pred = self.get_pred(split, model)
        metric = self.get_metric(split, model, metric_lbl="R2")
        logger.info(f"split: {split.name:<8}, model: {model.cleaning_model:<12} thry: {thry.shape}, pred: {pred.shape}, metric: {metric.shape}")
        return pred, metric

    def get_thry(self, split: Split) -> np.ndarray:
        with self.name_tracker.set_context("split", split.name):
            thry = self.in_thry.read()
        return thry

    def get_pred(self, 
                 split: Split, 
                 model: ModelConfig, 
                 ) -> np.ndarray:
        context = dict(
            split=split.name,
            working=model.working_dir,
        )
        with self.name_tracker.set_contexts(context):
            pred = self.in_pred.read()
        return pred

    def get_metric(self, 
                       split: Split, 
                       model: ModelConfig, 
                       metric_lbl: str) -> np.ndarray:
        context = dict(
            split=split.name,
            working=model.working_dir,
            metric=metric_lbl
        )
        with self.name_tracker.set_contexts(context):
            metric = self.in_per_ell_metric.read()
        return metric
