import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import NpyHandler
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.split_metrics import compute_metrics

logger = logging.getLogger(__name__)


class GetMetricsExecutor(BaseStageExecutor):
    """
    Goal: Calculate metrics to be used in the paper. Some metrics depend on 
    more information and are currently disabled.
    """
    def __init__(self, cfg: DictConfig, stage_str="generate_metrics") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_metric: Asset = self.assets_out["per_ell_metric"]
        out_metric_handler: NpyHandler
        self.in_pred: Asset = self.assets_in["pred"]
        self.in_thry: Asset = self.assets_in["thry"]
        in_pred_handler: NpyHandler
        in_thry_handler: NpyHandler

        self.ells = parse_ells(cfg.model.use_ells)
        self.spectra_kind = ""

        self.include_metrics = cfg.use_metrics

        self.sky_fraction = None
        if self.sky_fraction is not None:
            raise NotImplementedError("The default: 1, assumes no mask. We're using a mask. There's big questions here to address.")

        for banned_metric in ["cosmic_var", "treble_cosmic_var", "e_cv"]:
            if banned_metric in self.include_metrics:
                # There's an open question about how to use the mask and get the proper fsky for mode counting.
                raise NotImplementedError("These metrics depend on sky fraction. fsky needs to be handled.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            for epoch in self.model_epochs:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)
            logger.info(f"Done getting metrics for {split.name}.")

    def process(self, split: Split) -> None:
        pred = self.in_pred.read()
        thry = self.in_thry.read()

        fsky = 1 if self.sky_fraction is None else self.sky_fraction
        metrics = compute_metrics(pred, thry, ells=self.ells, fsky=fsky)

        for m in self.include_metrics:
            with self.name_tracker.set_context("metric", m):
                self.out_metric.write(data=metrics[m])
