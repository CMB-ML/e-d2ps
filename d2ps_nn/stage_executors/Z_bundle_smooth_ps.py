import logging

import numpy as np
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum, DictCambPowerSpectrum
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.polyfit_tools import setup_skl_pipeline, scale_ell


logger = logging.getLogger(__name__)


class BundleSmoothExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str="bundle_smooth") -> None:
        super().__init__(cfg, stage_str)
        self.out_ps: Asset = self.assets_out["ps_bundle"]
        self.in_ps: Asset = self.assets_in["pred"]

        self.ells = parse_ells(cfg.model.use_ells)

        self.spectra_kind = ""
        self.basis_kind = cfg.sm_basis_kind
        self.degree = cfg.sm_degree
        self.skl_pipeline = None
        self.scaled_ell_range = tuple(cfg.out_range)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        epoch_iter = [None] if self.model_epochs is None else self.model_epochs
        self.skl_pipeline = setup_skl_pipeline(self.basis_kind, self.degree)
        self.prepped_ell = scale_ell(self.ells, out_range=self.scaled_ell_range)

        for split in self.splits:
            for epoch in epoch_iter:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)
            logger.info(f"Done bundling {split.name} {self.spectra_kind} spectra.")

    def process(self, split: Split) -> None:
        data = self.in_ps.read()
        data = np.atleast_2d(data)

        sm = []
        for i in range(data.shape[0]):
            sm.append(self.smooth(data[i]))

        result = np.array(sm)
        self.out_ps.write(data=result)

    def smooth(self, cmb_ps) -> np.ndarray:
        fit_model: Pipeline = self.skl_pipeline.fit(self.prepped_ell, cmb_ps)
        pred = fit_model.predict(self.prepped_ell)
        return pred


class BundleSmoothTrueExecutor(BundleSmoothExecutor):
    def __init__(self, cfg: DictConfig, stage_str="bundle_smooth_true") -> None:
        super().__init__(cfg, stage_str)
