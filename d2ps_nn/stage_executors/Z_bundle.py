import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import TextPowerSpectrum, NumpyPowerSpectrum, DictCambPowerSpectrum
from d2ps_nn.utils.get_epochs import get_epochs_with_best
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class BundleExecutor(BaseStageExecutor):
    """
    Goal: Bundle together power spectra for external analysis.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_ps: Asset = self.assets_out["ps_bundle"]
        self.in_ps: Asset = self.assets_in["ps"]

        self.ells = parse_ells(cfg.model.use_ells)
        this_pipe = cfg.pipeline.get(stage_str)

        self.spectra_kind = ""

        # Brittle, not a fan of this solution...
        self.select_ell = this_pipe.assets_in.ps.stage != "predict"
        pass

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        epoch_iter = [None] if self.model_epochs is None else self.model_epochs

        for split in self.splits:
            for epoch in epoch_iter:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)
            logger.info(f"Done bundling {split.name} {self.spectra_kind} spectra.")

    def process(self, split: Split) -> None:
        data = []
        target_ells = np.asarray(self.ells)

        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                handler = self.in_ps.handler
                if isinstance(handler, DictCambPowerSpectrum):
                    tt = read_dict_camb_ps(self.in_ps, target_ells, sim)
                elif isinstance(handler, NumpyPowerSpectrum):
                    tt = read_npy_ps(self.in_ps, target_ells)
                else:
                    raise NotImplementedError(f"Can only use DictCambPowerSpectrum and NumpyPowerSpectrum, got {handler}")
            data.append(tt)

        data = np.stack(data, axis=0)
        self.out_ps.write(data=data)

def read_dict_camb_ps(asset, ells, sim):
    ps = asset.read(fields=["TT", "L"])
    tt = trim_tt_to_target_ells(
        ps,
        ells,
        label=f"Spectrum for sim {sim}",
    )
    return tt

def read_npy_ps(asset, ells):
    tt = asset.read()
    tt = tt[ells]
    return tt

class BundlePredExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="bundle_pred")
        self.spectra_kind = "prediction"

class BundleTrueExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="bundle_true")
        self.spectra_kind = "ground truth"

    def process(self, split: Split) -> None:
        if split.ps_fidu_fixed:
            spectrum = self.in_ps.read(
                use_alt_path=split.ps_fidu_fixed,
                fields=["TT", "L"],
            )

            tt = trim_tt_to_target_ells(
                spectrum,
                np.asarray(self.ells),
                label=f"Fixed fiducial spectrum {split.ps_fidu_fixed}",
            )

            data = np.broadcast_to(tt, (split.n_sims, tt.shape[0])).copy()

            self.out_ps.write(data=data)
            return

        super().process(split)

class BundleRealExecutor(BundleExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="bundle_real")
        self.spectra_kind = "realization"

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            context = dict(split=split.name)
            with self.name_tracker.set_contexts(context):
                self.process(split)
            logger.info(f"Done bundling {split.name} {self.spectra_kind} spectra.")


def trim_tt_to_target_ells(
    spectrum: dict,
    target_ells: np.ndarray,
    *,
    label: str = "",
) -> np.ndarray:
    ell = np.asarray(spectrum["L"])
    tt = np.asarray(spectrum["TT"])

    ell_to_idx = {int(l): i for i, l in enumerate(ell)}
    missing = [int(l) for l in target_ells if int(l) not in ell_to_idx]

    if missing:
        raise ValueError(
            f"{label} is missing required ells: {missing}"
        )

    idx = [ell_to_idx[int(l)] for l in target_ells]
    return tt[idx]
