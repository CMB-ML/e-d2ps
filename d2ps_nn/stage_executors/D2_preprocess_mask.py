import logging

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers import NumpyMap

from d2ps_nn.utils.numpy_dtype import get_np_dtype


logger = logging.getLogger(__name__)


class PreprocessMaskExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess_mask")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_mask: Asset = self.assets_out["mask"]
        out_mask_handler: NumpyMap

        # Handles either observations or ground-truth CMB maps
        self.in_mask = self.assets_in["mask"]
        self.in_lut: Asset = self.assets_in["lut"]
        in_lut_handler: NumpyMap

        self.scaling = cfg.model.get("scale_px", None)
        self.dtype = get_np_dtype(cfg.model.dtype)

        self.lut_style = cfg.model.patches.get("lut_style", None)
        self.lut = None
        self.dataset_stats = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        self.load_lut()
        mask = self.in_mask.read(precision=self.dtype)[0]
        out = mask[self.lut]
        context = dict(px_arrange=self.lut_style)
        with self.name_tracker.set_contexts(context):
            self.out_mask.write(data=out)

    def load_lut(self) -> None:
        with self.name_tracker.set_context("style", self.lut_style):
            self.lut = self.in_lut.read()
