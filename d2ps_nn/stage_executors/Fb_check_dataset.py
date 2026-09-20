import logging

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch

from d2ps_nn.stage_executors.F_train import TrainingExecutor


logger = logging.getLogger(__name__)


class CheckDatasetExecutor(TrainingExecutor):
    """
    Output: log report only (no files written, no model built).
    Builds the same datasets as TrainingExecutor and checks masking.

    Uses the 'train' stage config so the datasets are constructed
    from exactly the assets training will use.
    """
    def __init__(self, cfg: DictConfig, stage_str="train") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.n_check = 3

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        mask = self.get_mask()
        if mask is None:
            logger.warning("Masking is off; reporting shapes only.")

        n_local = self.batch_size  # sims 0..n_local-1 must exist locally for each split
        for label in ("train", "valid"):
            ds = self.set_up_dataset(self.splits.get(label), early_stop=n_local)
            dl = self._make_dataloader(ds, shuffle=False, label=f"check {label}")

            attached = getattr(ds, "mask", None)
            same = attached is not None and mask is not None and torch.equal(attached, torch.as_tensor(mask))
            logger.info(
                f"{label}: {type(ds).__name__}, {len(ds)} sims, "
                f"mask attached: {attached is not None} (matches stacked mask: {same}), "
                f"pos enc attached: {getattr(ds, 'pos_enc_lut', None) is not None}"
            )

            x_batch, y_batch = next(iter(dl))
            # m = x_batch[0][0].cpu().numpy()
            # plt.imshow(m)
            # plt.show()
            # exit()

            for b in range(min(self.n_check, x_batch.shape[0])):
                self.report_item(f"{label} batch0[{b}]", x_batch[b], y_batch[b], mask, ds.cat_axis)

        if self.cf_planck and not self.use_pos_enc_lut:
            self.set_up_planck()
            self.report_item("planck", self.pl_obs[0], None, mask, cat_axis=0)

    def report_item(self, tag, x, y, mask, cat_axis) -> None:
        n_dets = len(self.instrument.dets)
        expected = n_dets + (3 if self.use_pos_enc_lut else 0)
        label_str = f", label {tuple(y.shape)}" if y is not None else ""
        logger.info(f"{tag}: features {tuple(x.shape)} {x.dtype}{label_str}")

        xs = torch.as_tensor(x).movedim(cat_axis, 0)  # channels first, either layout
        if xs.shape[0] != expected:
            logger.error(f"  {tag}: {xs.shape[0]} channels, expected {expected}")
        if mask is None:
            return

        ms = torch.as_tensor(mask).movedim(cat_axis, 0).bool()
        for i, ch in enumerate(self.instrument.dets):
            xi, mi = xs[i], ms[i]
            leak = int((xi[~mi] != 0).sum())
            zeros_in_sky = float((xi[mi] == 0).float().mean())
            logger.info(
                f"  {ch} GHz: mask f_sky {mi.float().mean():.3f} | "
                f"nonzero in masked region: {leak} | "
                f"zeros in unmasked region: {zeros_in_sky:.4f}"
            )

        if self.use_pos_enc_lut:
            masked_any = ~ms.all(dim=0)  # pixels masked in at least one channel
            survives = bool((xs[n_dets:][:, masked_any] != 0).any())
            logger.info(f"  positional encoding present in masked pixels: {survives}")
