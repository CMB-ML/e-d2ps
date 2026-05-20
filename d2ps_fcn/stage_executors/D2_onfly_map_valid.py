import logging

import tqdm
import healpy as hp
import numpy as np
import torch
from omegaconf import DictConfig

from cmbml.core import Asset
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_fcn.dyn_prod_dataset import MapSampler


logger = logging.getLogger(__name__)


class GetDynValidMapsExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="bundle_maps") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]

        # Assets
        self.in_cmb_ps: Asset = self.assets_in["cmb_ps"]
        self.out_cmb_map_stats: Asset = self.assets_out["cmb_map"]
        self.out_bundle: Asset = self.assets_out["cmb_scalar"]

        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.n_maps           = cfg.splits.Valid.n_sims

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.choose_device()

        dl_bank = self.in_cmb_ps.read()
        if dl_bank.shape[1] < self.lmax + 1:
            raise ValueError(
                "Dl bank does not have enough ell. "
                "Please regenerate it or change lmax."
            )
        dl_bank = dl_bank[:, :self.lmax + 1]

        n_batches = self.n_maps // self.train_batch_size + 1

        sampler = MapSampler(
            dl_bank=dl_bank,
            nside=self.nside,
            lmax=self.lmax,
            batch_size=self.train_batch_size,
            n_batches=n_batches,
            device=self.device,
            dtype=self.dtype
        )

        npix = hp.nside2npix(self.nside)

        bundle = np.zeros(self.n_maps, npix, dtype=np.float64)
        count = 0

        for maps, _ in tqdm.tqdm(sampler):
            # maps: (batch_size, npix) on GPU
            maps_np = maps.cpu().to(torch.float64).numpy()
            for map_row in maps_np:
                bundle[count] = map_row
                count += 1
                if count > self.n_maps:
                    break

        self.out_bundle.write(data=bundle)
