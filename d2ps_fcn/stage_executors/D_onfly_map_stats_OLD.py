import logging

import tqdm
import healpy as hp
import numpy as np
import torch
from omegaconf import DictConfig

from cmbml.core import Asset
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_fcn.dataset_otf import make_map_sampler


logger = logging.getLogger(__name__)


class GetDynProdMapStatsExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="get_dyn_map_stats") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]

        # Assets
        self.in_cmb_ps: Asset = self.assets_in["cmb_ps"]
        self.out_cmb_map_stats: Asset = self.assets_out["cmb_map"]
        self.out_cmb_scalar_stats: Asset = self.assets_out["cmb_scalar"]

        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.n_batches        = cfg.model.train.n_batches * 10
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.sht_engine       = str(cfg.sim_engine).lower()

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

        sampler = make_map_sampler(
            sim_engine=self.sht_engine,
            dl_bank=dl_bank,
            nside=self.nside,
            batch_size=self.train_batch_size,
            n_batches=self.n_batches,
            lmax=self.lmax,
            device=self.device,
            dtype=self.dtype,
        )

        logger.info(f"Producing statistics based on {self.train_batch_size * self.n_batches} maps.")

        npix = hp.nside2npix(self.nside)

        # Welford online accumulation in float64 on CPU for numerical stability
        min_map  = np.full(npix, np.inf,  dtype=np.float64)
        max_map  = np.full(npix, -np.inf, dtype=np.float64)
        mean_map = np.zeros(npix, dtype=np.float64)
        m2_map   = np.zeros(npix, dtype=np.float64)
        count = 0

        for maps, _ in tqdm.tqdm(sampler):
            # maps: (batch_size, npix) on GPU
            maps_np = maps.cpu().to(torch.float64).numpy()
            for map_row in maps_np:
                count += 1
                min_map  = np.minimum(min_map, map_row)
                max_map  = np.maximum(max_map, map_row)
                delta    = map_row - mean_map
                mean_map += delta / count
                delta2   = map_row - mean_map
                m2_map   += delta * delta2

        if count < 2:
            raise ValueError("Need at least two maps to compute sample std.")

        stdv_map = np.sqrt(m2_map / (count - 1))

        out_maps = np.stack([min_map, max_map, mean_map, stdv_map])  # (4, npix)
        self.out_cmb_map_stats.write(data=out_maps)

        scalar_stats = dict(
            vmin=min_map.min(),
            vmax=max_map.max(),
            mean=mean_map.mean(),
            stdv=np.sqrt(np.mean(stdv_map ** 2))
        )
        self.out_cmb_scalar_stats.write(data=scalar_stats)

        logger.info(
            f"Map stats computed over {count} maps "
            f"({self.n_batches} batches x {self.train_batch_size} batch size). "
            f"Output statistics shape: {out_maps.shape}"
        )
