import logging

import math
import tqdm
import healpy as hp
import numpy as np
import torch
from omegaconf import DictConfig

from cmbml.core import Asset
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_fcn.dyn_prod_dataset import make_map_sampler
from d2ps_fcn.map2ps_batched import make_spectra_estimator


logger = logging.getLogger(__name__)


class GetDynValidMapsExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="dyn_bundle") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]

        # Assets
        self.in_cmb_ps: Asset = self.assets_in["cmb_ps"]
        self.out_thry_bundle: Asset = self.assets_out["ps_thry"]
        self.out_real_bundle: Asset = self.assets_out["ps_real"]
        self.out_map_bundle: Asset = self.assets_out["cmb_map"]

        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.n_maps           = cfg.splits.Valid.n_sims
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

        n_batches = self.n_maps // self.train_batch_size
        if self.n_maps % self.train_batch_size != 0:
            n_batches +=1

        sampler = make_map_sampler(
            sim_engine=self.sht_engine,
            dl_bank=dl_bank,
            nside=self.nside,
            batch_size=self.train_batch_size,
            n_batches=n_batches,
            lmax=self.lmax,
            device=self.device,
            dtype=self.dtype,
        )

        ps_maker = make_spectra_estimator(
            engine=self.sht_engine,
            nside=self.nside,
            lmax=self.lmax,
            return_cl=False,
            return_dl=True,
            compute_device=self.device,
            output_device="cpu"
        )

        npix = hp.nside2npix(self.nside)

        bundle_maps = np.zeros((self.n_maps, npix), dtype=np.float32)
        bundle_thry = np.zeros((self.n_maps, len(self.ells)), dtype=np.float32)
        bundle_real = np.zeros((self.n_maps, len(self.ells)), dtype=np.float32)
        count = 0

        for maps, thrys in tqdm.tqdm(sampler):
            # maps: (batch_size, npix) on GPU
            maps_np = maps.cpu().to(torch.float32).numpy()
            thrys = thrys.cpu().to(torch.float32).numpy()
            reals = ps_maker(maps)['dl'].numpy()

            for i, map_row in enumerate(maps_np):
                bundle_maps[count] = map_row
                bundle_thry[count] = thrys[i, self.ells]
                bundle_real[count] = reals[i, self.ells]
                count += 1
                if count >= self.n_maps:
                    break

        split = self.splits.get("valid")
        with self.name_tracker.set_context("split", split.name):
            self.out_map_bundle.write(data=bundle_maps)
            self.out_thry_bundle.write(data=bundle_thry)
            self.out_real_bundle.write(data=bundle_real)
