import logging
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


class GetDynProdStatsExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="get_dyn_stats") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]
        # Assets
        self.in_thry: Asset            = self.assets_in["cmb_ps"]
        self.out_map_local_stats : Asset = self.assets_out["map_local"]
        self.out_real_local_stats  : Asset = self.assets_out["rps_local"]
        self.out_map_global_stats: Asset = self.assets_out["map_global"]
        self.out_real_global_stats : Asset = self.assets_out["rps_global"]
        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.n_batches        = cfg.model.train.n_batches * 10
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.sht_engine       = str(cfg.sim_engine).lower()

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.choose_device()

        dl_bank = self.in_thry.read()
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

        spectra_estimator = make_spectra_estimator(
            engine=self.sht_engine,
            nside=self.nside,
            lmax=self.lmax,
            return_cl=True,
            return_dl=False,
            compute_device=str(self.device),
            output_device="cpu",
            dtype=torch.float64,
        )

        n_maps = self.train_batch_size * self.n_batches
        logger.info(f"Producing statistics based on {n_maps} maps.")

        npix  = hp.nside2npix(self.nside)
        nell  = self.lmax + 1

        # --- per-pixel Welford accumulators ---
        min_map  = np.full(npix, np.inf,  dtype=np.float64)
        max_map  = np.full(npix, -np.inf, dtype=np.float64)
        mean_map = np.zeros(npix, dtype=np.float64)
        m2_map   = np.zeros(npix, dtype=np.float64)

        # --- per-ell Welford accumulators ---
        min_dl  = np.full(nell, np.inf,  dtype=np.float64)
        max_dl  = np.full(nell, -np.inf, dtype=np.float64)
        mean_dl = np.zeros(nell, dtype=np.float64)
        m2_dl   = np.zeros(nell, dtype=np.float64)

        count = 0
        for maps, _ in tqdm.tqdm(sampler):
            # maps: (batch_size, npix) on GPU
            cl_batch = spectra_estimator(maps)["cl"].numpy()  # (B, lmax+1), float64, CPU
            maps_np  = maps.cpu().to(torch.float64).numpy()   # (B, npix)

            for map_row, cl_row in zip(maps_np, cl_batch):
                count += 1

                # pixel stats
                min_map  = np.minimum(min_map, map_row)
                max_map  = np.maximum(max_map, map_row)
                delta    = map_row - mean_map
                mean_map += delta / count
                m2_map   += delta * (map_row - mean_map)

                # spectra stats
                min_dl  = np.minimum(min_dl, cl_row)
                max_dl  = np.maximum(max_dl, cl_row)
                delta   = cl_row - mean_dl
                mean_dl += delta / count
                m2_dl   += delta * (cl_row - mean_dl)

        if count < 2:
            raise ValueError("Need at least two maps to compute sample std.")

        stdv_map = np.sqrt(m2_map / (count - 1))
        stdv_dl  = np.sqrt(m2_dl  / (count - 1))

        # write pixel stats
        out_maps = np.stack([min_map, max_map, mean_map, stdv_map])  # (4, npix)
        self.out_map_local_stats.write(data=out_maps)

        # write spectra stats
        out_spectra = np.stack([min_dl, max_dl, mean_dl, stdv_dl])   # (4, lmax+1)
        self.out_real_local_stats.write(data=out_spectra)

        map_scalar_stats = dict(
            # pixel stats (existing)
            vmin=float(min_map.min()),
            vmax=float(max_map.max()),
            mean=float(mean_map.mean()),
            stdv=float(np.sqrt(np.mean(stdv_map ** 2)))
        )
        self.out_map_global_stats.write(data=map_scalar_stats)

        ps_scalar_stats = dict(
            # spectra scalars
            dl_vmin=float(min_dl.min()),
            dl_vmax=float(max_dl.max()),
            dl_mean=float(mean_dl.mean()),
            dl_stdv=float(np.sqrt(np.mean(stdv_dl ** 2)))
        )
        self.out_real_global_stats.write(data=ps_scalar_stats)

        logger.info(
            f"Map stats computed over {count} maps "
            f"({self.n_batches} batches x {self.train_batch_size} batch size). "
            f"Pixel stats shape: {out_maps.shape}, "
            f"Spectra stats shape: {out_spectra.shape}"
        )
