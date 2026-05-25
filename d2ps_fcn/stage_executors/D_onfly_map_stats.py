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
from d2ps_fcn.realization_spectra import make_real_spectra_maker
from d2ps_nn.utils.stat_convention import STAT_MIN, STAT_MAX, STAT_MEAN, STAT_STDV


logger = logging.getLogger(__name__)


class GetDynProdStatsExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="get_dyn_stats") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]
        # Assets
        self.in_thry: Asset              = self.assets_in["cmb_ps"]
        self.out_map_local_stats : Asset = self.assets_out["map_local"]
        self.out_rps_local_stats : Asset = self.assets_out["rps_local"]
        # self.out_map_global_stats: Asset = self.assets_out["map_global"]
        # self.out_rps_global_stats: Asset = self.assets_out["rps_global"]
        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.n_batches        = cfg.model.train.n_batches * 10
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.sht_engine       = str(cfg.sim_engine).lower()

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.choose_device()

        tdl_bank = self.in_thry.read()
        if tdl_bank.shape[1] < self.lmax + 1:
            raise ValueError(
                "Dl bank does not have enough ell. "
                "Please regenerate it or change lmax."
            )
        tdl_bank = tdl_bank[:, :self.lmax + 1]

        sampler = make_map_sampler(
            sim_engine=self.sht_engine,
            dl_bank=tdl_bank,
            nside=self.nside,
            batch_size=self.train_batch_size,
            n_batches=self.n_batches,
            lmax=self.lmax,
            device=self.device,
            dtype=self.dtype,
        )

        spectra_estimator = make_real_spectra_maker(
            engine=self.sht_engine,
            nside=self.nside,
            lmax=self.lmax,
            return_cl=False,
            return_dl=True,
            compute_device=str(self.device),
            output_device="cpu",
            dtype=torch.float64,
        )

        n_maps = self.train_batch_size * self.n_batches
        logger.info(f"Producing statistics based on {n_maps} maps.")

        n_pix = hp.nside2npix(self.nside)
        n_ell = self.lmax + 1

        # --- per-pixel running min/max + Welford mean/variance accumulators ---
        min_map  = np.full(n_pix,  np.inf, dtype=np.float64)
        max_map  = np.full(n_pix, -np.inf, dtype=np.float64)
        mean_map = np.zeros(n_pix, dtype=np.float64)
        m2_map   = np.zeros(n_pix, dtype=np.float64)

        # --- per-ell running min/max + Welford mean/variance accumulators ---
        min_rdl  = np.full(n_ell,  np.inf, dtype=np.float64)
        max_rdl  = np.full(n_ell, -np.inf, dtype=np.float64)
        mean_rdl = np.zeros(n_ell, dtype=np.float64)
        m2_rdl   = np.zeros(n_ell, dtype=np.float64)

        count = 0
        for maps, _ in tqdm.tqdm(sampler):
            # maps: (batch_size, npix) on GPU
            rdl_batch = spectra_estimator(maps)["dl"].numpy()  # (B, lmax+1), float64, CPU
            maps_np   = maps.cpu().to(torch.float64).numpy()   # (B, npix)

            for map_row, rdl_row in zip(maps_np, rdl_batch):
                count += 1

                # pixel stats
                min_map  = np.minimum(min_map, map_row)
                max_map  = np.maximum(max_map, map_row)
                delta_m  = map_row - mean_map
                mean_map += delta_m / count
                m2_map   += delta_m * (map_row - mean_map)

                # spectra stats
                min_rdl   = np.minimum(min_rdl, rdl_row)
                max_rdl   = np.maximum(max_rdl, rdl_row)
                delta_rdl = rdl_row - mean_rdl
                mean_rdl  += delta_rdl / count
                m2_rdl    += delta_rdl * (rdl_row - mean_rdl)

        if count < 2:
            raise ValueError("Need at least two maps to compute sample std.")

        stdv_map = np.sqrt(m2_map / (count - 1))
        stdv_rdl = np.sqrt(m2_rdl / (count - 1))

        # write *local* pixel stats
        out_maps = np.empty((4, n_pix))
        out_maps[STAT_MIN] = min_map
        out_maps[STAT_MAX] = max_map
        out_maps[STAT_MEAN] = mean_map
        out_maps[STAT_STDV] = stdv_map
        self.out_map_local_stats.write(data=out_maps)

        # write *local* spectra stats
        out_rps = np.empty((4, self.lmax+1))
        out_rps[STAT_MIN] = min_rdl
        out_rps[STAT_MAX] = max_rdl
        out_rps[STAT_MEAN] = mean_rdl
        out_rps[STAT_STDV] = stdv_rdl
        self.out_rps_local_stats.write(data=out_rps)

        # Global stats are derived from local stats

        # Global stdv via law of total variance: within-element variance
        # (mean of per-element variances) + between-element variance (variance
        # of per-element means). RMS of per-element stdvs alone drops the
        # between term. Exact given equal counts per element (true here).
        
        # Turns out, global stats don't depend on M2. 
        #    This was an artifact of my confusion. Instead 
        #    I'll use the global stats everywhere and get the local stats from those.

        # map_within  = np.mean(stdv_map ** 2)   # average per-pixel variance
        # map_between = np.var(mean_map)         # variance of per-pixel means (ddof=0)
        # map_scalar_stats = dict(
        #     vmin=float(min_map.min()),
        #     vmax=float(max_map.max()),
        #     mean=float(mean_map.mean()),
        #     stdv=float(np.sqrt(map_within + map_between)),
        # )
        # self.out_map_global_stats.write(data=map_scalar_stats)

        # rdl_within  = np.mean(stdv_rdl ** 2)   # average per-ell variance
        # rdl_between = np.var(mean_rdl)          # variance of per-ell means (ddof=0)
        # rps_scalar_stats = dict(
        #     dl_vmin=float(min_rdl.min()),
        #     dl_vmax=float(max_rdl.max()),
        #     dl_mean=float(mean_rdl.mean()),
        #     dl_stdv=float(np.sqrt(rdl_within + rdl_between)),
        # )
        # self.out_rps_global_stats.write(data=rps_scalar_stats)

        logger.info(
            f"Map stats computed over {count} maps "
            f"({self.n_batches} batches x {self.train_batch_size} batch size). "
            f"Pixel stats shape: {out_maps.shape}, "
            f"Spectra stats shape: {out_rps.shape}"
        )
