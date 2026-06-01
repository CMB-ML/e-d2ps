from typing import Optional, Any
import logging

import healpy as hp
import numpy as np
import torch
from d2ps_fcn.scaler import ArrayScaler
from d2ps_nn.utils.px_scale import MapProcessor
from d2ps_nn.utils.px_shape import MapShaper

try:
    from cuhpx import iSHTCUDA
    _CUHPX_AVAILABLE = True
except ImportError:
    _CUHPX_AVAILABLE = False

logger = logging.getLogger(__name__)


def dl_to_cl_torch(dl: torch.Tensor) -> torch.Tensor:
    ell = torch.arange(dl.shape[-1], device=dl.device, dtype=dl.dtype)
    factor = ell * (ell + 1.0) / (2.0 * torch.pi)
    cl = torch.zeros_like(dl)
    good = factor > 0
    cl[..., good] = dl[..., good] / factor[good]
    return cl


class MapSampler:
    """
    Abstract base class for map samplers.

    Subclasses implement sample() to yield:
      - maps: (batch_size, npix) tensor
      - dl:   (batch_size, lmax+1) tensor, raw Dls for ells 0..lmax
    """
    def __init__(
        self,
        dl_bank: np.ndarray,
        nside: int,
        lmax: int,
        batch_size: int,
        n_batches: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.dl_bank = dl_bank
        self.nside = nside
        self.lmax = lmax
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.sample()

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MapSamplerCUHPX(MapSampler):
    """
    Generates batches of (maps, dl) on GPU from a power spectra bank.

    Owns the full Dl -> Cl -> alms -> maps pipeline.

    lmax is the highest ell used (ells 0..lmax inclusive).
    cuHPX requires alm arrays of size lmax+1; handled internally.

    Raises ImportError at construction time if cuHPX is not installed.
    """
    def __init__(self, *args, **kwargs):
        if not _CUHPX_AVAILABLE:
            raise ImportError(
                "cuHPX is not installed. Install it or use MapSamplerSYNFAST instead."
            )
        super().__init__(*args, **kwargs)

        alm_size = self.lmax + 1
        ell = torch.arange(alm_size, device=self.device)[:, None]
        m = torch.arange(alm_size, device=self.device)[None, :]
        self.tri_mask = m <= ell
        self.isht = iSHTCUDA(self.nside, lmax=alm_size, mmax=alm_size)

    def _draw_alms(self, cl: torch.Tensor) -> torch.Tensor:
        B, L = cl.shape
        cl = cl.clamp_min(0.0)

        sigma = torch.sqrt(cl[:, :, None] / 2.0)
        real = torch.randn((B, L, L), device=self.device) * sigma
        imag = torch.randn((B, L, L), device=self.device) * sigma
        alm = real + 1j * imag

        alm[:, :, 0] = torch.randn((B, L), device=self.device) * torch.sqrt(cl)
        alm = torch.where(
            self.tri_mask[None, :, :],
            alm,
            torch.zeros((), dtype=alm.dtype, device=self.device),
        )
        return alm.to(torch.complex64)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.dl_bank.shape[0], size=self.batch_size)
        tdl_np = np.array(self.dl_bank[idx], dtype=np.float32, copy=True)
        tdl = torch.as_tensor(tdl_np, device=self.device, dtype=self.dtype)

        tcl = dl_to_cl_torch(tdl)
        alm = self._draw_alms(tcl)
        maps = self.isht(alm).real.to(self.dtype)

        return maps, tdl


class MapSamplerSYNFAST(MapSampler):
    """
    Generates batches of (maps, dl) on CPU from a power spectra bank using healpy.

    Owns the full Dl -> Cl -> maps pipeline via healpy.synfast.

    lmax is the highest ell used (ells 0..lmax inclusive).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ell = np.arange(self.lmax + 1, dtype=np.float64)
        self._dl_to_cl_factor = np.where(
            ell * (ell + 1) > 0,
            2.0 * np.pi / (ell * (ell + 1)),
            1.0,
        )

    def _dl_to_cl(self, dl: np.ndarray) -> np.ndarray:
        return dl * self._dl_to_cl_factor  # (B, lmax+1)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.dl_bank.shape[0], size=self.batch_size)
        tdl_np = np.array(self.dl_bank[idx], dtype=np.float64, copy=True)
        tcl_np = self._dl_to_cl(tdl_np).clip(min=0.0)

        maps_np = np.stack([
            hp.synfast(tcl_np[i], nside=self.nside, lmax=self.lmax, verbose=False)
            for i in range(self.batch_size)
        ])  # (B, npix)

        maps = torch.as_tensor(maps_np, dtype=self.dtype).to(self.device)
        tdl  = torch.as_tensor(tdl_np, dtype=self.dtype).to(self.device)

        return maps, tdl


def make_map_sampler(
    *args,
    sim_engine: str,
    **kwargs,
) -> MapSampler:
    """
    Factory that returns a MapSamplerCUHPX when cuHPX is available and
    prefer_gpu=True, falling back to MapSamplerSYNFAST otherwise.
    """
    sim_engine = sim_engine.lower()
    if sim_engine == 'synfast':
        return MapSamplerSYNFAST(*args, **kwargs)
    elif sim_engine == 'cuhpx':
        if _CUHPX_AVAILABLE:
            logger.info("cuHPX available — using MapSamplerCUHPX.")
            return MapSamplerCUHPX(*args, **kwargs)
        else:
            raise ValueError("Can't find cuHPX")
    else:
        raise NotImplementedError("Only 'synfast' and 'cuhpx' are accepted.")


class TrainingMapSampler:
    """
    Wraps an inner MapSampler to produce (processed_maps, scaled_label) batches
    for training.

    The label depends on spectrum_kind:
      - "theory":      the theory Dl drawn from the bank (inner.sample()'s 2nd
                       return), sliced to label ells.
      - "realization": the realization Dl of the *generated maps*, computed by
                       spectra_estimator, sliced to label ells. The theory Dl
                       from the inner sampler is discarded.

    spectra_estimator is required for "realization" and unused for "theory".
    """
    def __init__(
        self,
        inner: MapSampler,
        map_processor: MapProcessor,
        map_shaper: MapShaper,
        label_scaler: ArrayScaler,
        spectrum_kind: str,
        spectra_estimator=None,
    ):
        self.inner = inner
        self.map_processor = map_processor
        self.map_shaper = map_shaper
        self.label_scaler = label_scaler
        self.spectrum_kind = spectrum_kind
        self.spectra_estimator = spectra_estimator

        if spectrum_kind == "realization" and spectra_estimator is None:
            raise ValueError(
                "spectra_estimator is required when spectrum_kind='realization'."
            )

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        for _ in range(len(self.inner)):
            yield self.sample()

    def sample(self):
        maps, tdl = self.inner.sample()

        if self.spectrum_kind == "theory":
            label = tdl
        else:  # "realization"
            label = self.spectra_estimator(maps)["dl"]  # (B, lmax+1) realization Dl

        maps = self.map_processor.preprocess(maps, channel="cmb", field_str="I")
        maps = self.map_shaper.forward(maps)
        # NOTE: ell-selection site — must match the stats-fit slice and the
        # validation-label ells. Currently [2:] (ells 2..lmax). See TODO on
        # unifying ell selection via self.ells.
        scaled_label = self.label_scaler.transform_tensor(label[:, 2:])
        return maps, scaled_label
