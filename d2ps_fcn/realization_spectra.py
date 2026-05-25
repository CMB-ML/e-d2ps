"""
realization_spectra.py
--------------------
Estimates realisation power spectra (Cl and/or Dl) from HEALPix maps.

Two engines are provided:
  - RealSpectraMakerCUHPX  : GPU, uses cuHPX forward SHT (map -> alm -> Cl)
  - RealSpectraMakerANAFAST: CPU, uses healpy.anafast

Use ``make_spectra_maker(engine='cuhpx'|'anafast', ...)`` as the entry point.

Both engines share the same interface:
  estimator(maps) -> dict with keys 'cl' and/or 'dl'
    maps : (npix,) or (B, npix) torch.Tensor or np.ndarray
    returns : dict of torch.Tensor with shape (lmax+1,) or (B, lmax+1)
              matching the input batch dimension.

The output key set is controlled by ``return_cl`` / ``return_dl`` flags set
at construction time. At least one must be True.
"""

from __future__ import annotations

from typing import Union
import logging

import healpy as hp
import numpy as np
import torch

try:
    from cuhpx import SHTCUDA            # forward SHT: map -> alm
    _CUHPX_AVAILABLE = True
except ImportError:
    _CUHPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ArrayLike = Union[torch.Tensor, np.ndarray]


def cl_to_dl(cl: torch.Tensor) -> torch.Tensor:
    """Cl -> Dl = ell*(ell+1)/(2*pi) * Cl.  Ell=0,1 left as zero."""
    ell = torch.arange(cl.shape[-1], device=cl.device, dtype=cl.dtype)
    factor = ell * (ell + 1.0) / (2.0 * torch.pi)
    return cl * factor


def dl_to_cl(dl: torch.Tensor) -> torch.Tensor:
    """Dl -> Cl.  Ell=0,1 left as zero."""
    ell = torch.arange(dl.shape[-1], device=dl.device, dtype=dl.dtype)
    factor = ell * (ell + 1.0) / (2.0 * torch.pi)
    cl = torch.zeros_like(dl)
    good = factor > 0
    cl[..., good] = dl[..., good] / factor[good]
    return cl


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RealizationSpectraMaker:
    """
    Base class.  Subclasses implement _estimate_cl_tensor(maps) -> Tensor
    where maps is (B, npix) float32 on compute_device and the return is (B, lmax+1).
    """

    def __init__(
        self,
        nside: int,
        lmax: int,
        return_cl: bool = True,
        return_dl: bool = False,
        compute_device: str = "cpu",
        output_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if not (return_cl or return_dl):
            raise ValueError("At least one of return_cl, return_dl must be True.")
        self.nside = nside
        self.lmax = lmax
        self.return_cl = return_cl
        self.return_dl = return_dl
        self.compute_device = compute_device
        self.output_device = output_device
        self.dtype = dtype

    def __call__(self, maps: ArrayLike) -> dict[str, torch.Tensor]:
        # Normalise input to a torch tensor on compute_device.
        # Subclasses receive it via _estimate_cl_tensor and work entirely
        # in torch, so no numpy round-trip is needed on the GPU path.
        if isinstance(maps, np.ndarray):
            maps_t = torch.as_tensor(maps, dtype=torch.float32, device=self.compute_device)
        else:
            maps_t = maps.to(device=self.compute_device, dtype=torch.float32)

        batched = maps_t.ndim == 2          # True -> (B, npix), False -> (npix,)
        if not batched:
            maps_t = maps_t.unsqueeze(0)    # (1, npix)

        cl = self._estimate_cl_tensor(maps_t)   # (B, lmax+1), on compute_device

        # Move to output_device only if needed
        if self.compute_device != self.output_device:
            cl = cl.to(device=self.output_device)
        cl = cl.to(dtype=self.dtype)

        if not batched:
            cl = cl.squeeze(0)              # (lmax+1,)

        out: dict[str, torch.Tensor] = {}
        if self.return_cl:
            out["cl"] = cl
        if self.return_dl:
            out["dl"] = cl_to_dl(cl)
        return out

    def _estimate_cl_tensor(self, maps: torch.Tensor) -> torch.Tensor:
        """
        maps : (B, npix) float32, on self.compute_device
        returns : (B, lmax+1) float32, on self.compute_device
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# cuHPX engine
# ---------------------------------------------------------------------------

class RealSpectraMakerCUHPX(RealizationSpectraMaker):
    """
    GPU-accelerated spectra estimator using cuHPX's forward SHT.

    map -> alm via SHTCUDA -> Cl = mean_{m} |alm|^2 with correct m=0 weight.

    Raises ImportError at construction time if cuHPX is not installed.
    """

    def __init__(self, *args, **kwargs):
        if not _CUHPX_AVAILABLE:
            raise ImportError(
                "cuHPX is not installed. Use SpectraEstimatorANAFAST instead."
            )
        super().__init__(*args, **kwargs)

        alm_size = self.lmax + 1
        # SHTCUDA: map (B, npix) -> alm (B, lmax+1, lmax+1) triangular
        self.fsht = SHTCUDA(self.nside, lmax=alm_size, mmax=alm_size)

        # Precompute masks and weights on compute_device
        L = alm_size
        dev = self.compute_device
        ell = torch.arange(L, device=dev)
        m   = torch.arange(L, device=dev)
        self._tri_mask     = m[None, :] <= ell[:, None]        # (L, L)
        self._tri_mask_pos = self._tri_mask[:, 1:]             # (L, L-1)
        self._two_ell_p1   = (2 * ell + 1).float()            # (L,)

    def _estimate_cl_tensor(self, maps: torch.Tensor) -> torch.Tensor:
        # alm: (B, lmax+1, lmax+1), complex64, triangular layout
        alm = self.fsht(maps)               # upper triangle is zero / unused

        power = alm.abs().pow(2)            # (B, L, L)

        # Cl = (1/(2*ell+1)) * [ |a_{l0}|^2  +  2 * sum_{m>0} |a_{lm}|^2 ]
        # m=0 contributes once; m>0 contributes twice (accounts for -m modes)
        m0   = power[:, :, 0]              # (B, L)
        mpos = power[:, :, 1:]             # (B, L, L-1)
        mpos = mpos * self._tri_mask_pos[None]

        cl = (m0 + 2.0 * mpos.sum(dim=-1)) / self._two_ell_p1  # (B, L)
        return cl


# ---------------------------------------------------------------------------
# healpy ANAFAST engine
# ---------------------------------------------------------------------------

class RealSpectraMakerANAFAST(RealizationSpectraMaker):
    """
    CPU spectra estimator using healpy.anafast.

    Straightforward drop-in when cuHPX is unavailable.
    """

    def _estimate_cl_tensor(self, maps: torch.Tensor) -> torch.Tensor:
        # anafast lives on CPU/numpy; convert, compute, return tensor
        maps_np = maps.cpu().numpy().astype(np.float64)
        cl_np = np.stack([
            hp.anafast(maps_np[i], lmax=self.lmax)
            for i in range(maps_np.shape[0])
        ])  # (B, lmax+1)
        return torch.as_tensor(cl_np, dtype=torch.float32, device=self.compute_device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_real_spectra_maker(
    engine: str,
    nside: int,
    lmax: int,
    return_cl: bool = True,
    return_dl: bool = False,
    compute_device: str = "cpu",
    output_device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> RealizationSpectraMaker:
    """
    Factory mirroring ``make_map_sampler``.

    Parameters
    ----------
    engine : 'cuhpx' or 'anafast'
    nside, lmax : HEALPix parameters
    return_cl : include 'cl' in the output dict
    return_dl : include 'dl' in the output dict
    compute_device : torch device for internal computation
        For 'cuhpx' this should be a CUDA device (e.g. 'cuda' or 'cuda:0').
        For 'anafast' this is ignored (anafast always runs on CPU).
    output_device : torch device for the returned tensors.
        If equal to compute_device, no transfer occurs.
        Set to 'cuda' when feeding directly into a training loop on GPU.
    dtype : torch dtype for output tensors
    **kwargs : passed through to the engine constructor
    """
    engine = engine.lower()
    common = dict(
        nside=nside, lmax=lmax,
        return_cl=return_cl, return_dl=return_dl,
        compute_device=compute_device, output_device=output_device,
        dtype=dtype,
    )

    if engine == "cuhpx":
        if not _CUHPX_AVAILABLE:
            raise ValueError(
                "engine='cuhpx' requested but cuHPX is not installed. "
                "Use engine='anafast' instead."
            )
        logger.info("cuHPX available — using SpectraEstimatorCUHPX.")
        return RealSpectraMakerCUHPX(**common, **kwargs)

    elif engine == "anafast":
        return RealSpectraMakerANAFAST(**common, **kwargs)

    else:
        raise NotImplementedError(f"Unknown engine '{engine}'. Choose 'cuhpx' or 'anafast'.")