import numpy as np
import healpy as hp
import astropy.units as u


def channel_to_field_label(channel) -> str:
    if isinstance(channel, bool):
        raise TypeError("Channel cannot be a bool")
    if isinstance(channel, float):
        if not channel.is_integer():
            raise ValueError(f"Channel must be integer-valued, got {channel}")
        channel = int(channel)
    if isinstance(channel, int):
        return f"F{channel}"
    if isinstance(channel, str):
        channel = channel.strip().upper()
        return channel if channel.startswith("F") else f"F{channel}"
    raise TypeError(f"Unrecognized channel type: {type(channel)}")


def _to_radians(fwhm) -> float:
    if isinstance(fwhm, u.Quantity):
        return fwhm.to(u.rad).value
    return (fwhm * u.arcmin).to(u.rad).value  # bare numbers are arcmin


def process_ps_mask(
    mask,
    fwhm,
    nside_out=512,
    smooth_thresh=0.99,
    dg_thresh=0.999,
    lmax=None,
    n_iter=None,
) -> np.ndarray:
    """
    Smooth, threshold, and downgrade a raw point-source mask.

    Args:
        mask (np.ndarray): Raw HEALPix mask (RING ordering).
        fwhm (float | astropy.units.Quantity): Beam FWHM. Bare numbers are arcmin.
        nside_out (int): Nside of the output mask.
        smooth_thresh (float): Pixels below this value after smoothing are masked.
        dg_thresh (float): Pixels at or below this value after downgrading are masked.
        lmax (int | None): Maximum multipole for smoothing; None = 3 * nside_in - 1.
        n_iter (int): Number of map2alm refinement iterations.

    Returns:
        np.ndarray: Boolean mask at nside_out; True = unmasked.
    """
    nside_in = hp.get_nside(mask)
    max_lmax = 3 * nside_in - 1
    if lmax is not None and not (0 < lmax <= max_lmax):
        raise ValueError(f"lmax must be in (0, {max_lmax}] for nside {nside_in}, got {lmax}")
    if n_iter is None:
        n_iter = 3  # Default of hp.smoothing
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter}")

    mask = hp.smoothing(mask, fwhm=_to_radians(fwhm), lmax=lmax, iter=n_iter)
    mask = (mask >= smooth_thresh).astype(np.float64)
    mask = hp.ud_grade(mask, nside_out)
    return mask > dg_thresh
