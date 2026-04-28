import numpy as np
from cmbml.sims.physics_cmb import make_camb_ps


def get_tt_spectrum_at_ells(cosmo_params, lmax: int, ells: np.ndarray, pivot_scalar=None):
    cosmo_params["pivot_scalar"] = pivot_scalar
    camb_results = make_camb_ps(cosmo_params, lmax=lmax)

    powers = camb_results.get_cmb_power_spectra(
        CMB_unit="muK",
        lmax=lmax,
    )

    # CAMB returns columns roughly [TT, EE, BB, TE].
    # Shape is usually (lmax + 1, 4), indexed directly by ell.
    tt_all = powers["total"][:, 0]

    return tt_all[ells]
