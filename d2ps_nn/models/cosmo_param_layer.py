import logging
from typing import Optional, Sequence, Dict, Any
import inspect

import camb
import numpy as np
import torch
import torch.nn as nn
from time import perf_counter


logger = logging.getLogger(__name__)


class BaseSpectrumLayer(nn.Module):
    """
    Abstract interface for deterministic coeffs -> spectrum mappings.

    Expected contract
    -----------------
    Input:
        coeffs   shape (B, n_coeffs)

    Output:
        spectrum shape (B, n_ell)
    """

    n_coefs: int
    n_ell: int

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ParameterSpectrumLayer(BaseSpectrumLayer):
    """
    Convert cosmological parameters to TT spectra. Not for use in training loops.
    """

    def __init__(self, 
                 param_names: list[str], 
                 ells: Sequence[int], 
                 pivot_scalar=None
                 ) -> None:
        super().__init__()

        self.param_names = tuple(param_names)
        self.n_coefs = len(self.param_names)  # compatibility name. TODO: Change from n_coefs

        ells = np.asarray(ells, dtype=int)
        if ells.ndim != 1:
            raise ValueError(f"ells must be 1D, got shape {ells.shape}")
        if len(ells) < 1:
            raise ValueError("ells must contain at least one value")

        self.ells = ells
        self.lmax = int(ells[-1])
        self.n_ell = len(ells)
        self.pivot_scalar = pivot_scalar

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "ParameterSpectrumLayer is not differentiable. "
                "Use return_mode='coeffs' during training."
            )

        if params.ndim != 2:
            raise ValueError(
                f"Expected coeffs with shape (B, {self.n_coefs}), "
                f"got {tuple(params.shape)}"
            )

        if params.shape[1] != self.n_coefs:
            raise ValueError(
                f"Expected coeffs.shape[1] == {self.n_coefs}, "
                f"got {params.shape[1]}"
            )

        rows = params.detach().cpu().numpy()

        spectra = []
        for row in rows:
            cosmo_params = dict(zip(self.param_names, row.tolist()))
            tt = get_tt_spectrum_at_ells(cosmo_params, self.lmax, self.ells, self.pivot_scalar)
            spectra.append(tt)

        return torch.as_tensor(
            np.stack(spectra),
            dtype=params.dtype,
            device=params.device,
        )


def get_tt_spectrum_at_ells(cosmo_params, lmax: int, ells: np.ndarray, pivot_scalar):
    cosmo_params["pivot_scalar"] = pivot_scalar

    pars = camb.CAMBparams()

    set_cosmology_args, init_power_args = _split_cosmo_params_dict(cosmo_params, pars)

    pars.set_cosmology(**set_cosmology_args)
    pars.InitPower.set_params(**init_power_args)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # pars: camb.CAMBparams = setup_camb(cosmo_params, lmax)

    results: camb.CAMBdata = camb.get_results(pars)

    camb_results = results

    powers = camb_results.get_cmb_power_spectra(
        CMB_unit="muK",
        lmax=lmax,
    )

    # CAMB returns columns roughly [TT, EE, BB, TE].
    # Shape is usually (lmax + 1, 4), indexed directly by ell.
    tt_all = powers["total"][:, 0]

    return tt_all[ells]


def _split_cosmo_params_dict(cosmo_params: Dict, camb_pars):
    """
    Turn cosmo_params (a single dictionary) into input suitable for CAMB.

    Args:
        cosmo_params (Dict): Dictionary of cosmological parameters.
        camb_pars (camb.CAMBparams): The CAMB parameters object.

    Returns:
        Tuple[Dict, Dict]: Tuple of dictionaries for set_cosmology and InitPower.set_params
    """
    def get_camb_input_params(method):
        sig = inspect.signature(method)
        return [param.name for param in sig.parameters.values() if param.name != 'self']

    set_cosmology_params = get_camb_input_params(camb_pars.set_cosmology)
    init_power_params = get_camb_input_params(camb_pars.InitPower.set_params)

    # Split the cosmo_params dictionary
    set_cosmo_args = {k: v for k, v in cosmo_params.items() if k in set_cosmology_params}
    init_power_args = {k: v for k, v in cosmo_params.items() if k in init_power_params}

    _ensure_all_params_used(set_cosmo_args, init_power_args, cosmo_params)
    # _log_camb_args(set_cosmo_args, init_power_args)

    return set_cosmo_args, init_power_args


def _ensure_all_params_used(set_cosmo_args, init_power_args, cosmo_params) -> None:
    out_params = list(set_cosmo_args.keys())
    out_params.extend(init_power_args.keys())
    for in_param in cosmo_params.keys():
        if in_param == "chain_idx":
            continue
        try:
            assert in_param in out_params
        except AssertionError:
            logger.warning(f"Parameter {in_param} not found in {out_params}.")
