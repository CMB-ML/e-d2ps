from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import legvander


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


class PolynomialSpectrumLayer(BaseSpectrumLayer):
    """
    Fixed, parameter-free spectrum layer based on a precomputed polynomial basis.

    Input
    -----
    coeffs : torch.Tensor
        Shape (B, degree+1)

    Output
    ------
    spectrum : torch.Tensor
        Shape (B, n_ell)

    Notes
    -----
    Uses a precomputed design matrix phi with shape (n_ell, degree+1), so:

        spectrum = coeffs @ phi.T
    """

    def __init__(
        self,
        n_ell: int,
        degree: int,
        basis_kind: str = "legendre",
        out_range: tuple[float, float] = (-1.0, 1.0),
        index_positions: Optional[Sequence[float]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if n_ell < 2:
            raise ValueError(f"n_ell must be >= 2, got {n_ell}")
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")

        x_np = make_scaled_positions(
            n_ell=n_ell,
            out_range=out_range,
            index_positions=index_positions,
        )
        phi_np = build_basis_matrix(
            x=x_np,
            degree=degree,
            basis_kind=basis_kind,
        )

        if index_positions is None:
            pos_np = np.arange(n_ell, dtype=np.float64)
        else:
            pos_np = np.asarray(index_positions, dtype=np.float64)

        self.n_ell = int(n_ell)
        self.degree = int(degree)
        self.n_coefs = self.degree + 1
        self.basis_kind = basis_kind
        self.out_range = out_range

        self.register_buffer(
            "x",
            torch.as_tensor(x_np, dtype=dtype),
            persistent=False  # Do not save this with the model
        )
        self.register_buffer(
            "phi",
            torch.as_tensor(phi_np, dtype=dtype),
            persistent=False  # Do not save this with the model
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 2:
            raise ValueError(
                f"Expected coeffs with shape (B, {self.n_coefs}), "
                f"got {tuple(coeffs.shape)}"
            )

        if coeffs.shape[1] != self.n_coefs:
            raise ValueError(
                f"Expected coeffs.shape[1] == {self.n_coefs}, "
                f"got {coeffs.shape[1]}"
            )

        phi = self.phi
        if coeffs.dtype != phi.dtype:
            phi = phi.to(dtype=coeffs.dtype)

        res = coeffs @ phi.T

        return res

    def extra_repr(self) -> str:
        return (
            f"n_ell={self.n_ell}, "
            f"degree={self.degree}, "
            f"basis_kind={self.basis_kind!r}, "
            f"out_range={self.out_range}"
        )


# class ParameterSpectrumLayer(BaseSpectrumLayer):
#     """
#     Convert cosmological parameters to TT spectra. Not for use in training loops.
#     """

#     def __init__(self, 
#                  param_names: list[str], 
#                  ells: Sequence[int], 
#                  ps_getter) -> None:
#         super().__init__()

#         self.param_names = tuple(param_names)
#         self.n_coefs = len(self.param_names)  # compatibility name. TODO: Change from n_coefs

#         ells = np.asarray(ells, dtype=int)
#         if ells.ndim != 1:
#             raise ValueError(f"ells must be 1D, got shape {ells.shape}")
#         if len(ells) < 1:
#             raise ValueError("ells must contain at least one value")

#         self.ells = ells
#         self.lmax = int(ells[-1])
#         self.n_ell = len(ells)
#         self.ps_getter = ps_getter

#     def forward(self, params: torch.Tensor) -> torch.Tensor:
#         if torch.is_grad_enabled():
#             raise RuntimeError(
#                 "ParameterSpectrumLayer is not differentiable. "
#                 "Use return_mode='coeffs' during training."
#             )

#         if params.ndim != 2:
#             raise ValueError(
#                 f"Expected coeffs with shape (B, {self.n_coefs}), "
#                 f"got {tuple(params.shape)}"
#             )

#         if params.shape[1] != self.n_coefs:
#             raise ValueError(
#                 f"Expected coeffs.shape[1] == {self.n_coefs}, "
#                 f"got {params.shape[1]}"
#             )

#         rows = params.detach().cpu().numpy()

#         spectra = []
#         for row in rows:
#             cosmo_params = dict(zip(self.param_names, row.tolist()))
#             tt = self.ps_getter(cosmo_params, self.lmax, self.ells)
#             spectra.append(tt)

#         return torch.as_tensor(
#             np.stack(spectra),
#             dtype=params.dtype,
#             device=params.device,
#         )


class IdentitySpectrumLayer(BaseSpectrumLayer):
    """
    Pass-through layer for cases where the backbone already predicts the spectrum.
    """

    def __init__(self, n_outputs: int) -> None:
        super().__init__()

        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")

        self.n_coefs = int(n_outputs)
        self.n_ell = int(n_outputs)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 2:
            raise ValueError(
                f"Expected tensor with shape (B, {self.n_coefs}), "
                f"got {tuple(coeffs.shape)}"
            )

        if coeffs.shape[1] != self.n_coefs:
            raise ValueError(
                f"Expected second dimension {self.n_coefs}, got {coeffs.shape[1]}"
            )

        return coeffs

    def extra_repr(self) -> str:
        return f"n_outputs={self.n_coefs}"
        

def make_scaled_positions(
    n_ell: int,
    out_range: tuple[float, float] = (-1.0, 1.0),
    index_positions: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Create 1D positions scaled into [a, b].

    Parameters
    ----------
    n_ell
        Number of output spectrum values.
    out_range
        Target interval for the scaled coordinate.
    index_positions
        Optional explicit 1D positions to scale instead of arange(n_ell).

    Returns
    -------
    x : np.ndarray
        Shape (n_ell,), dtype float64
    """
    a, b = out_range

    if n_ell < 2:
        raise ValueError(f"n_ell must be >= 2, got {n_ell}")

    if index_positions is None:
        pos = np.arange(n_ell, dtype=np.float64)
    else:
        pos = np.asarray(index_positions, dtype=np.float64)
        if pos.ndim != 1:
            raise ValueError(f"index_positions must be 1D, got shape {pos.shape}")
        if len(pos) != n_ell:
            raise ValueError(
                f"len(index_positions)={len(pos)} does not match n_ell={n_ell}"
            )

    pmin = pos.min()
    pmax = pos.max()

    if np.isclose(pmin, pmax):
        raise ValueError("index_positions must span more than one value")

    x01 = (pos - pmin) / (pmax - pmin)
    x = a + (b - a) * x01
    return x


def build_basis_matrix(
    x: np.ndarray,
    degree: int,
    basis_kind: str,
) -> np.ndarray:
    """
    Build a basis/design matrix of shape (n_x, degree+1).

    Supported basis_kind
    --------------------
    - "monomial"
    - "legendre"
    - "chebyshev"
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    basis_kind = basis_kind.lower()

    if basis_kind == "monomial":
        return np.vander(x, N=degree + 1, increasing=True)

    if basis_kind == "legendre":
        return legvander(x, degree)

    if basis_kind == "chebyshev":
        return chebvander(x, degree)

    raise ValueError(
        f"Unknown basis_kind={basis_kind!r}. "
        "Expected 'monomial', 'legendre', or 'chebyshev'."
    )
