"""Metric utilities for the d2ps theory-prediction task.

SCOPE: theory task only. The target is the theory spectrum (ring), the R^2
denominator and amplitude/bias normalizer are sigma_ring = sqrt(S1), and the
score ceiling is sum(rho0^2). Do not point `thry` at a realization array and read
the output as a realization-task score -- A, b, R^2, e_thry, and score_fraction
would all be normalized by the wrong spread.

The realization task is deliberately NOT folded in here. It shares Prior and the
matched-pair metrics (r, tension, MSE, score, locked_count) unchanged, but the
target-spread normalizer becomes sigma_tilde = sqrt(S1+S2) and the score ceiling
becomes n_ell. One denominator choice is unresolved and must be settled before
that path is built: realization R^2 normalized by analytic S1+S2 (bank, full-sky)
vs. the realization target's own empirical variance (honest to mask/noise). The
fork hinges on whether Real_PS is full-sky -- if masked/noisy its variance
exceeds analytic S1+S2 and the empirical denominator is the more correct one,
the opposite of how the theory task resolved it.

This module is the canonical definition of the per-ell metrics. Any caller --
the executor, an interactive session, a report -- imports from here, so the
numbers cannot fork across contexts. Where another file disagrees with this one,
this one is correct.

Two normalizer sources, by design. The rule, stated once and authoritative:

  * Matched-pair quantities (the MSE numerator, the mean offset, the
    correlation, the residual spread) come from the per-split targets `thry`.
    They are only defined relative to the prediction<->target pairing and have
    no bank analogue.

  * Distributional normalizers (S1, S2, sigma_tilde, rho0, and the
    cosmic-variance scale) come from `bank` -- the cosmological prior, estimated
    from the full draw (~100k) rather than the ~200 targets that happen to land
    in a split. The ceiling is a property of the prior, not of the split.

All metrics returned here are dimensionless (ratios), so the only requirement is
that `pred`, `thry`, and `bank` are in consistent units; no unit handling is
done in this module. Population convention throughout (ddof=0): Greek symbols
(sigma_ring, rho0, ...) are population quantities, computed from the bank.
"""

import numpy as np
from functools import cached_property


# --------------------------------------------------------------------------- #
# Prior: population functionals of the cosmological prior, from the theory bank
# --------------------------------------------------------------------------- #

class Prior:
    """Per-ell population functionals, computed once from the theory bank.

    The bank is full-width with column index == ell (column j holds ell j). It
    is cropped to `ells` on construction; everything downstream lives on that
    axis.
    """

    def __init__(self, bank, ells):
        ells = np.asarray(ells)
        bank = np.asarray(bank, dtype=float)
        hi = int(ells.max())
        if bank.shape[1] <= hi:
            raise ValueError(
                f"bank has {bank.shape[1]} columns but ells reach {hi}; "
                "bank must be indexed by ell (column j == ell j)"
            )
        self.ells = ells
        self._bank = bank[:, ells]          # (N_bank, n_ell)

    @property
    def n_ell(self):
        return self.ells.size

    @cached_property
    def nu(self):
        return 2.0 * self.ells + 1.0

    @cached_property
    def mu(self):                            # mu_ring : mean theory spectrum
        return self._bank.mean(0)

    @cached_property
    def S1(self):                            # between-cosmology variance (signal)
        return self._bank.var(0)

    @cached_property
    def S2(self):                            # cosmic-variance scatter = (2/nu) E[D_ring^2]
        return (2.0 / self.nu) * (self._bank ** 2).mean(0)

    @cached_property
    def sigma_ring(self):                    # sigma_ring = sqrt(S1)
        return np.sqrt(self.S1)

    @cached_property
    def sigma_tilde2(self):                  # realization variance S1 + S2
        return self.S1 + self.S2

    @cached_property
    def rho0_sq(self):                       # single-ell ceiling S1 / (S1 + S2)
        return self.S1 / self.sigma_tilde2

    @cached_property
    def rho0(self):
        return np.sqrt(self.rho0_sq)

    @cached_property
    def sigma_cv(self):                      # eq 25 cosmic-variance scale: C_ell sqrt(2/nu)
        return self.mu * np.sqrt(2.0 / self.nu)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _check_axes(pred, thry, prior):
    """Shared: 2-D pred, and pred/thry/prior all on the same ell-axis."""
    if pred.ndim != 2:
        raise ValueError(f"expected pred (N_samples, n_ell), got {pred.shape}")
    if pred.shape[1] != prior.n_ell:
        raise ValueError(
            f"n_ell mismatch: pred has {pred.shape[1]}, prior has {prior.n_ell}. "
            "Crop pred/thry to the prior's ells before scoring."
        )
    if thry.shape[1] != prior.n_ell:
        raise ValueError(
            f"n_ell mismatch: thry has {thry.shape[1]}, prior has {prior.n_ell}."
        )


def _check_matched(pred, thry, prior):
    """Pairwise metrics: target must vary sample-by-sample, matched to pred.
    A single-spectrum (constant) target has no across-sample correlation or
    spread, so A/r/b/R^2 are undefined and this fails loudly rather than
    broadcasting a constant into nonsense."""
    _check_axes(pred, thry, prior)
    if thry.shape != pred.shape:
        raise ValueError(
            f"pairwise metrics need a matched, varying target: pred {pred.shape} "
            f"vs thry {thry.shape}. A single-spectrum target supports tension only."
        )


def _check_broadcast(pred, thry, prior):
    """Tension metrics: target need only broadcast against pred. Either a matched
    (N, n_ell) target or a single (1, n_ell) spectrum is valid; any other N is a
    real mismatch and fails."""
    _check_axes(pred, thry, prior)
    if thry.shape[0] not in (1, pred.shape[0]):
        raise ValueError(
            f"thry N={thry.shape[0]} matches neither 1 (single target) nor "
            f"pred N={pred.shape[0]}."
        )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def pairwise_metrics(pred, thry, prior):
    """Per-ell A, r, b, R^2 for a theory prediction.

    Numerator quantities (means, residuals, correlation) use the matched targets
    `thry`. The normalizing spread sigma_ring = sqrt(S1) and the R^2 denominator
    S1 both come from `prior` (the bank), so R^2 and its rho0^2 ceiling live on
    the same axis. R^2 is taken directly from MSE, not reconstructed from the
    identity (the identity being R^2 = 2rA - A^2 - b^2).

    Note the identity is only *approximate* for the values returned here: r uses
    the local `thry` spread (a correlation must), while A, b, and the R^2
    denominator use the bank sigma_ring. The direct-from-MSE R^2 is exact
    regardless; the identity stays a diagnostic, not the definition.
    """
    pred = np.asarray(pred, dtype=float)
    thry = np.asarray(thry, dtype=float)
    _check_matched(pred, thry, prior)

    mu_p, mu_t = pred.mean(0), thry.mean(0)
    sd_p, sd_t = pred.std(0), thry.std(0)
    sigma = prior.sigma_ring                         # distributional denominator

    A = sd_p / sigma
    b = (mu_p - mu_t) / sigma
    # correlation is pairing-defined: it must use the actual spreads of the two
    # paired variables, so sd_t is local here (a bank std would not be a corr).
    cov = ((pred - mu_p) * (thry - mu_t)).mean(0)
    r = cov / (sd_p * sd_t)
    mse = ((pred - thry) ** 2).mean(0)
    R2 = 1.0 - mse / prior.S1                         # denominator from bank

    return {"ell": prior.ells, "A": A, "r": r, "b": b, "R2": R2, "mse": mse}


def tension_metrics(pred, thry, prior):
    """Per-ell mean-error normalizations.

    e_thry  : <E>/sigma_ring   -- mean error in units of theory spread (signed)
    e_cv    : <E>/sigma_cv     -- mean error in units of cosmic-variance (signed)
    tension : |<E>|/sigma(E)   -- mean error in units of its own across-sample
              spread; ~1 means the offset is the size of the scatter, <<1 buried
              in it, >>1 a real systematic.

    <E> and sigma(E) are matched-pair quantities (from thry); the two
    denominators sigma_ring and sigma_cv are distributional (from the bank).

    Unlike the pairwise metrics, tension never uses the target's across-sample
    spread, so it is well-defined on a single-spectrum (constant) target: <E> is
    the prediction's mean offset from the one true spectrum and sigma(E) is the
    scatter of the predictions about it. A 1-D or (1, n_ell) `thry` broadcasts.
    """
    pred = np.asarray(pred, dtype=float)
    thry = np.atleast_2d(np.asarray(thry, dtype=float))
    _check_broadcast(pred, thry, prior)

    me = pred.mean(0) - thry.mean(0)                  # <E> = <D_hat> - <D_ring>
    sd_err = (pred - thry).std(0)                     # sigma(E)

    return {
        "ell": prior.ells,
        "e_thry": me / prior.sigma_ring,
        "e_cv": me / prior.sigma_cv,
        "tension": np.abs(me) / sd_err,
    }


def compute_metrics(pred, thry, prior):
    """All applicable per-ell metrics plus the prior reference lines, in one dict.

    A matched, varying target (N>=2, same N as pred) yields the full set:
    pairwise (A, r, b, R^2, mse) and tension (e_thry, e_cv, tension). A
    single-spectrum target (N==1, e.g. a fixed-cosmology split) yields tension
    only -- the pairwise metrics are undefined without across-sample target
    variation, so they are omitted, not faked. Callers should branch on the
    presence of keys (e.g. ``"R2" in result``) rather than assume the full set.

    The reference block (rho0, rho0_sq) is prediction-independent and always
    present, so a figure or saved record carries its own ceiling.
    """
    pred = np.asarray(pred, dtype=float)
    thry = np.atleast_2d(np.asarray(thry, dtype=float))

    out = {}
    if thry.shape[0] >= 2 and thry.shape[0] == pred.shape[0]:
        out.update(pairwise_metrics(pred, thry, prior))
    out.update(tension_metrics(pred, thry, prior))
    out["rho0"] = prior.rho0
    out["rho0_sq"] = prior.rho0_sq
    return out


# --------------------------------------------------------------------------- #
# scalar reducers -- each answers one question; the caller composes what it saves
# --------------------------------------------------------------------------- #
#
# These three measure different things and are deliberately not bundled:
#   score / score_fraction : aggregate skill (total variance explained)
#   locked_count           : absolute quality -- how many multipoles clear a bar
#   cleared_count          : pooling -- how many beat their own single-ell ceiling
# All NaN-safe: a multipole with undefined R^2 contributes 0 / does not count.

def score(result):
    """sum_ell [R^2_ell]_+ : total variance explained, positive-clipped so an
    unlearned (negative R^2) multipole contributes 0 instead of dragging the
    total down. Has a theory-task ceiling of sum(rho0^2), not n_ell -- use
    score_fraction for a value comparable across ell-ranges or priors."""
    R2 = np.asarray(result["R2"], dtype=float)
    return float(np.where(np.isfinite(R2), np.clip(R2, 0.0, None), 0.0).sum())


def score_fraction(result):
    """score / sum(rho0^2) : fraction of the achievable (theory-task) ceiling
    captured. >1 is the cross-ell pooling signal (the model, in aggregate, beats
    the single-ell oracle). Requires rho0_sq in `result`."""
    ceiling = float(np.nansum(np.asarray(result["rho0_sq"], dtype=float)))
    return score(result) / ceiling if ceiling > 0 else float("nan")


def locked_count(result, threshold=0.5):
    """#{ell : R^2_ell > threshold} : absolute quality bar, prior-independent and
    comparable across runs. Caller loops for a multi-bar profile, e.g. 0.5 and
    0.9. On the theory task a bar above rho0^2 at some ell is unreachable there
    without pooling, so a high-bar count doubles as a pooling indicator."""
    R2 = np.asarray(result["R2"], dtype=float)
    return int(np.sum(np.isfinite(R2) & (R2 > threshold)))


def cleared_count(result):
    """#{ell : R^2_ell > rho0^2_ell} : multipoles beating their own single-ell
    ceiling -- the discrete pooling count, at 1.0x the ceiling (no arbitrary
    fraction). Compared against R^2, the ceiling is rho0^2, not rho0. Requires
    rho0_sq in `result`."""
    R2 = np.asarray(result["R2"], dtype=float)
    ceil = np.asarray(result["rho0_sq"], dtype=float)
    return int(np.sum(np.isfinite(R2) & (R2 > ceil)))