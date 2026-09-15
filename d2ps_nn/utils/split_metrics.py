"""Split-scope per-ell metrics for comparing predicted spectra to theory.

SCOPE. Everything here is normalized within a single data split. There is no
theory bank: the "population" is the N spectra of the split itself. This is the
deliberate simplification relative to ``spectrum_metrics.py``, which normalizes
distributional quantities against a ~100k-draw cosmological prior. That module
remains canonical for the pipeline's theory task; this one exists because the
model-comparison analysis has no bank available and does not need one.

Consequences of dropping the bank, stated once:

  * ``thry_var`` replaces the bank's ``S1`` / ``sigma_ring``. It is the across-
    instance variance of the split's own theory spectra.
  * ``rho0``, ``score``, ``score_fraction``, ``locked_count`` and
    ``cleared_count`` have no meaning here and are absent. They are properties
    of a prior, not of a split.
  * ``R2`` here is the local, identity-closing form (``spectrum_metrics``
    calls it ``R2_local``), NOT the bank-referenced ``1 - mse/S1``.


ORDER OF OPERATIONS. Metrics are computed per instance and aggregated over
instances as the last step. The exceptions -- quantities that are inherently
across-instance and have no per-instance value -- are ``thry_var``,
``decomposition`` (A, r, b, R2) and ``tension``. This ordering matters wherever
the per-instance operation is nonlinear, and the two orderings genuinely differ:
by Jensen's inequality mean(D**2) >= (mean D)**2, so the square-then-mean
``cosmic_var`` below is strictly larger than a mean-then-square version. The
notebook this code replaces used both spellings in different places; square-then-
mean is correct under the stated principle and is what treble's denominator has
always used in practice.

Functions returning shape (N, n_ell) are per-instance and it is the caller's job
to aggregate. Functions returning shape (n_ell,) have already aggregated.


DDOF=0 THROUGHOUT, and this is not a stylistic choice. The R^2 identity

    R^2 = r^2 - (A - r)^2 - b^2

closes only when the variance in the R^2 denominator uses the same divisor as
the MSE numerator, and MSE is a plain sample mean (divisor N). At ddof=1 the
identity picks up an N/(N-1) mismatch and R^2 stops equalling 1 - MSE/var.

The numerical cost of switching from the notebook's ddof=1 is negligible, and
worth knowing when reconciling numbers: A = sd_p/sd_t is ddof-invariant (the
correction cancels in the ratio), r is a correlation and likewise invariant, and
only b = (mu_p - mu_t)/sd_t moves -- by sqrt((N-1)/N), so b^2 shifts by 1/N
(0.5% at N=200).


UNITS. All returned quantities are dimensionless ratios except ``mse`` and
``cosmic_var``, which carry the square of the input units. No unit handling is
done here; ``pred`` and ``thry`` must already agree.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# ddof
# --------------------------------------------------------------------------- #

DDOF = 0
"""Delta degrees of freedom for every variance and standard deviation here.

DEFAULT 0, AND THAT IS NOT A STYLE CHOICE. The R^2 identity

    R^2 = r^2 - (A - r)^2 - b^2

closes only when the variance in the R^2 denominator uses the same divisor as
the MSE numerator, and MSE is a plain sample mean (divisor N). At ddof=1 the
identity picks up an N/(N-1) mismatch and R^2 stops equalling 1 - MSE/var.

EXPOSED AS A MODULE CONSTANT ONLY to reproduce the original analysis, which used
ddof=1 throughout. Set ``split_metrics.DDOF = 1`` once, before computing
metrics, and every function here follows. It is a global compatibility switch,
not a per-call parameter -- mixing conventions within one set of metrics would
break the identity in a way nothing would flag.

What actually moves between the two settings, at N=200:
  * A = sd_p/sd_t and r are ddof-invariant (the correction cancels)
  * b shifts by sqrt((N-1)/N), so b^2 and hence R^2 shift by ~0.5%
  * pred_var, thry_var shift by N/(N-1)  (~0.5%)
  * tension shifts by sqrt((N-1)/N)      (~0.25%)
  * mse, rel_err, cosmic_var are unaffected
"""


FSKY_EXPONENT = 1
"""Power of fsky in the cosmic-variance denominator.

DEFAULT 1, the mode-counting result::

    Var(C_ell) = 2 * C_ell**2 / ((2*ell + 1) * fsky)

EXPOSED AS A MODULE CONSTANT ONLY to reproduce the original analysis, which
applied the correction as (D_ell / fsky)**2 -- i.e. exponent 2, a further factor
of 1/sqrt(fsky) too much. At fsky = 0.7695 that made sigma_cv about 14% too
large. Set ``split_metrics.FSKY_EXPONENT = 2`` once, before computing metrics, to
reproduce it; the default is correct and should stay that way for new results.

Like DDOF, this is a global compatibility switch rather than a per-call
argument: mixing conventions within one set of metrics would be invisible.

Affects only cosmic_var and anything built on it (treble_cosmic_var, e_cv).
mse, R^2 and its decomposition, tension, rel_err, pred_var and thry_var are all
untouched by fsky at any exponent.
"""


# --------------------------------------------------------------------------- #
# shape validation
# --------------------------------------------------------------------------- #

def _as_matched(pred, thry, name="metric"):
    """Coerce to float arrays and require a matched, per-instance target.

    Matched means thry.shape == pred.shape: prediction i is compared against
    *its own* theory spectrum i. Broadcasting an ensemble-mean theory spectrum
    against every prediction is a different quantity -- it folds the theory
    ensemble's own scatter into the numerator -- and is rejected here rather
    than silently allowed, because the notebook this replaces did exactly that
    for smoothed spectra and then plotted the result alongside the matched
    version.
    """
    pred = np.asarray(pred, dtype=float)
    thry = np.asarray(thry, dtype=float)
    if pred.ndim != 2:
        raise ValueError(f"{name}: expected pred (N, n_ell), got {pred.shape}")
    if thry.shape != pred.shape:
        raise ValueError(
            f"{name}: needs matched per-instance targets; got pred {pred.shape} "
            f"vs thry {thry.shape}. Pass the per-realization theory spectra, not "
            f"their ensemble mean."
        )
    return pred, thry


def _as_ells(ells, n_ell, name="metric"):
    ells = np.asarray(ells)
    if ells.size != n_ell:
        raise ValueError(
            f"{name}: ells has {ells.size} entries but spectra have {n_ell}. "
            "Crop the spectra to the ell axis before scoring."
        )
    return ells


# --------------------------------------------------------------------------- #
# distributional quantities (split-scope, inherently across-instance)
# --------------------------------------------------------------------------- #

def thry_var(thry):
    """Across-instance variance of the theory spectra, per ell. Shape (n_ell,).

    The split-scope stand-in for the bank's S1. Named for what it is rather
    than with a Greek symbol, deliberately.

    Zero at any ell where the split has a single fixed cosmology (e.g. TestFP),
    which makes A, b and R^2 undefined there; ``decomposition`` returns NaN in
    that case rather than inf.
    """
    return np.asarray(thry, dtype=float).var(0, ddof=DDOF)


def cosmic_var(thry, ells, fsky=1.0):
    """Per-instance cosmic variance of each theory spectrum. Shape (N, n_ell).

        Var(D_ell^i) = 2 * (D_ell^i)**2 / ((2*ell + 1) * fsky)

    This is the mode-counting result: 1/fsky on the variance, 1/sqrt(fsky) on
    the standard deviation.

    CHANGED FROM THE ORIGINAL ANALYSIS, deliberately. The notebook this replaces
    applied the correction as (D_ell / fsky)**2, i.e. 1/fsky**2 on the variance
    -- a further factor of 1/sqrt(fsky) too much. At fsky = 0.7695 that made
    sigma_cv about 14% too large (1.140 vs 1.000 relative to the correct value),
    and anything normalized by it correspondingly too small.

    TO REPRODUCE THE ORIGINAL NUMBERS, set ``FSKY_EXPONENT = 2`` -- see its
    docstring at the top of the module. Do not patch this function: the constant
    is read at call time and reaches every consumer, including
    ``compute_metrics``.

    PER INSTANCE by design: aggregate with ``.mean(0)`` at the call site, which
    gives the square-then-mean ordering (see module docstring).

    fsky defaults to 1.0 (full sky, no correction). It is a scalar multiplying
    every instance and every ell identically, so it cancels out of any
    comparison at fixed fsky -- including the degree-selection criterion. It
    affects only the cosmic-variance-normalized outputs: ``cosmic_var`` itself,
    ``treble_cosmic_var``, and any e_cv built on them.

    NOTE ON THE fsky VALUE ITSELF, which is a separate question from the
    exponent. The original estimates it as sum(w**2)/n_pix on a Gaussian-
    smoothed mask (see ``data_helper.compute_fsky``), a w2-style estimator.
    Apodization makes the "right" fsky genuinely ambiguous, so that is a
    defensible convention -- but the notebook's own markdown records a lack of
    confidence in it. The exponent above is not a convention question; the value
    passed in still is.
    """
    thry = np.asarray(thry, dtype=float)
    ells = _as_ells(ells, thry.shape[-1], "cosmic_var")
    nu = 2.0 * ells + 1.0
    return 2.0 * thry ** 2 / (nu * fsky ** FSKY_EXPONENT)


# --------------------------------------------------------------------------- #
# per-instance metrics (shape (N, n_ell); caller aggregates)
# --------------------------------------------------------------------------- #

def squared_error(pred, thry):
    """Per-instance squared error. Shape (N, n_ell). ``.mean(0)`` gives MSE."""
    pred, thry = _as_matched(pred, thry, "squared_error")
    return (pred - thry) ** 2


def rel_err(pred, thry):
    """Per-instance signed fractional error (pred - thry) / thry. Shape (N, n_ell).

    Signed and fractional. The notebook plotted mean(|.|) * 100; that is
    ``np.abs(rel_err(...)).mean(0) * 100``, which the caller composes. Keeping
    the sign here means the same primitive serves both, and a signed mean
    distinguishes a biased model from a merely noisy one.

    Entries where thry == 0 are NaN; aggregate with ``np.nanmean``.
    """
    pred, thry = _as_matched(pred, thry, "rel_err")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            pred - thry, thry,
            out=np.full_like(pred, np.nan),
            where=(thry != 0),
        )


def treble(pred, thry, denom):
    """RMS residual in units of a chosen spread, per ell. Shape (n_ell,).

        treble(ell) = sqrt( (1/N) * sum_i (pred_i - thry_i)**2 / denom )

    Matched per-instance targets, per the definition: the superscript i sits on
    both the prediction and the truth. The alternative spelling found in the
    notebook -- broadcasting the ensemble-mean theory spectrum -- measures
    distribution adherence rather than prediction quality and is rejected by
    ``_as_matched``.

    ``denom`` is variance-like and may be per-instance (N, n_ell), e.g. the
    output of ``cosmic_var``, or aggregate (n_ell,), e.g. ``thry_var`` or the
    prediction variance. Both broadcast correctly.

    N is taken from ``pred``. The original took it from a module-level constant,
    which silently gave the wrong normalization on any array that did not happen
    to have exactly that many rows.
    """
    pred, thry = _as_matched(pred, thry, "treble")
    denom = np.asarray(denom, dtype=float)
    n = pred.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.sum((pred - thry) ** 2 / denom, axis=0) / n)


# --------------------------------------------------------------------------- #
# aggregate metrics (shape (n_ell,); no per-instance value exists)
# --------------------------------------------------------------------------- #

def _has_spread(sd_t, mu_t, rtol=1e-10):
    """Whether the target genuinely varies across instances, per ell.

    Compares the spread against the magnitude of the values rather than against
    exactly zero. A fixed-cosmology split has identical rows, but ``std(0)`` of
    identical floats is not exactly 0 -- numpy's variance leaves residue around
    1e-11 on values of order 1e3. A ``sd_t > 0`` test therefore passes, and A and
    b are then divided by that residue, producing R^2 around -1e27 instead of
    NaN.

    The margin is wide in both directions: residue sits ~1e-14 relative, while a
    real cosmological spread is ~1e-2 relative, so rtol=1e-10 separates them by
    eight orders either way.
    """
    return sd_t > rtol * np.abs(mu_t)


def decomposition(pred, thry, ells):
    """The R^2 budget R^2 = r^2 - (A - r)^2 - b^2, as an additive per-ell account.

        ceiling_r2 = r^2        variance a perfect linear rescale could explain
        loss_calib = (A - r)^2  lost to a mis-scaled amplitude, recoverable
        loss_bias  = b^2        lost to bias, recoverable
        R2         = ceiling_r2 - loss_calib - loss_bias

    Every term is normalized by the local across-instance target spread
    ``thry.std(0)``. That is mandatory, not incidental: the identity closes only
    when r, A, b and R^2 share one normalizer, and r -- being a correlation --
    is intrinsically local. This is why the split-scope form is the natural one
    here even setting the missing bank aside.

    BOUNDS. Since loss_calib and loss_bias are both non-negative and r^2 <= 1,
    R2 <= r^2 <= 1 always. An upper clip at 1 can therefore never bind; only a
    lower clip at 0 has any effect. See ``mean_r2`` in degree_selection.

    Takes ``ells`` directly rather than a Prior object: nothing here is
    distributional, so requiring a theory bank to compute it was a false
    dependency.

    Returns NaN at any ell where the target does not vary (thry_var == 0), e.g.
    a fixed-cosmology split.
    """
    pred, thry = _as_matched(pred, thry, "decomposition")
    ells = _as_ells(ells, pred.shape[1], "decomposition")

    mu_p, mu_t = pred.mean(0), thry.mean(0)
    sd_p, sd_t = pred.std(0, ddof=DDOF), thry.std(0, ddof=DDOF)

    with np.errstate(divide="ignore", invalid="ignore"):
        varies = _has_spread(sd_t, mu_t)
        A = np.where(varies, sd_p / sd_t, np.nan)
        b = np.where(varies, (mu_p - mu_t) / sd_t, np.nan)
        cov = ((pred - mu_p) * (thry - mu_t)).sum(0) / (pred.shape[0] - DDOF)
        r = np.where(varies & (sd_p > 0), cov / (sd_p * sd_t), np.nan)

    ceiling_r2 = r ** 2
    loss_calib = (A - r) ** 2
    loss_bias = b ** 2

    return {
        "ell": ells,
        "A": A,
        "r": r,
        "b": b,
        "A_minus_r": A - r,
        "ceiling_r2": ceiling_r2,
        "loss_calib": loss_calib,
        "loss_bias": loss_bias,
        "R2": ceiling_r2 - loss_calib - loss_bias,
    }


def tension(pred, thry):
    """|<E>| / sigma(E) per ell, where E = pred - thry. Shape (n_ell,).

    The mean error in units of its own across-instance spread: ~1 means the
    offset is the size of the scatter, <<1 means it is buried in it, >>1 means a
    real systematic.

    Algebraically identical to the notebook's

        abs_err / sqrt(var_pred + var_thry - 2*cov)

    since sigma(E)**2 expands to exactly that. The notebook carried both
    spellings and reconciled them in a later cell; there is one here.
    """
    pred, thry = _as_matched(pred, thry, "tension")
    me = pred.mean(0) - thry.mean(0)
    sd_err = (pred - thry).std(0, ddof=DDOF)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd_err > 0, np.abs(me) / sd_err, np.nan)


def e_thry(pred, thry):
    """Signed mean error in units of the theory spread. Shape (n_ell,).

    <E> / sqrt(thry_var). Split-scope replacement for the bank-normalized
    e_thry; same quantity, local normalizer.
    """
    pred, thry = _as_matched(pred, thry, "e_thry")
    me = pred.mean(0) - thry.mean(0)
    sd_t = thry.std(0, ddof=DDOF)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(_has_spread(sd_t, thry.mean(0)), me / sd_t, np.nan)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def compute_metrics(pred, thry, ells, fsky=1.0):
    """All per-ell metrics for one (model, split) pair, in one dict.

    Aggregation happens here and only here, explicitly, so the ordering is
    visible at the point it occurs rather than hidden inside each metric.

    All values have shape (n_ell,). ``pred`` and ``thry`` must be matched
    (N, n_ell) arrays on the same ell axis as ``ells``.
    """
    pred, thry = _as_matched(pred, thry, "compute_metrics")
    ells = _as_ells(ells, pred.shape[1], "compute_metrics")

    cv = cosmic_var(thry, ells, fsky=fsky)               # per instance
    tv = thry_var(thry)
    pv = pred.var(0, ddof=DDOF)

    out = dict(decomposition(pred, thry, ells))

    # Note that the following are in addition to the metrics from lines 327-335
    out.update({
        # per-instance metrics, aggregated here
        "mse": squared_error(pred, thry).mean(0),
        "rel_err": np.nanmean(rel_err(pred, thry), axis=0),
        "abs_rel_err_pct": np.nanmean(np.abs(rel_err(pred, thry)), axis=0) * 100.0,
        "cosmic_var": cv.mean(0),

        # aggregate metrics
        "thry_var": tv,
        "pred_var": pv,
        "tension": tension(pred, thry),
        "e_thry": e_thry(pred, thry),

        # treble against each of the three denominators the analysis uses
        "treble_cosmic_var": treble(pred, thry, cv),
        "treble_thry_var": treble(pred, thry, tv),
        "treble_pred_var": treble(pred, thry, pv),
    })
    return out
