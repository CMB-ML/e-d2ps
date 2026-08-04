import logging
import numpy as np


logger = logging.getLogger(__name__)
STRICT = False


class PeakFitError(Exception):
    def __init__(self, peak_index, message="Peak fit failed"):
        self.peak_index = peak_index
        super().__init__(f"Peak {peak_index}: {message}")


def get_coarse_windows_using_known_peaks(): 
    sig_locs = np.array([0, 221, 412, 539, 683, 822, 1024, 1138, 1328], dtype=float) 
    bounds = 0.5 * (sig_locs[:-1] + sig_locs[1:]) 
    return bounds


def get_coarse_windows(ell_min, ell_max, width=20):
    """
    Build fixed-width windows covering [ell_min, ell_max].

    Returns
    -------
    windows : list[tuple[float, float]]
        Inclusive coarse search windows.
    """
    ell_min = float(ell_min)
    ell_max = float(ell_max)
    width = float(width)

    if ell_max <= ell_min:
        raise ValueError(f"Require ell_max > ell_min, got {ell_min}, {ell_max}")
    if width <= 0:
        raise ValueError(f"Require width > 0, got {width}")

    edges = np.arange(ell_min, ell_max + width, width, dtype=float)
    if edges[-1] < ell_max:
        edges = np.append(edges, ell_max)

    windows = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return windows


def quad_vertex_in_window(ps, ells, lo, hi, min_pts=7, return_coeffs=False):
    lo = float(lo)
    hi = float(hi)

    m = (ells >= lo) & (ells <= hi)
    x = ells[m].astype(float)
    y = ps[m].astype(float)

    if x.size < min_pts:
        raise ValueError(f"Too few points in window [{lo}, {hi}] ({x.size} < {min_pts})")

    a, b, c = np.polyfit(x, y, deg=2)

    if not np.isfinite(a) or abs(a) < 1e-20:
        raise ValueError(f"Degenerate quadratic fit in window [{lo}, {hi}]")

    ell0 = -b / (2.0 * a)

    if not np.isfinite(ell0):
        raise ValueError(f"Non-finite vertex in window [{lo}, {hi}]")

    if return_coeffs:
        return float(ell0), a, b, c
    return float(ell0)


def iterate_crit(
    ps,
    ells,
    ell_lo,
    ell_hi,
    window_n,
    o_width,
    window_frac,
    tol,
    max_iter,
    strict,
    min_pts=7,
):
    history = []
    ell_old = np.inf

    for _ in range(max_iter):
        try:
            ell = quad_vertex_in_window(
                ps, ells, ell_lo, ell_hi, min_pts=min_pts
            )
        except ValueError as e:
            if strict:
                raise PeakFitError(window_n, str(e))
            logger.debug(f"Window {window_n}: {e}")
            return None

        history.append(ell)

        if np.isfinite(ell_old) and abs(ell_old - ell) < tol:
            return float(ell)

        # crude cycle detection
        for i in range(len(history) - 1):
            if np.isclose(history[i], history[-1], atol=tol, rtol=0):
                cycle_vals = history[i:]
                return float(np.mean(cycle_vals))

        half_width = max(o_width * window_frac, 1.0)
        ell_lo = ell - half_width
        ell_hi = ell + half_width
        ell_old = ell

    if strict:
        raise PeakFitError(window_n, "Hit max iterations")

    logger.debug(f"Window {window_n}: Hit max iterations")
    return None


def dedupe_close_crits(crits, min_sep=10):
    """
    Merge nearby critical points. If multiple candidates land within min_sep,
    keep a single representative at their mean location.

    Parameters
    ----------
    crits : sequence of float or None
    min_sep : float

    Returns
    -------
    merged : list[float]
    """
    vals = sorted(float(c) for c in crits if c is not None and np.isfinite(c))
    if not vals:
        return []

    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) < min_sep:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [float(np.mean(g)) for g in groups]


def find_crits(
    ps,
    ells,
    coarse_width=20,
    dedupe_sep=10,
    window_frac=0.5,
    tol=0.05,
    max_iter=10,
    min_pts=7,
    strict=STRICT,
):
    """
    Find approximate critical-point locations for plotting.

    This version does not assume known spectral peak locations.
    It sweeps fixed-width windows across the ell range, refines one
    quadratic vertex per window, then merges near-duplicate results.

    Parameters
    ----------
    ps : array-like
        1D values over ell.
    ells : array-like
        1D ell coordinates.
    coarse_width : float
        Width of coarse sweep windows.
    dedupe_sep : float
        Merge critical-point candidates closer than this distance.
    window_frac : float
        Fraction of the original coarse width used for iterative refinement.
    tol : float
        Iteration convergence tolerance in ell.
    max_iter : int
        Maximum refinement iterations per coarse window.
    min_pts : int
        Minimum number of points required for quadratic fit.
    strict : bool
        Raise on fitting failures if True; otherwise warn and continue.

    Returns
    -------
    crits : list[float]
        Approximate critical-point locations.
    """
    ells = np.asarray(ells, dtype=float)
    ps = np.asarray(ps, dtype=float)

    if ells.ndim != 1 or ps.ndim != 1:
        raise ValueError("ells and ps must be 1D")
    if ells.size != ps.size:
        raise ValueError(f"ells and ps must have same size, got {ells.size}, {ps.size}")
    if ells.size < min_pts:
        raise ValueError(f"Need at least {min_pts} points, got {ells.size}")

    order = np.argsort(ells)
    ells = ells[order]
    ps = ps[order]

    windows = get_coarse_windows(ells.min(), ells.max(), width=coarse_width)

    crits = []
    for window_n, (ell_lo, ell_hi) in enumerate(windows):
        o_width = ell_hi - ell_lo

        ell = iterate_crit(
            ps=ps,
            ells=ells,
            ell_lo=ell_lo,
            ell_hi=ell_hi,
            window_n=window_n,
            o_width=o_width,
            window_frac=window_frac,
            tol=tol,
            max_iter=max_iter,
            strict=strict,
            min_pts=min_pts,
        )
        crits.append(ell)

    return dedupe_close_crits(crits, min_sep=dedupe_sep)
