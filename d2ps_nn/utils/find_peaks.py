import logging
import numpy as np


logger = logging.getLogger(__name__)
STRICT=True


def get_coarse_windows():
    sig_locs = np.array([0, 221, 412, 539, 683, 822, 1024, 1138, 1328], dtype=float)
    bounds = 0.5 * (sig_locs[:-1] + sig_locs[1:])
    return bounds


class PeakFitError(Exception):
    def __init__(self, peak_index, message="Peak fit failed"):
        self.peak_index = peak_index
        super().__init__(f"Peak {peak_index}: {message}")


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


def iterate_peak(ps, ells, ell_lo, ell_hi, window_n, o_width, window_frac, tol, max_iter, strict):
    history = []
    ell_old = 0

    for iter_idx in range(max_iter):
        try:
            ell = quad_vertex_in_window(ps, ells, ell_lo, ell_hi)
        except ValueError as e:
            if not strict:
                logger.warning(str(e))
                return None
            else:
                raise

        history.append(ell)

        if abs(ell_old - ell) < tol:
            return ell

        for i in range(len(history)-1):
            if np.isclose(history[i], history[-1]):
                cycle_vals = history[i:-1]
                return ell if len(cycle_vals) == 0 else np.mean(cycle_vals)

        ell_lo = ell - o_width * window_frac
        ell_hi = ell + o_width * window_frac
        ell_old = ell

    if strict:
        raise PeakFitError(window_n, "Hit max iterations")
    logger.warning(f"Peak {window_n}: Hit max iterations")
    return None


def find_crits(ps, ells, window_frac=0.1, tol=0.05, max_iter=10, strict=STRICT):
    """
    Given Dl ("log-normalized" Cl or bandpower), return locations of
    peaks/troughs. Assumes standard peak locations as starting points.
    """

    crits = []

    bounds = get_coarse_windows()

    for window_n in range(7):
        ell_lo, ell_hi = bounds[window_n:window_n+2]
        o_width = ell_hi - ell_lo

        ell = iterate_peak(ps, ells, ell_lo, ell_hi,
                           window_n, o_width, window_frac, tol,
                           max_iter, strict)

        crits.append(ell)

    return crits
