import numpy as np


def get_coarse_windows():
    sig_locs = [0, 221, 412, 539, 683, 822, 1024, 1138, 1328]
    sig_locs = np.array(sig_locs, dtype=float)
    bounds = 0.5 * (sig_locs[:-1] + sig_locs[1:])
    return bounds

class QuadFitError(Exception):
    def __init__(self, message="Quad fit failed"):
        super().__init__(message)
class QuadFitRangeError(Exception):
    def __init__(self, lo, hi, ell, message="Fit out of range"):
        self.lo = lo
        self.hi = hi
        self.ell = ell
        super().__init__(message)

def quad_vertex_in_window(ps, ells, lo, hi, min_pts=7, func=False):
    lo = float(lo); hi = float(hi)
    m = (ells >= lo) & (ells <= hi)
    x = ells[m].astype(float)
    y = ps[m].astype(float)

    if x.size < min_pts:
        raise QuadFitError("x.size < min_pts")

    a, b, c = np.polyfit(x, y, deg=2)

    if not np.isfinite(a) or abs(a) < 1e-20:
        raise QuadFitError("Quadratic term is 0 or infinity")

    ell0 = -b / (2.0 * a)

    # if not (lo <= ell0 <= hi):
    #     raise QuadFitRangeError(lo, hi, ell0, "ell0 not in range between hi and lo")
    # Turns out, this is fine.

    if func:
        return float(ell0), a, b, c
    else:
        return float(ell0)


class PeakFitError(Exception):
    def __init__(self, peak_index, message="Peak fit failed"):
        self.peak_index = peak_index
        super().__init__(message)


def find_crits(dl, ells, window_frac=0.1, tol=0.05):
    """
    Given Dl ("log-normalized" Cl or bandpower), return locations of
    peaks. This assumes usual peak locations as starting point. If 
    using non-standard locations, this may give garbage output.
    """
    crits = []
    for window_n in range(7):  # For each peak or dip
        ell_old = 0
        ell_lo, ell_hi = get_coarse_windows()[window_n:window_n+2]
        o_width = ell_hi-ell_lo

        for iter in range(10):
            ell = quad_vertex_in_window(dl, ells, ell_lo, ell_hi)
            if ell is np.nan:
                raise PeakFitError(window_n, f"Encountered NaN")
            ell_lo = ell - o_width * window_frac
            ell_hi = ell + o_width * window_frac
            if np.abs(ell_old - ell) < tol:
                break
            if iter > 5:
                raise PeakFitError(window_n, f"Hit {iter} iterations")
            # if iter >= 9:
            #     raise ValueError("Too many iterations!")
            ell_old = ell
        crits.append(ell)
    return crits
