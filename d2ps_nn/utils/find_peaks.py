import numpy as np


def get_coarse_windows():
    sig_locs = np.array([0, 221, 412, 539, 683, 822, 1024, 1138, 1328], dtype=float)
    bounds = 0.5 * (sig_locs[:-1] + sig_locs[1:])
    return bounds


class PeakFitError(Exception):
    def __init__(self, peak_index, message="Peak fit failed"):
        self.peak_index = peak_index
        super().__init__(f"Peak {peak_index}: {message}")


def quad_vertex_in_window(ps, ells, lo, hi, min_pts=7, func=False):
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

    if func:
        return float(ell0), a, b, c
    return float(ell0)


def find_crits(dl, ells, window_frac=0.1, tol=0.05, max_iter=6):
    """
    Given Dl ("log-normalized" Cl or bandpower), return locations of
    peaks/troughs. Assumes standard peak locations as starting points.
    """
    crits = []
    bounds = get_coarse_windows()

    for peak_index in range(7):
        ell_old = None
        ell_lo, ell_hi = bounds[peak_index:peak_index + 2]
        o_width = ell_hi - ell_lo

        for iteration in range(max_iter):
            try:
                ell = quad_vertex_in_window(dl, ells, ell_lo, ell_hi)
            except Exception as e:
                raise PeakFitError(
                    peak_index,
                    f"quadratic fit failed in window [{ell_lo:.2f}, {ell_hi:.2f}]"
                ) from e

            if np.isnan(ell):
                raise PeakFitError(peak_index, "encountered NaN vertex")

            if ell_old is not None and abs(ell_old - ell) < tol:
                break

            ell_lo = ell - o_width * window_frac
            ell_hi = ell + o_width * window_frac
            ell_old = ell
        else:
            raise PeakFitError(
                peak_index,
                f"failed to converge after {max_iter} iterations"
            )

        crits.append(ell)

    return crits
