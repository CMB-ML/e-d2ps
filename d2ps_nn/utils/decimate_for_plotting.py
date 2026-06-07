import numpy as np


def decimate_for_plot(x, series, target=2000, envelope=False):
    """Downsample x and aligned series for fast plotting of long traces.

    No-op when len(x) <= target — short traces are returned unchanged.

    Parameters
    ----------
    x : 1D array (e.g. epochs), the shared horizontal axis.
    series : dict {key: 1D array} of equal-length traces sharing `x`.
    target : approximate number of points to keep per trace.
    envelope : if False, simple striding (fast, can alias spikes).
               if True, min/max per bin — preserves the visual envelope,
               returns 2x target points and a doubled x.

    Returns
    -------
    x_out : 1D array.
    series_out : dict {key: 1D array} aligned to x_out.
    """
    n = len(x)
    x = np.asarray(x)

    if n <= target:
        return x, {k: np.asarray(v) for k, v in series.items()}

    if not envelope:
        step = n // target
        x_out = x[::step]
        return x_out, {k: np.asarray(v)[::step] for k, v in series.items()}

    # min/max envelope: bin into `target` chunks, keep both extremes per bin
    n_bins = target
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    # representative x per bin (bin center), duplicated for the min/max pair
    x_centers = (x[np.minimum(edges[:-1], n - 1)] + x[np.minimum(edges[1:] - 1, n - 1)]) / 2
    x_out = np.repeat(x_centers, 2)

    series_out = {}
    for k, v in series.items():
        v = np.asarray(v)
        out = np.empty(2 * n_bins, dtype=float)
        for b in range(n_bins):
            seg = v[edges[b]:edges[b + 1]]
            if seg.size == 0:
                out[2 * b] = out[2 * b + 1] = np.nan
            else:
                out[2 * b] = seg.min()
                out[2 * b + 1] = seg.max()
        series_out[k] = out

    return x_out, series_out
