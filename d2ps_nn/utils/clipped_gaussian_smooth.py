import numpy as np


def clipped_gaussian_smooth(
    values: np.ndarray,
    sigma: float,
    width_factor: float = 3.0,
    start_method: str = "margin",
    end_method: str = "margin",
) -> np.ndarray:
    """Gaussian smoother with configurable edge handling.

    In the interior, applies a truncated Gaussian with the given sigma,
    spanning ±(sigma * width_factor) samples.

    Near each edge, behavior is chosen by `start_method` / `end_method`:
      - "interior": use the full-width kernel, truncated to available samples
                    and renormalized. Same as the interior calculation, just
                    with fewer samples on one side.
      - "margin": horizontally compress the kernel to fit available samples,
                  preserving Gaussian shape. Collapses to the raw value at i=0.

    Parameters
    ----------
    values : 1D array to smooth.
    sigma : Gaussian standard deviation in samples (interior value).
    width_factor : Half-width of the truncated Gaussian in units of sigma.
    start_method, end_method : "interior" or "margin".
    """
    valid = {"interior", "margin"}
    if start_method not in valid or end_method not in valid:
        raise ValueError(f"methods must be one of {valid}")

    n = len(values)
    half_width = int(np.ceil(sigma * width_factor))
    out = np.empty(n)
    full_offsets = np.arange(-half_width, half_width + 1, dtype=float)
    full_kernel = np.exp(-0.5 * (full_offsets / sigma) ** 2)

    for i in range(n):
        dist_start = i
        dist_end = n - 1 - i
        distance_to_edge = min(dist_start, dist_end)

        if distance_to_edge >= half_width:
            # Interior: full kernel, already normalized below
            kernel = full_kernel / full_kernel.sum()
            out[i] = np.dot(kernel, values[i - half_width : i + half_width + 1])
            continue

        # Near an edge — pick method based on which edge
        method = start_method if dist_start < dist_end else end_method

        if method == "interior":
            # Truncate the full kernel to whatever samples are available
            lo = max(0, i - half_width)
            hi = min(n - 1, i + half_width)
            k_lo = lo - (i - half_width)
            k_hi = k_lo + (hi - lo)
            kernel = full_kernel[k_lo : k_hi + 1]
            kernel = kernel / kernel.sum()
            out[i] = np.dot(kernel, values[lo : hi + 1])

        else:  # "margin"
            local_half = distance_to_edge
            if local_half == 0:
                out[i] = values[i]
                continue
            compression = local_half / half_width
            offsets = np.linspace(-local_half, local_half, 2 * local_half + 1)
            kernel = np.exp(-0.5 * (offsets / (sigma * compression)) ** 2)
            kernel /= kernel.sum()
            out[i] = np.dot(kernel, values[i - local_half : i + local_half + 1])

    return out
