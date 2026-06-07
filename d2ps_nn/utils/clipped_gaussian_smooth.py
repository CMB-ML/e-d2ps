import numpy as np
from scipy.ndimage import convolve1d


def clipped_gaussian_smooth(
    values: np.ndarray,
    sigma: float,
    width_factor: float = 3.0,
    start_method: str = "margin",
    end_method: str = "margin",
) -> np.ndarray:
    valid = {"interior", "margin"}
    if start_method not in valid or end_method not in valid:
        raise ValueError(f"methods must be one of {valid}")

    n = len(values)
    half_width = int(np.ceil(sigma * width_factor))
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    full_kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    full_kernel /= full_kernel.sum()

    # --- Interior: one vectorized convolution over the whole array ---
    # convolve1d with mode="reflect" or similar is fine for interior samples;
    # margin regions are overwritten afterward.
    out = convolve1d(values.astype(float), full_kernel, mode="nearest")

    # --- Edge regions: indices [0, half_width) and (n-1-half_width, n-1] ---
    margin_indices = list(range(min(half_width, n)))
    end_start = max(0, n - half_width)
    end_indices = list(range(end_start, n))

    def apply_interior_edge(i, dist_start, dist_end):
        lo = max(0, i - half_width)
        hi = min(n - 1, i + half_width)
        k_lo = lo - (i - half_width)
        k_hi = k_lo + (hi - lo)
        kernel = full_kernel[k_lo : k_hi + 1]
        kernel = kernel / kernel.sum()
        return np.dot(kernel, values[lo : hi + 1])

    def apply_margin_edge(i, distance_to_edge):
        local_half = distance_to_edge
        if local_half == 0:
            return values[i]
        compression = local_half / half_width
        edge_offsets = np.linspace(-local_half, local_half, 2 * local_half + 1)
        kernel = np.exp(-0.5 * (edge_offsets / (sigma * compression)) ** 2)
        kernel /= kernel.sum()
        return np.dot(kernel, values[i - local_half : i + local_half + 1])

    for i in margin_indices:
        dist_start, dist_end = i, n - 1 - i
        method = start_method if dist_start <= dist_end else end_method
        if method == "interior":
            out[i] = apply_interior_edge(i, dist_start, dist_end)
        else:
            out[i] = apply_margin_edge(i, dist_start)

    for i in end_indices:
        dist_start, dist_end = i, n - 1 - i
        if dist_start == dist_end:
            continue  # already handled by start loop if n is odd
        method = end_method if dist_end <= dist_start else start_method
        if method == "interior":
            out[i] = apply_interior_edge(i, dist_start, dist_end)
        else:
            out[i] = apply_margin_edge(i, dist_end)

    return out
