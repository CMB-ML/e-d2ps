import numpy as np
import astropy.units as u


def masked_mean(m, mask):
    """Mean over unmasked pixels (mask True = unmasked); equals the least-squares monopole."""
    mask = np.asarray(mask).astype(bool)
    if m.shape[-1] != mask.size:
        raise ValueError(f"Map and mask sizes differ: {m.shape[-1]} vs {mask.size}")
    return m[..., mask].mean(axis=-1)


def summarize(values):
    """Mean and std across sims, preserving units if present."""
    q = u.Quantity(values)
    return dict(mean=q.mean(), std=q.std(), n_sims=len(values))
