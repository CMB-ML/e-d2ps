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


def adjust_zero_level(m, mask, target):
    """
    Shift a map so its mean over the unmasked region equals `target`.

    Args:
        m: HEALPix map (array or Quantity), RING ordering.
        mask: Boolean-like mask, True = unmasked.
        target: Desired masked mean (same units as m, or convertible).

    Returns:
        (adjusted_map, offset, measured): offset = measured - target, and
        adjusted_map = m - offset.
    """
    measured = masked_mean(m, mask)
    if isinstance(m, u.Quantity):
        target = u.Quantity(target).to(m.unit)
    offset = measured - target
    return m - offset, offset, measured
