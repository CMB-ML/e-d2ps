import numpy as np
from d2ps_nn.utils.masks.mask_composition import gal_mask_label, provenance_cards
from cmbml.utils.fits_inspection import get_header_cards


def read_channel_mask(asset, name_tracker, ch, gal_source, gal_cut_cfg):
    """
    Read a composed per-channel mask and its provenance.

    Returns:
        (mask, info): mask as a bool array; info as dict(path, f_sky, header),
        the same structure recorded in monopole and arrange_masks notes.
    """
    gal_mask = gal_mask_label(ch, gal_source, gal_cut_cfg).lower()
    with name_tracker.set_contexts(dict(freq=ch, gal_mask=gal_mask)):
        mask = np.asarray(asset.read()[0]).astype(bool)
        path = str(asset.path)
    header = {k: v for k, v, _ in provenance_cards(get_header_cards(path))}
    return mask, dict(path=path, f_sky=float(mask.mean()), header=header)
