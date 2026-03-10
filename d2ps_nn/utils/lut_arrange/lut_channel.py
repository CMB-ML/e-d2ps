import numpy as np
from cmbml.utils.patch_healpix import make_pixel_index_lut


def channel_nest2lut(nside, nside_patch=None):
    # The handling below, of "regions" limits this to top level pixels only.
    if nside_patch is not None and nside_patch != nside:
        raise NotImplementedError("Nside of each patch must be same as Nside")

    raw_lut = make_pixel_index_lut(nside_obs=nside, nside_patches=nside_patch)
    regions = range(12)

    lut = np.zeros((12, nside, nside)).astype(np.int64)
    for k, tlp in enumerate(regions):
        r_ids = raw_lut[tlp]
        # HEALPix Top Level Pixels
        #   North pole are 0, 1, 2, 3
        #   Equatorial are 4, 5, 6, 7
        #   South pole are 8, 9, 10, 11
        if k > 3:  # Don't rotate North Pole
            patch = r_ids
        else:      # Rotate South Pole and Equatorial (equatorial don't matter)
            patch = np.rot90(r_ids, 2)
        lut[k, :, :] = patch

    return lut
