import numpy as np
from cmbml.utils.patch_healpix import make_pixel_index_lut


def cmbNNCS_nest2lut(nside, nside_patch=None):
    # The handling below, of "regions" limits this to top level pixels only.
    if nside_patch is not None and nside_patch != nside:
        raise NotImplementedError("Nside of each patch must be same as Nside")

    raw_lut = make_pixel_index_lut(nside_obs=nside, nside_patches=nside_patch)

    nside = nside
    regions = [
        [0, 4, 11],
        [1, 5,  8],
        [2, 6,  9],
        [3, 7, 10],
    ]

    lut = np.zeros((4*nside, 3*nside)).astype(np.int64)
    for j, tlps in enumerate(regions):
        for i, tlp in enumerate(tlps):
            r_ids = raw_lut[tlp]
            patch = np.rot90(r_ids, 2)
            lut[j*nside:(j+1)*nside, (i)*nside:(i+1)*nside] = patch

    return lut
