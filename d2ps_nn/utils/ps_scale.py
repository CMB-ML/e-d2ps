import numpy as np


def make_cl_lognorm_scale(lmax):
    """
    Returns the scale factor for converting Cl to Dl.
    The scale factor is defined as 2 * pi / (l * (l + 1)) for l = 0, 1, ..., lmax.

    The monopole and dipole are set to 1 to prevent division by zero.
    Args:
        lmax (int): The maximum multipole moment. lmax=100 will give a size 101 array.
    Returns:
        np.ndarray: The scale factor for each multipole moment.
    """
    ells = np.arange(lmax + 1)
    ells[0] = 1  # Set monopole and dipole to prevent division by zero
    return (2 * np.pi) / (ells * (ells + 1))
