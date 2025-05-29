import numpy as np


def make_cl_lognorm_scale(ells: np.ndarray) -> np.ndarray:
    """
    Returns the scale factor for converting Cl to Dl.
    The scale factor is defined as 2 * pi / (l * (l + 1)) for l = 0, 1, ..., lmax.

    The monopole and dipole are set to 1 to prevent division by zero.
    Args:
        lmax (int): The maximum multipole moment. lmax=100 will give a size 101 array.
    Returns:
        np.ndarray: The scale factor for each multipole moment.
    """
    use_ells = ells.copy()
    use_ells[ells==0] = 1
    return (2 * np.pi) / (ells * (ells + 1))


def get_ps_scale(scale_ps_choice, ells) -> np.ndarray:
    """
    Returns the scale factor for converting Cl to Dl.
    The scale factor is defined as 2 * pi / (l * (l + 1)) for l = 0, 1, ..., lmax.

    The monopole and dipole are set to 1 to prevent division by zero.
    Args:
        lmax (int): The maximum multipole moment. lmax=100 will give a size 101 array.
    Returns:
        np.ndarray: The scale factor for each multipole moment.
    """
    if scale_ps_choice == "Cl":
        scale_ps = make_cl_lognorm_scale(ells)
    elif scale_ps_choice is None:
        scale_ps = 1
    else:
        scale_ps = scale_ps_choice
    return scale_ps

