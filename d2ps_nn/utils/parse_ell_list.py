import numpy as np


def parse_slice(slice_str):
    """Parses a slice string like '2:11' or '20:201:10' into a numpy array."""
    parts = [int(p) for p in slice_str.split(':')]
    return np.arange(*parts) if len(parts) > 1 else np.array([parts[0]])


def parse_ells(ell_list_str):
    """Parses a list of slices from a string and returns a concatenated numpy array."""
    segments = ell_list_str.split(',')
    ells = []
    for seg in segments:
        seg = seg.strip('[] ')
        if seg:  # skip empty
            ells.append(parse_slice(seg))
    return np.concatenate(ells)
