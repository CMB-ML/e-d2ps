import numpy as np


def safe_eval_plus(expr):
    """Safely evaluate a simple addition expression like '11+1' or '20'."""
    parts = expr.split('+')
    try:
        return sum(int(p.strip()) for p in parts)
    except ValueError:
        raise ValueError(f"Invalid expression: {expr}")

def parse_slice(slice_str):
    """Parses a slice string like '2:11' or '20:201:10' into a numpy array."""
    parts = [safe_eval_plus(p) for p in slice_str.split(':')]
    ells = np.arange(*parts) if len(parts) > 1 else np.array([parts[0]])
    return ells

def parse_ells(ell_list_str: str, lmax=None):
    """Parses a list of slices from a string and returns a concatenated numpy array."""
    if lmax is not None:
        try:
            ell_list_str = ell_list_str.format(lmax=lmax)
        except:
            print("A!!!!!!!")
            exit()
    segments = ell_list_str.split(',')
    ells = []
    for seg in segments:
        seg = seg.strip('[] ')
        if seg:
            ells.append(parse_slice(seg))
    return np.concatenate(ells)
