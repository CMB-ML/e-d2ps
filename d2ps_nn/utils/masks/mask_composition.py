from typing import Any
import numpy as np


def fmt_gal_cut(gc) -> str:
    gc = str(gc).strip().upper()
    return gc if gc.startswith("GAL") else f"GAL{int(gc):03d}"


def gal_cut_for_channel(ch, gal_cut_cfg) -> str:
    per_ch = {int(k): v for k, v in (gal_cut_cfg.get("per_channel") or {}).items()}
    return fmt_gal_cut(per_ch.get(int(ch), gal_cut_cfg.default))


def gal_mask_label(ch, gal_source, gal_cut_cfg) -> str:
    """Label identifying the galactic mask used for a channel: 'NILC' or e.g. 'GAL060'."""
    if gal_source == "nilc":
        return "NILC"
    if gal_source == "galcut":
        return gal_cut_for_channel(ch, gal_cut_cfg)
    raise ValueError(f"Unknown gal_source {gal_source!r}; expected 'nilc' or 'galcut'")


def compose_masks(*masks) -> np.ndarray:
    """Logical AND of boolean-like masks of equal length; True = unmasked."""
    if not masks:
        raise ValueError("Need at least one mask")
    n = masks[0].size
    for m in masks[1:]:
        if m.size != n:
            raise ValueError(f"Mask sizes differ: {n} vs {m.size}")
    out = masks[0].astype(bool)
    for m in masks[1:]:
        out &= m.astype(bool)
    return out


_STRUCTURAL_KEYS = {
    "XTENSION", "BITPIX", "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME",
    "PIXTYPE", "ORDERING", "NSIDE", "FIRSTPIX", "LASTPIX", "INDXSCHM",
    "OBJECT", "COORDSYS", "POLCCONV", "COMMENT", "HISTORY", "",
}
_STRUCTURAL_PREFIXES = ("NAXIS", "TTYPE", "TFORM", "TUNIT")


def provenance_cards(cards, prefix=None) -> list[tuple[str, Any, str]]:
    """Keep non-structural header cards, optionally prefixing keywords (e.g., 'PS SM_LMAX')."""
    out = []
    for key, value, comment in cards:
        if key in _STRUCTURAL_KEYS or key.startswith(_STRUCTURAL_PREFIXES):
            continue
        out.append((f"{prefix} {key}" if prefix else key, value, comment))
    return out
