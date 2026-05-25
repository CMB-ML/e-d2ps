from d2ps_fcn.scaler import ArrayScaler


VALID_PREP_SPECS = {
    "none": ("none", "global"),
    "minmax_global": ("minmax", "global"),
    "minmax_local": ("minmax", "local"),
    "meanstdv_global": ("meanstdv", "global"),
    "meanstdv_local": ("meanstdv", "local"),
}


def make_scaler_from_prep_spec(spec: str) -> ArrayScaler:
    try:
        kind, scope = VALID_PREP_SPECS[spec]
    except KeyError:
        valid = ", ".join(VALID_PREP_SPECS)
        raise ValueError(
            f"Unsupported preprocessing spec: {spec!r}. "
            f"Supported values are: {valid}"
        )

    return ArrayScaler(kind=kind, scope=scope)
