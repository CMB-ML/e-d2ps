def expand_down_reduction_spec(spec: str, n_levels: int) -> list[str]:
    if not isinstance(spec, str):
        raise TypeError(f"feature_extraction spec must be a string; got {type(spec)}")

    spec = spec.strip().lower()

    # Handle full-word cases
    if spec in {"mean", "avg"}:
        return ["avg"] * n_levels
    if spec == "max":
        return ["max"] * n_levels

    # Compact spec using m/x
    if any(ch not in {"m", "x"} for ch in spec):
        raise ValueError(
            f"Invalid feature_extraction spec {spec!r}; "
            "allowed: 'm', 'x', or full strings 'mean', 'avg', 'max'"
        )

    if len(spec) > n_levels:
        raise ValueError(
            f"feature_extraction spec {spec!r} has length {len(spec)} > n_levels={n_levels}"
        )

    # Pad with mean
    spec = spec + "m" * (n_levels - len(spec))

    mapping = {
        "m": "avg",
        "x": "max",
    }

    return [mapping[ch] for ch in spec]
