"""Turn pipeline config into the keyword arguments ``figures.py`` wants.

THE PAIR. ``figures.py`` holds no visual defaults: colours, labels, scales, and
canvas size all arrive as arguments. That makes its call sites verbose, so each
caller gets an adapter that absorbs the verbosity.

    publication.py   notebook containers + its own constants -> figures.py
    figure_config.py Hydra config + ModelConfig             -> figures.py

The two are deliberately redundant. They serve different callers and neither
imports the other, so the notebook does not need Hydra and the pipeline does not
need the notebook's constants.

WHAT THIS TAKES. Sub-configs, not the whole ``cfg``. Every function here can be
called with a plain dict, which keeps it testable without composing a config and
keeps it from depending on where in the config tree these blocks happen to live.

WHAT THIS DOES NOT TAKE. ``FigData`` is never imported, only duck-typed via
``.get``; the executor module imports this one, and importing back would be a
cycle.
"""

import logging

from omegaconf import DictConfig, ListConfig, OmegaConf


logger = logging.getLogger(__name__)


def _plain(value):
    """OmegaConf container -> plain dict/list, leaving anything else alone."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


# ---------------------------------------------------------------------------
# Per-model appearance
# ---------------------------------------------------------------------------

def styles_for_models(models, fill_alpha=None):
    """``{model_key: plot kwargs}`` from the per-model configs.

    ``models`` is the executor's ordered ``{key: ModelConfig}``. Colour comes
    from each model's own config, so it cannot depend on plot order or on which
    subset is being drawn.

    ``fill_alpha`` adds a +/-1 sigma band for every model in figures that draw
    one. Omit it for a multi-model comparison, where four overlapping bands
    obscure more than they show.
    """
    styles = {}
    for key, model in models.items():
        st = {"color": model.fig_color}
        if fill_alpha is not None:
            st["fill_alpha"] = fill_alpha
        styles[key] = st
    return styles


def labels_for_models(models):
    """``{model_key: legend label}`` from the per-model configs."""
    return {key: model.fig_model_name for key, model in models.items()}


# ---------------------------------------------------------------------------
# Per-quantity and per-run appearance
# ---------------------------------------------------------------------------

def display_for_metric(cfg_figs, metric):
    """``{xlabel, ylabel, yscale}`` for one metric.

    ``xlabel`` is always the multipole label: every figure here is against ell.
    A metric absent from ``metric_display`` falls back to its own name with a
    linear axis and warns, so adding a metric upstream produces a plottable if
    unpolished figure rather than a crash.
    """
    cfg_figs = _plain(cfg_figs)
    table = cfg_figs.get("metric_display", {})

    entry = table.get(metric)
    if entry is None:
        logger.warning(
            "No metric_display entry for %r; using the bare name with a linear "
            "axis. Add it to the figures config.", metric
        )
        entry = {"ylabel": metric}

    display = {"xlabel": cfg_figs["axis_labels"]["ell"]}
    for field in ("ylabel", "yscale"):
        if entry.get(field) is not None:
            display[field] = entry[field]
    return display


def display_for_dl(cfg_figs):
    """``{xlabel, ylabel}`` for a spectrum panel, whose y-axis is D_ell."""
    cfg_figs = _plain(cfg_figs)
    return {
        "xlabel": cfg_figs["axis_labels"]["ell"],
        "ylabel": cfg_figs["axis_labels"]["dl"],
    }


def label_for_metric(cfg_figs, metric):
    """The display name of a metric, for titles. Falls back to the key."""
    cfg_figs = _plain(cfg_figs)
    entry = cfg_figs.get("metric_display", {}).get(metric) or {}
    return entry.get("label", metric)


def ylim_for_metric(cfg_figs, metric):
    """Fixed y-limits for a metric, or None to autoscale."""
    cfg_figs = _plain(cfg_figs)
    entry = cfg_figs.get("metric_display", {}).get(metric) or {}
    ylim = entry.get("ylim")
    return tuple(ylim) if ylim is not None else None


def figure_kwargs(cfg_figs, canvas="figure"):
    """``{figsize, dpi}`` for the canvas named by ``canvas``.

    ``canvas`` selects among the sizes in config -- "figure" for a single panel,
    "figure_tall" for the two-panel layouts -- so figure size is a run-level
    decision made once rather than a default buried in a plotting function.
    """
    cfg_figs = _plain(cfg_figs)
    try:
        block = cfg_figs[canvas]
    except KeyError:
        raise KeyError(
            f"No {canvas!r} block in the figures config; "
            f"has {sorted(k for k in cfg_figs)}"
        ) from None
    return {"figsize": tuple(block["figsize"]), "dpi": block["dpi"]}


def theory_style(cfg_figs):
    """Appearance of the ground-truth curve and its band."""
    cfg_figs = _plain(cfg_figs)
    return dict(cfg_figs["theory_style"])


# ---------------------------------------------------------------------------
# Data marshalling
# ---------------------------------------------------------------------------

def metrics_for(data, split, model_keys, metric_keys):
    """``{model: {metric: array}}`` for the given models and metrics.

    ``data`` is a ``FigData``; only ``.get`` is used, so this module does not
    import it. A metric that was not loaded raises from ``FigData.get`` with a
    message naming what to add to the stage's ``metrics`` list.
    """
    if isinstance(metric_keys, str):
        metric_keys = [metric_keys]
    return {
        model: {key: data.get(split, model, key) for key in metric_keys}
        for model in model_keys
    }


def preds_for(data, split, model_keys):
    """``{model: (N, n_ell) array}`` of predictions."""
    return {model: data.get(split, model, "pred") for model in model_keys}
