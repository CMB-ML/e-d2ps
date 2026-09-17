"""Figure functions over plain arrays. The layer the pipeline imports.

NO DECISIONS. Nothing here chooses a colour, a label, a font size, or an axis
scale. Every visual choice arrives as a parameter, because the two callers get
those choices from different places: the pipeline reads them from Hydra config
(via ``figure_config.py``), and the notebook reads them from constants it owns
(``publication.py``). A default living here would be a third source of truth,
silently winning whenever a caller forgot to pass something.

NO PROJECT TYPES either. Everything is numpy arrays, plain dicts, and strings.
No ``ModelSpectra``, no ``DictConfig``, no ``Asset``.

CONVENTIONS. Every function returns a matplotlib ``Figure`` and calls neither
``savefig`` nor ``show``: the caller decides, which is what lets one function
serve a notebook cell and a pipeline executor that handles output itself.

THE DICTS. Series data and per-series appearance are keyed by model name, not
zipped by position, so a reordering in one argument cannot silently mismatch
another. Draw order and legend order follow the insertion order of the series
argument (``metrics`` / ``preds`` / ``spectra``).

    metrics / preds / spectra
        ``{model: array}``, or for metrics ``{model: {metric_name: array}}``.
    styles
        ``{model: kwargs}`` splatted into ``ax.plot``. ``color`` is expected;
        anything else matplotlib accepts is passed through. The non-matplotlib
        key ``fill_alpha`` is popped out and used for that model's +/-1 sigma
        band, where a figure draws one.
    labels
        ``{model: str}`` for the legend.
    display
        One dict for the quantity on the axes: ``xlabel``, ``ylabel``, and
        optionally ``yscale`` and ``ylim``. Per-metric, so it changes with what
        is plotted. Every axis property arrives by this one route, which is what
        lets a pipeline stage override any of them with a single config block.
    figure
        One dict for the canvas: ``figsize``, ``dpi``. Per-run, so it is the
        same for every figure in a paper.
    theory_style
        One dict for the ground-truth curve and band: ``color``, ``fill``,
        ``alpha``, ``line_label``, ``band_label``. Required only when a theory
        argument is actually passed.
"""

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

def _check_series(series, styles, labels):
    """Validate the keyed arguments and return the draw order.

    Returns the model keys in the order given by ``series``. Raises naming the
    offending model rather than reporting a length mismatch, which is the point
    of keying these by name instead of zipping by position.
    """
    if not series:
        raise ValueError("no series to plot: the series dict is empty")

    keys = list(series)
    for what, mapping in (("styles", styles), ("labels", labels)):
        missing = [k for k in keys if k not in mapping]
        if missing:
            raise KeyError(
                f"{what} is missing entries for {missing}; "
                f"has {sorted(mapping)}, needs {keys}"
            )
    return keys


def _split_style(style):
    """Separate the plot kwargs from the band alpha.

    ``fill_alpha`` is carried alongside the line style because it is per-model
    appearance, but matplotlib's ``plot`` does not accept it.
    """
    style = dict(style)
    fill_alpha = style.pop("fill_alpha", None)
    return style, fill_alpha


def _new_figure(figure, nrows=1, **kwargs):
    """Create a figure from the ``figure`` dict, which must supply the canvas."""
    if figure is None:
        raise ValueError(
            "figure is required when ax is not given; "
            "supply {'figsize': ..., 'dpi': ...}"
        )
    return plt.subplots(nrows, 1, figsize=figure["figsize"], dpi=figure["dpi"],
                        **kwargs)


def _require_theory_style(theory_style):
    if theory_style is None:
        raise ValueError(
            "theory_style is required when a theory argument is given; "
            "supply {'color': ..., 'fill': ..., 'alpha': ..., "
            "'line_label': ..., 'band_label': ...}"
        )
    return theory_style


def _apply_display(ax, display):
    """Set the axis labels, scale, and limits from a display dict."""
    if "xlabel" in display:
        ax.set_xlabel(display["xlabel"])
    if "ylabel" in display:
        ax.set_ylabel(display["ylabel"])
    if display.get("yscale"):
        ax.set_yscale(display["yscale"])
    if display.get("ylim") is not None:
        ax.set_ylim(*display["ylim"])


# ---------------------------------------------------------------------------
# Fixed-cosmology detection
# ---------------------------------------------------------------------------

def has_spread(arr, rtol=1e-10):
    """Whether ``arr`` varies across instances, per column. Shape (n_ell,).

    Compares the spread against the magnitude of the values, not against exactly
    zero: ``std(0)`` of identical float rows is not 0 but residue around 1e-11 on
    values of order 1e3, so a ``> 0`` test wrongly reports variation. Mirrors
    ``split_metrics._has_spread``, which is private; if that tolerance ever
    changes it must change in both places.
    """
    arr = np.asarray(arr, dtype=float)
    return arr.std(0) > rtol * np.abs(arr.mean(0))


def is_fixed(arr):
    """True when every row of ``arr`` is the same -- a fixed-cosmology split."""
    return not bool(np.any(has_spread(arr)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_per_ell_metric(
    ells, metrics, metric, *, styles, labels, display, figure=None,
    theory_metric=None, theory_style=None, title=None, ax=None,
):
    """One per-ell metric for every model, on shared axes.

    ``metrics``       ``{model: {metric_name: (n_ell,) array}}``
    ``metric``        which key to read from each model's dict
    ``theory_metric`` optional (n_ell,) reference line, e.g. ``thry_var``
                      alongside ``pred_var``

    Needs no spectra: it reads only the metric arrays.
    """
    ells = np.asarray(ells)
    keys = _check_series(metrics, styles, labels)

    if ax is None:
        fig, ax = _new_figure(figure)
    else:
        fig = ax.figure

    if theory_metric is not None:
        ts = _require_theory_style(theory_style)
        ax.plot(ells, theory_metric, lw=1, color=ts["color"],
                label=ts.get("line_label", "Theory"))

    for key in keys:
        met = metrics[key]
        if metric not in met:
            raise KeyError(
                f"{metric!r} not loaded for {key!r}; available: {sorted(met)}"
            )
        line_style, _ = _split_style(styles[key])
        ax.plot(ells, met[metric], lw=1, label=labels[key], **line_style)

    _apply_display(ax, display)
    ax.legend(loc="best")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_model_vs_theory_with_tension(
    ells, preds, metrics, *, styles, labels, display, bottom_display,
    figure=None, theory=None, theory_style=None, bottom_key="tension",
    title=None, title_fontsize=14, ax=None,
):
    """Prediction mean and 1-sigma against theory, with a second panel below.

    ``preds``           ``{model: (N, n_ell) array}``
    ``metrics``         ``{model: {metric_name: (n_ell,) array}}``; needs
                        ``pred_var`` for the band and ``bottom_key`` for the
                        lower panel
    ``theory``          (N, n_ell) ground truth, or None to omit it
    ``display``         axes for the top panel (D_ell)
    ``bottom_display``  axes for the lower panel (whatever ``bottom_key`` is)

    ``bottom_key`` selects what the lower panel shows -- ``"tension"`` by
    default, but any per-ell key in the metric dicts works, which is how an
    alternative or deliberately-reproduced quantity gets plotted without
    touching this function.

    The theory band is omitted when the theory does not vary across instances,
    detected numerically rather than by testing a split name.
    """
    ells = np.asarray(ells)
    keys = _check_series(preds, styles, labels)

    missing = [k for k in keys if k not in metrics]
    if missing:
        raise KeyError(f"metrics is missing entries for {missing}")

    if ax is None:
        fig, (ax1, ax2) = _new_figure(
            figure, nrows=2, sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    if theory is not None:
        ts = _require_theory_style(theory_style)
        theory = np.asarray(theory, dtype=float)
        thry_mean = theory.mean(0)
        ax1.plot(ells, thry_mean, color=ts["color"], lw=1,
                 label=ts.get("line_label", "Theory"))
        if not is_fixed(theory):
            thry_std = theory.std(0)
            ax1.fill_between(
                ells, thry_mean - thry_std, thry_mean + thry_std,
                color=ts["fill"], edgecolor=ts["color"], alpha=ts["alpha"],
                label=ts.get("band_label"),
            )

    single = len(keys) == 1
    for key in keys:
        met = metrics[key]
        for needed in ("pred_var", bottom_key):
            if needed not in met:
                raise KeyError(
                    f"{needed!r} not loaded for {key!r}; available: {sorted(met)}"
                )

        line_style, fill_alpha = _split_style(styles[key])
        pred = np.asarray(preds[key], dtype=float)
        mean = pred.mean(0)
        std = np.sqrt(met["pred_var"])

        ax1.plot(ells, mean, lw=1, label=labels[key], **line_style)
        if fill_alpha is not None:
            ax1.fill_between(
                ells, mean - std, mean + std,
                color=line_style.get("color"), edgecolor=line_style.get("color"),
                alpha=fill_alpha, zorder=line_style.get("zorder"),
                label=f"$\\pm\\sigma$ {labels[key]}" if not single else None,
            )
        ax2.plot(ells, met[bottom_key], lw=1, label=labels[key], **line_style)

    _apply_display(ax1, display)
    ax1.legend(loc="best")

    _apply_display(ax2, bottom_display)
    if not single:
        ax2.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout()
    return fig


def plot_realization(
    ells, spectra, realization=0, *, styles, labels, display, figure=None,
    theory=None, theory_style=None, title=None, ax=None,
):
    """One realization from each series.

    ``spectra`` ``{model: (N, n_ell) array}``

    Covers smooth-versus-raw (pass both under distinct keys) and
    smooth-versus-theory (pass the smoothed array with ``theory``). Needs no
    metrics.

    For a theory with a single row, or one that does not vary, that row is used
    regardless of ``realization``.
    """
    ells = np.asarray(ells)
    keys = _check_series(spectra, styles, labels)

    for key in keys:
        n = np.asarray(spectra[key]).shape[0]
        if not 0 <= realization < n:
            raise IndexError(
                f"realization {realization} out of range for {key!r} "
                f"with {n} spectra"
            )

    if ax is None:
        fig, ax = _new_figure(figure)
    else:
        fig = ax.figure

    if theory is not None:
        ts = _require_theory_style(theory_style)
        theory = np.asarray(theory, dtype=float)
        idx = 0 if (theory.shape[0] == 1 or is_fixed(theory)) else realization
        ax.plot(ells, theory[idx], lw=1, color=ts["color"],
                alpha=ts.get("alpha", 1.0),
                label=ts.get("line_label", "Theory"))

    for key in keys:
        line_style, _ = _split_style(styles[key])
        ax.plot(ells, np.asarray(spectra[key])[realization], lw=1,
                label=labels[key], **line_style)

    _apply_display(ax, display)
    ax.legend(loc="best")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
