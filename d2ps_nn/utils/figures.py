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
from matplotlib.lines import Line2D
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
        raise KeyError(
            f"metrics is missing entries for {missing}; has {sorted(metrics)}. "
            f"metrics should be {{model: {{metric_name: array}}}}, parallel to preds."
        )

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


def plot_single_sim(
    ells, spectra, realization=0, *, styles, labels, display, figure=None,
    theory=None, theory_style=None, annotation=None, annotation_style=None,
    title=None, ax=None, thry_band=None, thry_band_label=None
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
        target = theory[idx]
        ax.plot(ells, target, lw=1, color=ts["color"],
                alpha=ts.get("alpha", 1.0),
                label=ts.get("line_label", "Theory"))
        if thry_band is not None:
            ax.fill_between(
                ells, target - thry_band, target + thry_band,
                color=ts.get("band_color", ts["color"]),
                alpha=ts.get("band_alpha", 0.2),
                linewidth=0, zorder=0,
                label=thry_band_label,
            )

    for key in keys:
        line_style, _ = _split_style(styles[key])
        ax.plot(ells, np.asarray(spectra[key])[realization], lw=1,
                label=labels[key], **line_style)

    _apply_display(ax, display)
    if annotation:
        a = annotation_style or {}
        ax.text(a.get("x", 0.02), a.get("y", 0.05), annotation,
                transform=ax.transAxes, fontsize=a.get("fontsize", 9),
                va="bottom", ha="left")
    ax.legend(loc="best")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_prediction_cloud(
    ells, pred, errors, *, style, label, display, bottom_display,
    cloud_style, figure=None, theory=None, theory_style=None,
    theory_peaks=None, pred_peaks=None, peak_style=None,
    title=None, title_fontsize=12, ax=None,
):
    """Every realization of one model as a faint trace, with an error cloud below.
 
    SINGLE MODEL, deliberately. Unlike the other functions here this takes one
    array rather than a dict keyed by model: N overlaid traces are already dense,
    and two models' clouds on one axis are unreadable. ``style`` and ``label``
    are therefore that one model's, not mappings.
 
    ``pred``    (N, n_ell) predictions, one trace each
    ``errors``  (N, n_ell) ALREADY NORMALIZED errors for the lower panel. The
                caller chooses the denominator -- theory, or its spread across
                realizations -- because that choice is analysis, not drawing,
                and it changes what the panel means. ``bottom_display`` should
                carry the matching label.
    ``theory``  (N, n_ell) or (1, n_ell); its mean is drawn, and its band when
                it varies across realizations
     ``theory_peaks`` and ``pred_peaks`` are each a sequence of peak-position
    sequences, one per realization, marked as short ticks at opposite edges of
    the top panel. Both optional. They arrive as positions rather than being found here:
    peak fitting is analysis, and keeping it out leaves this module free of
    project dependencies.
    """
    ells = np.asarray(ells)
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    errors = np.atleast_2d(np.asarray(errors, dtype=float))
 
    if errors.shape != pred.shape:
        raise ValueError(
            f"errors has shape {errors.shape}, pred has {pred.shape}; "
            f"they must match -- errors is per realization, already normalized"
        )
 
    line_style, fill_alpha = _split_style(style)
    color = line_style.get("color")
 
    if ax is None:
        fig, (ax_top, ax_err) = _new_figure(
            figure, nrows=2, sharex=True,
            gridspec_kw={"height_ratios": cloud_style.get("height_ratios", [1, 2.4])},
            layout="constrained",
        )
    else:
        ax_top, ax_err = ax
        fig = ax_top.figure
 
    # --- bands, behind the traces ---
    if theory is not None:
        ts = _require_theory_style(theory_style)
        theory = np.atleast_2d(np.asarray(theory, dtype=float))
        thry_mean = theory.mean(0)
        if not is_fixed(theory):
            thry_std = theory.std(0)
            ax_top.fill_between(
                ells, thry_mean - thry_std, thry_mean + thry_std,
                color=ts["fill"], alpha=ts["alpha"], linewidth=0,
                label=ts.get("band_label"), zorder=0,
            )
    else:
        thry_mean = None
 
    pred_mean = pred.mean(0)
    if fill_alpha is not None:
        pred_std = pred.std(0)
        ax_top.fill_between(
            ells, pred_mean - pred_std, pred_mean + pred_std,
            color=color, alpha=fill_alpha, linewidth=0,
            label=f"$\\pm\\sigma$ {label}", zorder=1,
        )
 
    # --- the clouds ---
    trace_kwargs = dict(
        color=color,
        alpha=cloud_style["trace_alpha"],
        linewidth=cloud_style["trace_lw"],
    )
    for row in pred:
        ax_top.plot(ells, row, zorder=2, **trace_kwargs)
    for row in errors:
        ax_err.plot(ells, row, **trace_kwargs)
 
    # --- peak ticks ---
    if theory_peaks is not None or pred_peaks is not None:
        ps = peak_style or {}
        for crits in (theory_peaks or []):
            mark_crits_at_axis_edge(
                ax_top, crits, color=ps.get("color", "0.15"),
                where=ps.get("theory_where", "top"),
                tick_frac=ps.get("tick_frac", 0.025),
                linewidth=ps.get("linewidth", 0.45),
                alpha=ps.get("alpha", 0.35),
                ls=ps.get("theory_linestyle", "-"),
            )
        for crits in (pred_peaks or []):
            mark_crits_at_axis_edge(
                ax_top, crits, color=ps.get("color", "0.15"),
                where=ps.get("pred_where", "bottom"),
                tick_frac=ps.get("tick_frac", 0.025),
                linewidth=ps.get("linewidth", 0.45),
                alpha=ps.get("alpha", 0.35),
                ls=ps.get("pred_linestyle", "--"),
            )
 
    # --- mean lines, on top of everything ---
    if thry_mean is not None:
        ax_top.plot(ells, thry_mean, color=theory_style["color"], lw=1.2,
                    label=theory_style.get("line_label", "Theory"), zorder=4)
    ax_top.plot(ells, pred_mean, color=color, lw=1.2, ls="--",
                label=f"Mean {label}", zorder=5)
 
    # --- legend proxies: the traces carry no labels of their own ---
    handles, _ = ax_top.get_legend_handles_labels()
    proxies = [Line2D([0], [0], color=color, lw=1.0, alpha=0.8, label=label)]
    if theory_peaks is not None or pred_peaks is not None:
        ps = peak_style or {}
        pc = ps.get("color", "0.15")
        if theory_peaks is not None:
            proxies.append(Line2D([0], [0], color=pc, lw=1.0,
                                  ls=ps.get("theory_linestyle", "-"),
                                  label=f"Theory peaks ({ps.get('theory_where', 'top')})"))
        if pred_peaks:
            proxies.append(Line2D([0], [0], color=pc, lw=1.0,
                                  ls=ps.get("pred_linestyle", "--"),
                                  label=f"Prediction peaks ({ps.get('pred_where', 'bottom')})"))
    ax_top.legend(handles=[*handles, *proxies],
                  fontsize=cloud_style.get("legend_fontsize", 8),
                  loc="upper right")
 
    _apply_display(ax_top, display)
    ax_top.set_xlim(ells.min(), ells.max())
 
    ax_err.axhline(0, color="0.5", ls=":", lw=1.0)
    _apply_display(ax_err, bottom_display)
    ax_err.set_xlim(ells.min(), ells.max())
 
    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    return fig
 
 
def mark_crits_at_axis_edge(ax, crit_ells, color, where="bottom", tick_frac=0.025,
                            linewidth=1.2, alpha=1.0, ls=None):
    """Mark extrema as short ticks at the top or bottom edge of an axis.
 
    where="bottom" -> ticks rise from the bottom edge.
    where="top"    -> ticks drop from the top edge.
    """
    if where not in {"bottom", "top"}:
        raise ValueError(f"where must be 'bottom' or 'top', got {where!r}")
 
    trans = ax.get_xaxis_transform()        # x in data coords, y in axes fraction
    y_pos = [0.0, tick_frac] if where == "bottom" else [1.0 - tick_frac, 1.0]
 
    for ell in crit_ells:
        if ell is None:
            continue
        ax.plot([ell, ell], y_pos, transform=trans, color=color,
                linewidth=linewidth, alpha=alpha, ls=ls,
                solid_capstyle="butt", clip_on=False)


def plot_r2_decomposition(
    ells, ceiling, losses, r2, *, style, label, display, loss_styles,
    budget_style, figure=None, ceiling_label=None, exemplar=None,
    exemplar_display=None, theory_style=None, identity_rtol=1e-6,
    title=None, title_fontsize=14, ax=None,
):
    """One model's R^2 budget: what it could reach, and what it gives away.

    SINGLE MODEL, like the cloud figure. The terms only mean anything as a
    decomposition of one model's R^2, and four stacked budgets on one axis do
    not read.

    ``ceiling``  (n_ell,) the achievable ceiling -- r^2, what R^2 would be with
                 perfect calibration and no bias
    ``losses``   ordered {name: (n_ell,)} subtracted from the ceiling, stacked
                 in the order given. Insertion order is the stacking order.
    ``r2``       (n_ell,) what was actually achieved

    ``exemplar`` is an optional ``(theory_row, pred_row)`` pair drawn in a panel
    above: one realization, for context. The budget is a per-ell statistic over
    the whole split and says nothing about where in the spectrum a loss sits, so
    the panel is there to let a reader put a dip at ell~1000 against the damping
    tail rather than against nothing.

    The bands are drawn between running partial sums, so the lowest band edge
    lands exactly on ``r2`` when the identity holds. It is checked rather than
    assumed: a mismatch means the terms were computed under different
    conventions -- a ddof change, or a stale artifact -- and the figure would
    otherwise show a gap nobody could explain.
    """
    ells = np.asarray(ells)
    ceiling = np.asarray(ceiling, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    missing = [name for name in losses if name not in loss_styles]
    if missing:
        raise KeyError(
            f"loss_styles is missing entries for {missing}; has {sorted(loss_styles)}"
        )

    residual = ceiling - sum(np.asarray(v, dtype=float) for v in losses.values())
    finite = np.isfinite(residual) & np.isfinite(r2)
    if finite.any() and not np.allclose(residual[finite], r2[finite],
                                        rtol=identity_rtol, atol=identity_rtol):
        worst = np.nanmax(np.abs(residual[finite] - r2[finite]))
        raise ValueError(
            f"ceiling - {' - '.join(losses)} does not equal r2 for {label!r}: "
            f"worst discrepancy {worst:.3g}. The terms were computed under "
            f"different conventions, or one artifact is stale."
        )

    line_style, _ = _split_style(style)
    color = line_style.get("color")

    ax_top = None
    if ax is not None:
        try:
            ax_top, ax = ax
        except TypeError:
            pass
        fig = ax.figure
    elif exemplar is None:
        fig, ax = _new_figure(figure)
    else:
        fig, (ax_top, ax) = _new_figure(
            figure, nrows=2, sharex=True,
            gridspec_kw={"height_ratios": budget_style.get("height_ratios", [1, 2.4])},
            layout="constrained",
        )

    # --- exemplar panel: one realization, for context ---
    if exemplar is not None and ax_top is not None:
        ts = _require_theory_style(theory_style)
        thry_row, pred_row = exemplar
        ax_top.plot(ells, thry_row, color=ts["color"],
                    lw=budget_style.get("exemplar_theory_lw", 1.0),
                    label=ts.get("line_label", "Theory"))
        ax_top.plot(ells, pred_row, color=color,
                    lw=budget_style.get("exemplar_pred_lw", 0.8), label=label)
        _apply_display(ax_top, exemplar_display or {})
        ax_top.legend(fontsize=budget_style.get("legend_fontsize", 7),
                      loc="upper right")

    # --- the budget ---
    upper = ceiling
    for name, loss in losses.items():
        lower = upper - np.asarray(loss, dtype=float)
        ls = loss_styles[name]
        ax.fill_between(
            ells, lower, upper, color=ls["color"], alpha=ls.get("alpha", 0.5),
            linewidth=0, label=ls.get("label", name),
        )
        upper = lower

    ax.plot(ells, ceiling, color=budget_style.get("ceiling_color", "0.3"),
            lw=budget_style.get("ceiling_lw", 1.0),
            label=ceiling_label or "Ceiling")
    ax.plot(ells, r2, lw=budget_style.get("achieved_lw", 1.2), label=label,
            **line_style)

    if budget_style.get("zero_line", True):
        ax.axhline(0, color="0.5", ls=":", lw=1.0)

    _apply_display(ax, display)
    ax.set_xlim(ells.min(), ells.max())
    ax.legend(loc=budget_style.get("legend_loc", "lower right"),
              fontsize=budget_style.get("legend_fontsize", 7),
              ncol=budget_style.get("legend_ncol", 2))

    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    if exemplar is None:
        fig.tight_layout()
    return fig


def plot_paired_realizations(
    ells, theory, pred, *, colors, display, theory_kwargs, pred_kwargs,
    figure=None, theory_peaks=None, pred_peaks=None, peak_style=None,
    legend_labels=("Target", "Prediction"), title=None, ax=None,
):
    """A few realizations, each prediction against its own matched theory.

    POSITIONAL, unlike the keyed functions here. The series are realizations,
    identified by index, so there is no reordering hazard for a keyed contract
    to protect against: ``theory[i]``, ``pred[i]``, and ``colors[i]`` are one
    realization by construction.

    Colour pairs a prediction with its target; ``pred_kwargs["dash_pattern"]``
    is offset per realization so overlapping predictions stay separable where
    they agree.

    Peak ticks go at opposite edges -- theory at the top, prediction at the
    bottom -- so a shifted acoustic peak shows as a horizontal offset between
    two ticks of the same colour. That comparison is the point of the figure,
    which is why the legend names only the two kinds and not each realization.
    """
    ells = np.asarray(ells)
    theory = np.atleast_2d(np.asarray(theory, dtype=float))
    pred = np.atleast_2d(np.asarray(pred, dtype=float))

    if theory.shape != pred.shape:
        raise ValueError(
            f"theory has shape {theory.shape}, pred has {pred.shape}; "
            f"rows are paired by index and must match"
        )
    if len(colors) < len(pred):
        raise ValueError(
            f"{len(colors)} colours for {len(pred)} realizations"
        )

    if ax is None:
        fig, ax = _new_figure(figure)
    else:
        fig = ax.figure

    theory_kwargs = dict(theory_kwargs)
    pred_kwargs = dict(pred_kwargs)
    dash = pred_kwargs.pop("dash_pattern", None)

    ps = peak_style or {}
    mark = theory_peaks is not None or pred_peaks is not None

    for i, (thry_row, pred_row) in enumerate(zip(theory, pred)):
        color = colors[i]
        ls = (i, tuple(dash)) if dash else pred_kwargs.get("linestyle")

        ax.plot(ells, thry_row, color=color, **theory_kwargs)
        ax.plot(ells, pred_row, color=color,
                **{**pred_kwargs, "linestyle": ls})

        if mark:
            if theory_peaks is not None and theory_peaks[i] is not None:
                mark_crits_at_axis_edge(
                    ax, theory_peaks[i], color=color,
                    where=ps.get("theory_where", "top"),
                    tick_frac=ps.get("tick_frac", 0.025),
                    linewidth=ps.get("linewidth", 1.0)*0.5,
                    alpha=ps.get("alpha", 1.0),
                    ls="-",
                )
            if pred_peaks is not None and pred_peaks[i] is not None:
                mark_crits_at_axis_edge(
                    ax, pred_peaks[i], color=color,
                    where=ps.get("pred_where", "bottom"),
                    tick_frac=ps.get("tick_frac", 0.025),
                    linewidth=ps.get("linewidth", 1.0),
                    alpha=ps.get("alpha", 1.0),
                    ls="-",
                )

    # Legend names the two kinds, not the realizations: which sim a trace is
    # does not matter, only that a prediction sits against its own target.
    thry_label, pred_label = legend_labels
    proxies = [
        Line2D([0], [0], color="black", label=thry_label,
               **{k: v for k, v in theory_kwargs.items() if k != "label"}),
        Line2D([0], [0], color="black", label=pred_label,
               **{k: v for k, v in pred_kwargs.items() if k != "label"}),
    ]
    if mark:
        tick_lw = ps.get("linewidth", 1.0)
        if theory_peaks is not None:
            proxies.append(Line2D(
                [0], [0], color="black", lw=tick_lw*0.5, ls="-",
                label=f"{thry_label} peaks ({ps.get('theory_where', 'top')})"))
        if pred_peaks is not None:
            proxies.append(Line2D(
                [0], [0], color="black", lw=tick_lw, ls="-",
                label=f"{pred_label} peaks ({ps.get('pred_where', 'bottom')})"))
    ax.legend(handles=proxies, loc="upper right")

    _apply_display(ax, display)
    ax.set_xlim(ells.min(), ells.max())

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

