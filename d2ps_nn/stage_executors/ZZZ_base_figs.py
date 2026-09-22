from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import NpyHandler, MPLFigure
from d2ps_nn.utils.parse_ell_list import parse_ells
import d2ps_nn.utils.figures as figs
import d2ps_nn.utils.figure_config as fcfg
from d2ps_nn.utils.find_peaks import crits_for_rows


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    working_dir: str
    cleaning_model: str
    fig_model_name: str
    use_epoch: str
    fig_color: str


@dataclass
class FigData:
    ells: np.ndarray
    thry: dict
    arrays: dict
    requested: dict = None      # {"pred": bool, "thry": bool, "metrics": list | None}

    def select(self, split=None, model=None, quantity=None):
        """Every array matching the given constraints. None means 'any'."""
        return [
            v for (s, m, q), v in self.arrays.items()
            if (split is None or s == split)
            and (model is None or m == model)
            and (quantity is None or q == quantity)
        ]

    def get(self, split, model, quantity):
        """One array. Raises with a useful message if it wasn't loaded."""
        try:
            return self.arrays[(split, model, quantity)]
        except KeyError:
            raise KeyError(
                f"{quantity!r} not loaded for model {model!r}, split {split!r}. "
                f"Add it to the stage's 'metrics' list in the pipeline yaml."
            ) from None

    def range(self, split=None, model=None, quantity=None, pad=0.0):
        vals = self.select(split, model, quantity)
        if not vals:
            return None
        stacked = np.concatenate([np.asarray(v).ravel() for v in vals])
        lo, hi = np.nanmin(stacked), np.nanmax(stacked)
        if pad:
            margin = (hi - lo) * pad
            lo, hi = lo - margin, hi + margin
        return float(lo), float(hi)

    @property
    def contents(self) -> str:
        splits = sorted({s for s, _, _ in self.arrays})
        models = sorted({m for _, m, _ in self.arrays})
        quantities = sorted({q for _, _, q in self.arrays})
        expected = len(splits) * len(models) * len(quantities)
        req = self.requested or {}
        lines = [
            f"splits: {splits}",
            f"models: {models}",
            f"quantities: {quantities}",
            f"arrays: {len(self.arrays)} of {expected} possible",
            f"theory: {sorted(self.thry)}",
            f"requested: {req}",
        ]
        if len(self.arrays) != expected:
            missing = [
                (s, m, q)
                for s in splits for m in models for q in quantities
                if (s, m, q) not in self.arrays
            ]
            lines.append(f"missing: {missing}")
        return "\n  ".join(lines)


class BaseFigsExecutor(BaseStageExecutor):
    """Shared machinery for the cross-model analysis figures.

    Loads every split, model, and metric the stage declares BEFORE any figure is
    drawn, so a missing file fails the run outright rather than after three
    figures have already been written.

    Subclasses implement make_figures(data). The two below fix where output goes
    -- per model or to the comparison directory -- and concrete figure classes
    inherit from those.
    """
    def __init__(self, cfg: DictConfig, stage_str="plot_base") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_fig: Asset = self.assets_out["fig"]
        out_fig_handler: MPLFigure
        self.in_pred: Asset = self.assets_in.get("ps_pred", None)
        self.in_thry: Asset = self.assets_in.get("ps_thry", None)
        self.in_per_ell_metric: Asset = self.assets_in.get("per_ell_metric", None)
        in_pred_handler: NpyHandler
        in_thry_handler: NpyHandler
        in_metric_handler: NpyHandler

        self.models: dict[str, ModelConfig] = \
            self.set_up_models(cfg.models,
                               model_order=cfg.get("model_order", None))

        # Metrics are not necessarily needed
        self.metrics_to_load = self.get_stage_element_silent("metrics")

        # Appearance: everything figures.py needs and does not decide for itself.
        # cfg_figs holds what a quantity IS; the two overrides hold what this
        # particular figure needs, and win over it.
        self.cfg_figs = cfg.model.analysis.figures
        self.display_override = self.get_stage_element_silent("display")
        self.figure_override = self.get_stage_element_silent("figure")
        # Two-panel figures need their own block for the lower axes; one
        # display block cannot serve two panels with different quantities.
        self.bottom_display_override = self.get_stage_element_silent("bottom_display")

        self.ells = parse_ells(cfg.model.use_ells)
        self.fig_type = cfg.fig_type

    @property
    def metrics_to_load(self):
        return self._metrics_to_load

    @metrics_to_load.setter
    def metrics_to_load(self, value):
        if value is None:
            self._metrics_to_load = None
        elif isinstance(value, str):
            self._metrics_to_load = [value]
        else:
            self._metrics_to_load = list(value)

    def set_up_models(self, cfg_models: DictConfig, model_order=None) -> dict[str, ModelConfig]:
        """Build the per-model configs, in the order figures will use them.

        model_order sets both order and membership: models omitted from it are
        not loaded and do not appear in any figure. A short list is therefore a
        toggle as well as a reordering — useful for rerunning a subset, but note
        that axis limits computed from the loaded data will then reflect only
        the models present.
        """
        keys = list(model_order) if model_order is not None else list(cfg_models.keys())

        unknown = [k for k in keys if k not in cfg_models]
        if unknown:
            raise ValueError(
                f"model_order names models not in the defaults list: {unknown}. "
                f"Available: {sorted(cfg_models.keys())}"
            )

        models = {}
        for model in keys:
            model_cfg = cfg_models[model]
            models[model] = ModelConfig(
                working_dir=model_cfg["working_dir"],
                cleaning_model=model_cfg["cleaning_model"],
                fig_model_name=model_cfg["fig_model_name"],
                use_epoch=model_cfg["use_epoch"],
                fig_color=model_cfg["fig_color"],
            )
        return models

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.out_fig.handler.set_fig_types(self.fig_type)
        data = self.load_all_data()
        logger.info(f"Data contains: {data.contents}")
        self.make_figures(data)

    def make_figures(self, data: FigData) -> None:
        raise NotImplementedError

    def load_all_data(self) -> FigData:
        data = FigData(
            ells=self.ells, thry={}, arrays={},
            requested={
                "pred": self.in_pred is not None,
                "thry": self.in_thry is not None,
                "metrics": self.metrics_to_load,
            },
        )

        for split in self.splits:
            if self.in_thry is not None:
                data.thry[split.name] = self.get_thry(split)

            for model_key, model_cfg in self.models.items():
                with self.name_tracker.set_context("epoch", model_cfg.use_epoch):
                    if self.in_pred is not None:
                        data.arrays[(split.name, model_key, "pred")] = \
                            self.get_pred(split, model_cfg)
                    if self.metrics_to_load is not None:
                        for metric_lbl in self.metrics_to_load:
                            data.arrays[(split.name, model_key, metric_lbl)] = \
                                self.get_metric(split, model_cfg, metric_lbl=metric_lbl)

        return data

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_thry(self, split: Split) -> np.ndarray:
        with self.name_tracker.set_context("split", split.name):
            thry = self.in_thry.read()
        return thry

    def get_pred(self, split: Split, model: ModelConfig) -> np.ndarray:
        context = dict(
            split=split.name,
            working=model.working_dir,
        )
        with self.name_tracker.set_contexts(context):
            pred = self.in_pred.read()
        return pred

    def get_metric(self, split: Split, model: ModelConfig,
                   metric_lbl: str) -> np.ndarray:
        context = dict(
            split=split.name,
            working=model.working_dir,
            metric=metric_lbl,
        )
        with self.name_tracker.set_contexts(context):
            metric = self.in_per_ell_metric.read()
        return metric

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_fig(self, fig, split: Split, model: ModelConfig = None, **extra) -> None:
        """Write one figure, letting the handler handle fig_type iteration.

        Extra keyword arguments become name-tracker context, for path templates
        carrying tokens beyond split and model -- {metric}, for instance.

        With model=None the output goes to the comparison directory: {working}
        keeps whatever the Namer was constructed with, which is this pipeline's
        own working_dir. That fallback is deliberate here and nowhere else.
        """
        context = dict(split=split.name, **extra)
        if model is not None:
            context["working"] = model.working_dir
            context["epoch"] = model.use_epoch
        with self.name_tracker.set_contexts(context):
            self.out_fig.write(fig=fig, verbose=True)

    # ------------------------------------------------------------------
    # Appearance, from config
    # ------------------------------------------------------------------

    def styles(self, fill_alpha=None, only=None):
        """Plot kwargs per model. ``only`` narrows to a subset, for the
        single-model figures -- the figure functions reject a styles dict
        carrying models that are not being drawn.
        """
        return fcfg.styles_for_models(self._subset(only), fill_alpha=fill_alpha)

    def labels(self, only=None):
        return fcfg.labels_for_models(self._subset(only))

    def _subset(self, only=None):
        if only is None:
            return self.models
        if isinstance(only, str):
            only = [only]
        return {key: self.models[key] for key in only}


class PerModelFigsExecutor(BaseFigsExecutor):
    """One figure per model per split, written to that model's working dir."""

    def make_figures(self, data: FigData) -> None:
        for split in self.splits:
            for model_key, model_cfg in self.models.items():
                fig = self.make_figure(data, split.name, model_key)
                self.save_fig(fig, split, model=model_cfg)

    def make_figure(self, data: FigData, split_name: str, model_key: str):
        raise NotImplementedError


class AllModelFigsExecutor(BaseFigsExecutor):
    """One figure per split covering every model, written to the comparison dir."""

    def make_figures(self, data: FigData) -> None:
        for split in self.splits:
            fig = self.make_figure(data, split.name)
            self.save_fig(fig, split, model=None)

    def make_figure(self, data: FigData, split_name: str):
        raise NotImplementedError


class ComparePerEllMetricExecutor(AllModelFigsExecutor):
    """One per-ell metric, every model on shared axes. One figure per split.

    One metric per stage rather than a loop over several. That costs a stage
    block per figure, and buys each figure its own limits, scale, canvas, and
    output path -- which is what paper figures end up needing. It also keeps
    {metric} out of the path template, since {stage} already differs.

    The metric comes from the stage's 'metrics' list, which must name exactly
    one. Per-figure appearance comes from the stage's optional 'display' and
    'figure' blocks, each merging over the defaults in model.analysis.figures.
    """
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        if stage_str is None:
            raise NotImplementedError("This is a base class")
        super().__init__(cfg, stage_str=stage_str)

        # 'metrics' is the load list; 'metric' is which one to plot. They differ
        # only when the stage loads something extra -- the variance figure loads
        # thry_var for its reference line. When the load list names one thing,
        # that is what gets plotted, so the stage need not say it twice.
        self.metric = self.get_stage_element_silent("metric")
        if self.metric is None:
            if not self.metrics_to_load or len(self.metrics_to_load) != 1:
                raise ValueError(
                    f"{self.stage_str} has no 'metric' and 'metrics' is "
                    f"{self.metrics_to_load}; name which one to plot, or load "
                    f"exactly one."
                )
            self.metric = self.metrics_to_load[0]
        elif self.metric not in (self.metrics_to_load or []):
            raise ValueError(
                f"{self.stage_str} plots {self.metric!r} but 'metrics' is "
                f"{self.metrics_to_load}; add it to the load list."
            )

    def make_figures(self, data: FigData) -> None:
        for split in self.splits:
            fig = self.make_figure(data, split.name)
            self.save_fig(fig, split, model=None, metric=self.metric)

    def make_figure(self, data: FigData, split_name: str):
        return figs.plot_per_ell_metric(
            data.ells,
            fcfg.metrics_for(data, split_name, self.models, self.metric),
            self.metric,
            styles=self.styles(),
            labels=self.labels(),
            display=fcfg.display_for_dl(self.cfg_figs, self.display_override),
            figure=fcfg.figure_kwargs(
                self.cfg_figs, override=self.figure_override),
            title=f"{fcfg.label_for_metric(self.cfg_figs, self.metric)} — {split_name}",
        )


class ComparePerEllMSEExecutor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_mse"):
        super().__init__(cfg, stage_str=stage_str)


class ComparePerEllTensionExecutor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_tension"):
        super().__init__(cfg, stage_str=stage_str)


class ComparePerEllR2Executor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_r2"):
        super().__init__(cfg, stage_str=stage_str)


class ComparePerEllAbsRelErrPctExecutor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_abs_rel_err_pct"):
        super().__init__(cfg, stage_str=stage_str)


class ModelVsTheoryWithTensionExecutor(PerModelFigsExecutor):
    """One model's mean and 1-sigma against theory, with a metric panel below.

    Per model rather than all-model: four overlapping sigma bands on one axis
    obscure more than they show.

    The stage names the lower panel's quantity with 'bottom_metric' rather than
    taking it by position from 'metrics', so the load list and the panel choice
    stay independent. 'metrics' must cover both pred_var (the band) and whatever
    bottom_metric names.
    """

    def __init__(self, cfg: DictConfig, stage_str="model_vs_theory_tension"):
        super().__init__(cfg, stage_str=stage_str)
        self.bottom_metric = self.get_stage_element("bottom_metric")
        needed = {"pred_var", self.bottom_metric}
        missing = needed - set(self.metrics_to_load or [])
        if missing:
            raise ValueError(
                f"{self.stage_str} needs {sorted(needed)} but 'metrics' is "
                f"{self.metrics_to_load}; add {sorted(missing)}."
            )
        self.band_alpha = self.get_stage_element_silent("band_alpha")

    def make_figure(self, data: FigData, split_name: str, model_key: str):
        return figs.plot_model_vs_theory_with_tension(
            data.ells,
            fcfg.preds_for(data, split_name, [model_key]),
            fcfg.metrics_for(data, split_name, [model_key],
                             ["pred_var", self.bottom_metric]),
            styles=self.styles(fill_alpha=self.band_alpha, only=model_key),
            labels=self.labels(only=model_key),
            display=fcfg.display_for_dl(self.cfg_figs, self.display_override),
            bottom_display=fcfg.display_for_metric(
                self.cfg_figs, self.bottom_metric, self.bottom_display_override),
            figure=fcfg.figure_kwargs(
                self.cfg_figs, canvas="figure_tall", override=self.figure_override),
            bottom_key=self.bottom_metric,
            theory=data.thry[split_name],
            theory_style=fcfg.theory_style(self.cfg_figs),
            title=f"{self.models[model_key].fig_model_name} — {split_name}",
        )


class ComparePerEllVarianceExecutor(ComparePerEllMetricExecutor):
    """pred_var for every model, with the theory variance as a reference line.

    The notebook's variance figure: the same plot as the other per-ell metrics
    plus a black thry_var line to compare against.

    On a fixed-cosmology split the theory has no spread, so thry_var is
    numerical residue rather than a variance. The reference line is dropped
    there rather than drawn at 1e-11 on a log axis.
    """

    def __init__(self, cfg: DictConfig, stage_str="plot_variance"):
        super().__init__(cfg, stage_str=stage_str)
        self.theory_metric = self.get_stage_element("theory_metric")
        if self.theory_metric not in self.metrics_to_load:
            raise ValueError(
                f"{self.stage_str} draws {self.theory_metric!r} as its reference "
                f"line but 'metrics' is {self.metrics_to_load}; add it."
            )

    def make_figure(self, data: FigData, split_name: str):
        # thry_var is identical for every model -- same theory, same split --
        # so it is read from whichever happens to be first. Read from the
        # artifact rather than recomputed from data.thry, so the figure and the
        # metrics file cannot disagree.
        first = next(iter(self.models))
        theory_metric = data.get(split_name, first, self.theory_metric)
        if figs.is_fixed(data.thry[split_name]):
            theory_metric = None

        return figs.plot_per_ell_metric(
            data.ells,
            fcfg.metrics_for(data, split_name, self.models, self.metric),
            self.metric,
            styles=self.styles(),
            labels=self.labels(),
            display=fcfg.display_for_metric(
                self.cfg_figs, self.metric, self.display_override),
            figure=fcfg.figure_kwargs(
                self.cfg_figs, override=self.figure_override),
            theory_metric=theory_metric,
            theory_style=fcfg.theory_style(self.cfg_figs),
            title=f"{fcfg.label_for_metric(self.cfg_figs, self.metric)} — {split_name}",
        )


class RealizationFigsExecutor(PerModelFigsExecutor):
    """One realization's spectrum against theory, per model.

    Subclasses supply which realizations: fixed indices from config, or ranked
    per model from the data. Output paths need a discriminator beyond split and
    model, since each model emits several figures per split.
    """
    THRY_BAND_OPTS = ("thry_var", "cosm_var")

    def __init__(self, cfg, stage_str=None):
        if stage_str is None:
            raise NotImplementedError("This class is intended as an abstract class.")
        super().__init__(cfg, stage_str=stage_str)

        self.thry_band_src = self.get_stage_element_silent("thry_band")
        if self.thry_band_src is not None:
            if self.thry_band_src not in self.THRY_BAND_OPTS:
                raise ValueError(f"{self.stage_str}: compare_to must be one of "
                                 f"{self.THRY_BAND_OPTS} or null, got {self.thry_band_src!r}")
            if (self.thry_band_src == "thry_var"
                    and "thry_var" not in (self.metrics_to_load or [])):
                raise ValueError(f"{self.stage_str} compares to thry_var but "
                                 f"'metrics' is {self.metrics_to_load}; add it.")

    def make_figures(self, data: FigData) -> None:
        for split in self.splits:
            for model_key, model_cfg in self.models.items():
                for tag, idx, note, tag_label in self.realizations(
                        data, split.name, model_key):
                    fig = self.make_figure(data, split.name, model_key, idx,
                                           note, tag_label)
                    self.save_fig(fig, split, model=model_cfg, realization=tag)

    def realizations(self, data, split_name, model_key):
        """Which realizations to draw: [(tag, index, note, tag_label), ...].

        tag        path discriminator, since each model emits several figures
                   per split
        index      row of the prediction bundle to draw
        note       in-figure annotation, or None
        tag_label  parenthetical for the title, or None
        """
        raise NotImplementedError

    def get_thry_band(self, data, split_name, model_key, idx):
        """1-sigma band around this realization's theory, or None."""
        if self.thry_band_src is None:
            return None
        thry = np.atleast_2d(data.thry[split_name])
        target = thry[0] if thry.shape[0] == 1 else thry[idx]
        if self.thry_band_src == "cosm_var":
            ells = np.asarray(self.ells)
            var = 2.0 / (2 * ells + 1) * target**2
        else:  # thry_var
            if figs.is_fixed(data.thry[split_name]):
                logger.info(f"{split_name} is fixed-cosmology; no thry_var band.")
                return None
            var = data.get(split_name, model_key, "thry_var")
        return np.sqrt(var)

    def make_figure(self, data: FigData, split_name: str, model_key: str,
                    idx: int, note: str = None, tag_label: str = None):
        title = f"{self.models[model_key].fig_model_name} — {split_name} #{idx}"
        if tag_label:
            title = f"{title} ({tag_label})"
        return figs.plot_realization(
            data.ells,
            {model_key: data.get(split_name, model_key, "pred")},
            realization=idx,
            styles=self.styles(only=model_key),
            labels=self.labels(only=model_key),
            display=fcfg.display_for_dl(self.cfg_figs, self.display_override),
            figure=fcfg.figure_kwargs(self.cfg_figs, override=self.figure_override),
            theory=data.thry[split_name],
            theory_style=fcfg.theory_style(self.cfg_figs),
            annotation=note,
            annotation_style=fcfg.annotation_style(self.cfg_figs),
            title=title,
            thry_band=self.get_thry_band(data, split_name, model_key, idx)
        )


class FixedRealizationFigsExecutor(RealizationFigsExecutor):
    """The same realization indices for every model, so panels are comparable."""

    def __init__(self, cfg: DictConfig, stage_str="plot_realizations"):
        super().__init__(cfg, stage_str=stage_str)
        idxs = self.get_stage_element("realizations")
        self.indices = [idxs] if isinstance(idxs, int) else list(idxs)

    def realizations(self, data, split_name, model_key):
        return [(f"sim{i:04d}", i, None, None) for i in self.indices]


class RankedRealizationFigsExecutor(RealizationFigsExecutor):
    """The N best and N worst realizations per model, by a per-sim metric.

    Best and worst are tagged distinctly in the path so they can be shown
    separately: eight traces on one axis leaves a reader unable to tell which
    end is which.
    """

    def __init__(self, cfg: DictConfig, stage_str="plot_realizations_ranked"):
        super().__init__(cfg, stage_str=stage_str)
        self.rank_metric = self.get_stage_element("rank_metric")
        self.n_show = self.get_stage_element("n_show")
        self.in_per_sim_metric = self.assets_in["per_sim_metric"]
        self.rank_abs = bool(self.get_stage_element_silent("rank_abs"))

    def realizations(self, data, split_name, model_key):
        scores = self.get_rank_scores(split_name, model_key)
        if self.rank_abs:
            scores = np.abs(scores)
        order = np.argsort(scores)
        label = fcfg.label_for_metric(self.cfg_figs, self.rank_metric)

        out = []
        for bucket, idxs in (("best", order[:self.n_show]),
                             ("worst", order[-self.n_show:][::-1])):
            for i, idx in enumerate(idxs):
                idx = int(idx)
                out.append((
                    f"{bucket}{i}_sim{idx:04d}",
                    idx,
                    f"{label}: {scores[idx]:.3g}",
                    f"among {bucket}",
                ))
        return out

    def get_rank_scores(self, split_name: str, model_key: str) -> np.ndarray:
        """Per-sim scores for ranking, shape (N,).

        Read from the artifact rather than recomputed, so the realization a
        figure calls "worst" is worst by the same number a table would quote.
        """
        model = self.models[model_key]
        context = dict(
            split=split_name,
            working=model.working_dir,
            epoch=model.use_epoch,
            metric=self.rank_metric,
        )
        with self.name_tracker.set_contexts(context):
            return np.asarray(self.in_per_sim_metric.read())


class PredictionCloudExecutor(PerModelFigsExecutor):
    """Every realization of one model as a faint trace, with an error cloud below.

    Per model: N traces are dense enough on their own.

    Two normalizations for the lower panel -- by theory, and by theory's spread
    across realizations. The second is meaningless on a fixed-cosmology split,
    where that spread is numerical residue, so it is skipped there. That makes
    the loop data-dependent and therefore the executor's rather than config's.

    Peak positions are found here and passed to the figure as ell values: peak
    fitting is analysis, and keeping it out of figures.py leaves that module
    free of project dependencies.
    """

    NORMALIZATIONS = {
        "thry": "err_norm_thry",            # -> metric_display key
        "sd_thry": "err_norm_sd_thry",
    }

    def __init__(self, cfg: DictConfig, stage_str="plot_prediction_cloud"):
        super().__init__(cfg, stage_str=stage_str)
        self.band_alpha = self.get_stage_element_silent("band_alpha")
        self.mark_peaks = bool(self.get_stage_element_silent("mark_peaks"))

    def make_figures(self, data: FigData) -> None:
        for split in self.splits:
            thry = data.thry[split.name]
            norms = ["thry"]
            if not figs.is_fixed(thry):
                norms.append("sd_thry")
            else:
                logger.info(
                    f"{split.name} has one fixed theory spectrum; "
                    f"skipping the sd_thry normalization."
                )

            for model_key, model_cfg in self.models.items():
                for norm in norms:
                    fig = self.make_figure(data, split.name, model_key, norm)
                    self.save_fig(fig, split, model=model_cfg, norm=norm)

    def make_figure(self, data: FigData, split_name: str, model_key: str,
                    norm: str):
        thry = np.atleast_2d(data.thry[split_name])
        pred = np.atleast_2d(data.get(split_name, model_key, "pred"))

        theory_peaks, pred_peaks = self.find_peaks(pred, thry)

        return figs.plot_prediction_cloud(
            data.ells, pred, self.normalized_errors(pred, thry, norm),
            style=self.styles(fill_alpha=self.band_alpha, only=model_key)[model_key],
            label=self.models[model_key].fig_model_name,
            display=fcfg.display_for_dl(self.cfg_figs, self.display_override),
            bottom_display=fcfg.display_for_metric(
                self.cfg_figs, self.NORMALIZATIONS[norm],
                self.bottom_display_override),
            cloud_style=fcfg.cloud_style(self.cfg_figs),
            figure=fcfg.figure_kwargs(
                self.cfg_figs, canvas="figure_cloud", override=self.figure_override),
            theory=thry,
            theory_style=fcfg.theory_style(self.cfg_figs),
            theory_peaks=theory_peaks,
            pred_peaks=pred_peaks,
            peak_style=fcfg.peak_style(self.cfg_figs, "cloud_peak_style"),
            title=f"{self.models[model_key].fig_model_name} — {split_name}",
        )

    def normalized_errors(self, pred, thry, norm: str) -> np.ndarray:
        """(N, n_ell) prediction error, normalized as `norm` says.

        Theory is broadcast when a split carries one fixed spectrum.
        """
        thry_matched = thry if thry.shape[0] == pred.shape[0] else thry[:1]

        if norm == "thry":
            denom = thry_matched
        elif norm == "sd_thry":
            denom = thry.std(0)
        else:
            raise ValueError(f"unknown normalization {norm!r}")

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(
                pred - thry_matched, denom,
                out=np.full_like(pred, np.nan, dtype=float),
                where=(np.asarray(denom) != 0),
            )

    def find_peaks(self, pred, thry):
        if not self.mark_peaks:
            return None, None
        theory_rows = thry[:1] if figs.is_fixed(thry) else thry
        return (crits_for_rows(theory_rows, self.ells, "theory", logger),
                crits_for_rows(pred, self.ells, "prediction", logger))


class R2DecompositionExecutor(PerModelFigsExecutor):
    """One model's R^2 budget: the ceiling, and the losses that eat into it.

    R^2 = r^2 - (A-r)^2 - b^2, so the figure stacks the two losses downward
    from the ceiling and lands on R^2. What it shows is WHERE a method loses
    ground -- miscalibrated amplitude, or systematic bias -- which the R^2
    curve alone cannot distinguish.

    An exemplar panel above puts one realization's spectrum on the same ell
    axis: the budget is a statistic over the split and cannot say which spectral
    feature a dip belongs to. Set exemplar: null in the stage to drop it.

    Per model, since the terms decompose one model's R^2 and four stacked
    budgets on one axis do not read. The stage names the terms rather than
    hard-coding them, so the same executor serves a different decomposition.
    """

    def __init__(self, cfg: DictConfig, stage_str="plot_r2_decomposition"):
        super().__init__(cfg, stage_str=stage_str)
        self.ceiling_metric = self.get_stage_element("ceiling_metric")
        self.loss_metrics = list(self.get_stage_element("loss_metrics"))
        self.total_metric = self.get_stage_element("total_metric")

        needed = {self.ceiling_metric, self.total_metric, *self.loss_metrics}
        missing = needed - set(self.metrics_to_load or [])
        if missing:
            raise ValueError(
                f"{self.stage_str} needs {sorted(needed)} but 'metrics' is "
                f"{self.metrics_to_load}; add {sorted(missing)}."
            )

        # Which realization the context panel shows; null drops the panel.
        self.exemplar = self.get_stage_element_silent("exemplar")
        self.exemplar_display_override = \
            self.get_stage_element_silent("exemplar_display")

    def make_figure(self, data: FigData, split_name: str, model_key: str):
        get = lambda name: data.get(split_name, model_key, name)

        exemplar = None
        canvas = "figure"
        if self.exemplar is not None:
            thry = np.atleast_2d(data.thry[split_name])
            pred = np.atleast_2d(get("pred"))
            idx = int(self.exemplar)
            # A fixed-cosmology split carries one distinct theory row.
            exemplar = (thry[0] if thry.shape[0] == 1 else thry[idx], pred[idx])
            canvas = "figure_wide"

        return figs.plot_r2_decomposition(
            data.ells,
            get(self.ceiling_metric),
            {name: get(name) for name in self.loss_metrics},
            get(self.total_metric),
            style=self.styles(only=model_key)[model_key],
            label=self.models[model_key].fig_model_name,
            display=fcfg.display_for_metric(
                self.cfg_figs, "r2_decomposition", self.display_override),
            loss_styles=fcfg.loss_styles(self.cfg_figs, self.loss_metrics),
            budget_style=fcfg.budget_style(self.cfg_figs),
            figure=fcfg.figure_kwargs(
                self.cfg_figs, canvas=canvas, override=self.figure_override),
            ceiling_label=fcfg.label_for_metric(self.cfg_figs, self.ceiling_metric),
            exemplar=exemplar,
            exemplar_display=fcfg.display_for_dl(
                self.cfg_figs, self.exemplar_display_override),
            theory_style=fcfg.theory_style(self.cfg_figs),
            title=f"{self.models[model_key].fig_model_name} — {split_name}",
        )


class PairedRealizationPeaksExecutor(PerModelFigsExecutor):
    """A few realizations of one model, each against its own matched theory.

    What it shows is peak POSITION: the ticks at the two edges make a shifted
    acoustic peak visible as a horizontal offset between two marks of the same
    colour, which is hard to see in the curves themselves.

    Capped at n_show realizations -- the figure is about a handful of examples,
    not the ensemble. The cloud figure is the one that shows all of them.
    """

    def __init__(self, cfg: DictConfig, stage_str="plot_paired_realizations"):
        super().__init__(cfg, stage_str=stage_str)
        self.n_show = self.get_stage_element("n_show")
        self.mark_peaks = bool(self.get_stage_element_silent("mark_peaks"))
        self.cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def make_figure(self, data: FigData, split_name: str, model_key: str):
        thry = np.atleast_2d(data.thry[split_name])
        pred = np.atleast_2d(data.get(split_name, model_key, "pred"))

        n = min(self.n_show, pred.shape[0])
        pred = pred[:n]
        # A fixed-cosmology split carries one distinct theory row; repeat it so
        # the pairing by index still holds.
        thry = np.repeat(thry[:1], n, axis=0) if thry.shape[0] == 1 else thry[:n]

        theory_peaks = pred_peaks = None
        if self.mark_peaks:
            theory_peaks = crits_for_rows(thry, self.ells, "theory",
                                          logger, keep_none=True)
            pred_peaks = crits_for_rows(pred, self.ells, "prediction",
                                        logger, keep_none=True)

        theory_kwargs, pred_kwargs = fcfg.paired_realization_style(self.cfg_figs)

        return figs.plot_paired_realizations(
            data.ells, thry, pred,
            colors=self.cycle_colors,
            display=fcfg.display_for_dl(self.cfg_figs, self.display_override),
            theory_kwargs=theory_kwargs,
            pred_kwargs=pred_kwargs,
            figure=fcfg.figure_kwargs(
                self.cfg_figs, canvas="figure_wide", override=self.figure_override),
            theory_peaks=theory_peaks,
            pred_peaks=pred_peaks,
            peak_style=fcfg.peak_style(self.cfg_figs, "paired_peak_style"),
            title=f"{self.models[model_key].fig_model_name} — {split_name}",
        )

