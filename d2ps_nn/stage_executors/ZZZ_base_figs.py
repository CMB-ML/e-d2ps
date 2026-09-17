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

        self.ells = parse_ells(cfg.model.use_ells)

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
        """Write one figure, once per configured fig_type.

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
            for fig_type in self.cfg.fig_type:
                with self.name_tracker.set_context("fig_type", fig_type):
                    self.out_fig.write(fig=fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Appearance, from config
    # ------------------------------------------------------------------

    def styles(self, fill_alpha=None):
        return fcfg.styles_for_models(self.models, fill_alpha=fill_alpha)

    def labels(self):
        return fcfg.labels_for_models(self.models)


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
        if not self.metrics_to_load or len(self.metrics_to_load) != 1:
            raise ValueError(
                f"{self.stage_str} plots one metric, but 'metrics' is "
                f"{self.metrics_to_load}. Name exactly one, and add a stage per "
                f"metric you want plotted."
            )
        self.metric = self.metrics_to_load[0]

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
            display=fcfg.display_for_metric(
                self.cfg_figs, self.metric, self.display_override),
            figure=fcfg.figure_kwargs(
                self.cfg_figs, override=self.figure_override),
            title=f"{fcfg.label_for_metric(self.cfg_figs, self.metric)} — {split_name}",
        )


class ComparePerEllMSEExecutor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_mse"):
        super().__init__(cfg, stage_str=stage_str)


class ComparePerEllR2Executor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_r2"):
        super().__init__(cfg, stage_str=stage_str)


class ComparePerEllAbsRelErrPctExecutor(ComparePerEllMetricExecutor):
    def __init__(self, cfg: DictConfig, stage_str="plot_abs_rel_err_pct"):
        super().__init__(cfg, stage_str=stage_str)
