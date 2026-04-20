import logging

import numpy as np
from numpy.polynomial.legendre import legvander, legval
from numpy.polynomial.chebyshev import chebvander, chebval

from omegaconf import DictConfig

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset, AssetWithPathAlts
from cmbml.core.asset_handlers import NumpyPowerSpectrum
# from d2ps_nn.utils.ps_scale import make_cl_lognorm_scale
from d2ps_nn.utils.ps_scale import make_ps_processor
from d2ps_nn.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class PreprocessGetSpectraCoefExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="prep_get_polycoef")

        self.out_cmb_ps_coef: Asset = self.assets_out["cmb_ps_coef"]
        out_coef_handler: NumpyPowerSpectrum

        self.in_cmb_ps: AssetWithPathAlts = self.assets_in["ps_bundle"]
        in_cmb_ps_handler: NumpyPowerSpectrum

        self.ells = parse_ells(cfg.model.use_ells)
        self.basis_kind = cfg.basis_kind
        self.degree = cfg.degree
        self.out_range = tuple(cfg.out_range)
        if self.degree < 0:
            raise ValueError(f"cfg.degree must be >= 0, got {self.degree}")
        self.skl_pipeline = None
        self.prepped_ell = None

    def setup_skl_pipeline(self):
        basis = self.basis_kind
        degree = self.degree
        if basis == "monomial":
            feat = PolynomialFeatures(degree=degree, include_bias=True)
        elif basis == "legendre":
            feat = LegendreFeatures(degree=degree)
        elif basis == "chebyshev":
            feat = ChebyshevFeatures(degree=degree)
        else:
            raise ValueError(f"Unknown basis: {basis}")

        model = make_pipeline(
            feat,
            LinearRegression(fit_intercept=False),
        )
        self.skl_pipeline = model

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        self.setup_skl_pipeline()
        self.prepped_ell = _scale_ell(self.ells, out_range=self.out_range)

        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, 
                      split: Split) -> None:
        # A split is composed of simulations
        # No need to handle fixed ps splits; those are only used for comparison
        ps_data = self.in_cmb_ps.read()
        coefs = []
        for sim in tqdm(split.iter_sims()):
            new_coefs = self.process_sim(ps_data[sim])
            coefs.append(new_coefs)
        coefs = np.array(coefs)
        self.out_cmb_ps_coef.write(data=coefs)

    def process_sim(self, cmb_ps) -> None:
        fit_model = self.skl_pipeline.fit(self.prepped_ell, cmb_ps)
        coefs = fit_model.named_steps['linearregression'].coef_
        return coefs


class LegendreFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1)
        return legvander(x, self.degree)


class ChebyshevFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1)
        return chebvander(x, self.degree)


def _scale_ell(ell, out_range=(0.0, 1.0)):
    r"""
    Linearly map ell from [ell_min, ell_max] --> [a, b]

    Parameters
    ----------
    ell : array-like
        Input ell values
    out_range : tuple (a, b)
        Desired output range

    Returns
    -------
    x : ndarray, shape (n, 1)
        Rescaled values
    """
    ell = np.asarray(ell)
    a, b = out_range

    ell_min = ell.min()
    ell_max = ell.max()

    # Normalize to [0, 1]
    x = (ell - ell_min) / (ell_max - ell_min)

    # Scale to [a, b]
    x = a + (b - a) * x

    return x.reshape(-1, 1)
