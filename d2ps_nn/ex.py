from .stage_executors.B_make_lut import MakeLutExecutor
from .stage_executors.B_make_pos_enc_lut import MakePosEncLutExecutor
from .stage_executors.C_find_dataset_stats_parallel import FindDatasetStatsParallelExecutor
from .stage_executors.C2_find_ps_stats_serial import FindSpectraStatsExecutor
from .stage_executors.C3_find_map_stats_serial import FindPxMapStatsExecutor
from .stage_executors.D_preprocess_maps_serial import PreprocessMapsExecutor
from .stage_executors.D2_preprocess_mask import PreprocessMaskExecutor
from .stage_executors.D3_preprocess_masks import PrepGalNILCMaskExecutor, PrepPSMaskExecutor, PrepGalCutMaskExecutor
from .stage_executors.D4_compose_masks import ComposeMasksExecutor
from .stage_executors.D5_get_monopole import GetMonopoleExecutor
from .stage_executors.E_preprocess_ps_serial import PreprocessSpectraExecutor
from .stage_executors.E1_prep_coef_get_coef import PreprocessGetSpectraCoefExecutor
from .stage_executors.E2_prep_coef_get_stats import PreprocessStatSpectraCoefExecutor
from .stage_executors.E3_prep_coef_process import PreprocessPrepSpectraCoefExecutor
from .stage_executors.E4_prep_cosmo_param import BundleParamsExecutor, PreprocessStatParamsExecutor, PreprocessPrepParamsExecutor
from .stage_executors.Fa_init_model import ModelInitExecutor
from .stage_executors.F_train_check_bn import BatchNormCheckExecutor
from .stage_executors.F_train import TrainingExecutor
from .stage_executors.F_train_overfit import OverfitTrainingExecutor, LRFindingTrainingExecutor
from .stage_executors.F_train_coefs import TrainingCoefExecutor, OverfitTrainingCoefExecutor
from .stage_executors.F_train_params import TrainingParamsExecutor, OverfitTrainingParamsExecutor
from .stage_executors.F_train_abn import AttBNTrainingExecutor
from .stage_executors.G_predict import PredictExecutor
from .stage_executors.G_predict_coefs import PredictCoefsExecutor
from .stage_executors.G_predict_params import PredictParamsExecutor
from .stage_executors.H_ps_analysis import PowerSpectrumAnalysisExecutor
from .stage_executors.I_ps_agg import PowerSpectrumAggExecutor
from .stage_executors.J_ps_error_analysis import ErrorAnalysisExecutor
from .stage_executors.K_per_split_analysis import PerSplitAnalysis

# from .stage_executors._check_model_size import CheckGPULoadExecutor
# from .stage_executors._show_chosen_ells import ShowChosenEllsExecutor
# from .stage_executors._loss_summary import LossSummaryExecutor
from .stage_executors._loss_plot import LossPlotExecutor, LossDeltaPlotExecutor, LearningRatePlotExecutor
# from .stage_executors._rel_update_plot import RelativeUpdatePlotExecutor
# from .stage_executors._show_predictions import ShowPredictionsExecutor
# from .stage_executors._show_predictions_peaks import ShowPredictionsPeaksExecutor
# from .stage_executors._show_predictions_bw import ShowBWPredictionsExecutor
# from .stage_executors._show_predictions_indiv import ShowIndivPredictionsExecutor
# from .stage_executors._show_predictions_error_analysis import ShowPredictionsWithChiSquareExecutor

# from .stage_executors.__show_rel_err_old import ShowRelativeError
# from .stage_executors.__show_cosmic_fractions import ShowCosmicFractions
# from .stage_executors.__show_chi_sq import ShowChiSquare

# from .stage_executors.__show_petroff_like import ShowPetroffSingle
# from .stage_executors.__show_petroff_like import ShowPetroffSet
# from .stage_executors.__show_petroff_like import ShowPetroffSetBinned

from .stage_executors.Y_get_best_poly_deg import GetPolyDegreeExecutor
from .stage_executors.Z_bundle import BundlePredExecutor, BundleTrueExecutor
from .stage_executors.Z_bundle_maps import BundleMapsExecutor
# from .stage_executors._show_treble_ell import ShowTrebleEll
from .stage_executors.Z_bundle_smooth_ps import BundleSmoothExecutor, BundleSmoothTrueExecutor

# from .stage_executors._new_metric import NewMetricExecutor, NewMetricSmExecutor
# from .stage_executors._cloud import ShowPredictionsCloudsExecutor

from .stage_executors.ZZ_per_ell_metrics import GetPerEllMetricsExecutor
from .stage_executors.ZZ_per_sim_scalar_metrics import GetPerSimMetricsExecutor
from .stage_executors.ZZZ_base_figs import (
    ComparePerEllAbsRelErrPctExecutor,
    ComparePerEllMSEExecutor,
    ComparePerEllTensionExecutor,
    ComparePerEllR2Executor,
    ModelVsTheoryWithTensionExecutor,
    ComparePerEllVarianceExecutor,
    FixedRealizationFigsExecutor,
    RankedRealizationFigsExecutor,
    PredictionCloudExecutor,
    R2DecompositionExecutor,
    PairedRealizationPeaksExecutor
    )
