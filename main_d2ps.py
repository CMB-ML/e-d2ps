"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using <TODO: Name it>

The pipeline consists of the following steps:
1. <TODO: Steps here>

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_<TODO: Name it>.py
"""
import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from d2ps_nn.ex import (
    PrepGalNILCMaskExecutor,
    PrepPSMaskExecutor,
    PrepGalCutMaskExecutor,
    ComposeMasksExecutor,
    MakeLutExecutor,
    MakePosEncLutExecutor,
    FindDatasetStatsParallelExecutor,
    FindPxMapStatsExecutor,
    GetMonopoleExecutor,
    ArrangeMasksExecutor
)

from d2ps_nn.ex import (
    FindSpectraStatsExecutor,
    PreprocessMapsExecutor,
    PreprocessMaskExecutor,
)

from d2ps_nn.ex import PreprocessSpectraExecutor

from d2ps_nn.ex import ModelInitExecutor

from d2ps_nn.ex import (
    # CheckGPULoadExecutor,
    # OverfitTrainingExecutor,
    # RelativeUpdatePlotExecutor,
    # LRFindingTrainingExecutor,
    # LearningRatePlotExecutor,
    TrainingExecutor,
    LossPlotExecutor,
    LossDeltaPlotExecutor,
)

from d2ps_nn.ex import PredictExecutor

from d2ps_nn.ex import (
    BundlePredExecutor, 
    BundleTrueExecutor,
    GetPolyDegreeExecutor,
    BundleSmoothExecutor,
    GetPerEllMetricsExecutor,
    GetPerSimMetricsExecutor,
    ModelVsTheoryWithTensionExecutor,
    FixedSingleSimFigsExecutor,
    RankedSingleSimFigsExecutor,
    PredictionCloudExecutor,
    R2DecompositionExecutor,
    PairedRealizationPeaksExecutor
    )

from d2ps_nn.ex import AdjustZeroLevelExecutor

from d2ps_nn.ex import CheckDatasetExecutor


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_1_prelim_px")
def run_1_prelim_px(cfg):
    logger.info(f"Running prelim_px in {__file__}")
    pipes = [
        PrepPSMaskExecutor,
        PrepGalNILCMaskExecutor,
        PrepGalCutMaskExecutor,
        ComposeMasksExecutor,

        # MakeLutExecutor,
        # MakePosEncLutExecutor,
        # FindDatasetStatsParallelExecutor,
        # FindPxMapStatsExecutor,

        GetMonopoleExecutor,
        ArrangeMasksExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_2_prelim_ps")
def run_2_prelim_ps(cfg):
    logger.info(f"Running prelim_ps in {__file__}")
    pipes = [
        FindSpectraStatsExecutor,  # Only when standardizing spectra
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_3_prep_px")
def run_3_prep_px(cfg):
    logger.info(f"Running prep_px in {__file__}")
    pipes = [
        # PreprocessMapsExecutor,
        PreprocessMaskExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_4_prep_ps")
def run_4_prep_ps(cfg):
    logger.info(f"Running prep_ps in {__file__}")
    pipes = [
        # PreprocessSpectraExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_5_make_models")
def run_5_make_models(cfg):
    logger.info(f"Running prep_run in {__file__}")
    pipes = [
        ModelInitExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_8")
def run_8_prep_planck(cfg):
    logger.debug(f"Running train in {__file__}")
    pipes = [
             AdjustZeroLevelExecutor,
             PreprocessMapsExecutor,
            #  PreprocessSpectraExecutor,
             BundleTrueExecutor,
             ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_6")
def run_6_train(cfg):
    if cfg.model.n_levels == 7 and cfg.nside == 512:
        raise ValueError("Don't make this mistake again, please.")
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        # CheckGPULoadExecutor,
        # OverfitTrainingExecutor,
        # RelativeUpdatePlotExecutor,
        # LRFindingTrainingExecutor,
        # LearningRatePlotExecutor,
        TrainingExecutor,
        LossPlotExecutor,
        LossDeltaPlotExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_6")
def run_6b_show_dataset(cfg):
    pipes = [
        CheckDatasetExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_7")
def run_7_infer(cfg):
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        PredictExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_7B")
def run_7B_analysis(cfg):
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,  # A no-op, but simpler to use it instead of rearranging
        BundleSmoothExecutor,   # A no-op, but simpler to use it instead of rearranging
        GetPerEllMetricsExecutor,
        GetPerSimMetricsExecutor,
        ModelVsTheoryWithTensionExecutor,
        FixedSingleSimFigsExecutor,
        RankedSingleSimFigsExecutor,
        PredictionCloudExecutor,
        R2DecompositionExecutor,
        PairedRealizationPeaksExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_9")
def run_9_run_planck(cfg):
    logger.debug(f"Running train in {__file__}")
    pipes = [
             PredictExecutor,
             BundlePredExecutor,
             FixedSingleSimFigsExecutor,
             ]
    run(cfg, pipes)


def run(cfg, pipes):
    # scale_px = cfg.get("scale_px", "~~~~~")
    # scale_ps = cfg.get("scale_ps", "~~~~~")
    # optimizer = cfg.get("optimizer", "~~~~~")
    # learning_rate_final = cfg.get("learning_rate_final", "~~~~~")
    # adamwd_wd = cfg.get("adamw_wd", "~~~~~")
    # norm_kind = cfg.get("norm_kind", "~~~~~")
    # activation = cfg.get("activation", "~~~~~")
    # down_reductions = cfg.get("down_reductions", "~~~~~")
    # spatial_reductions = cfg.get("spatial_reduction", "~~~~~")
    # print(scale_px, scale_ps, optimizer, learning_rate_final, adamwd_wd, norm_kind, activation, down_reductions, spatial_reductions)

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)
    for pipe in pipes:
        pipeline_context.add_pipe(pipe)
    pipeline_context.prerun_pipeline()

    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    # run_1_prelim_px()
    # run_2_prelim_ps()
    # run_3_prep_px()
    # run_4_prep_ps()
    # run_5_make_models()
    run_8_prep_planck()
    # run_6_train()
    # run_7_infer()
    # run_7B_analysis()
    run_9_run_planck()
