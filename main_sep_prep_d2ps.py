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
from d2ps_nn import (
    MakeLutExecutor,
    FindDatasetStatsParallelExecutor,
    PreprocessMapsExecutor,
    PreprocessSpectraExecutor,
    TrainingExecutor,
    PredictExecutor,
    CheckGPULoadExecutor,
    PowerSpectrumAnalysisExecutor,
    # PerSplitAnalysis,
    # PowerSpectrumAggExecutor,
    ShowPredictionsExecutor,
    ShowBWPredictionsExecutor,
    # ShowIndivPredictionsExecutor,
    # ErrorAnalysisExecutor,
    # ShowPredictionsWithChiSquareExecutor,
    LossPlotExecutor,
    # ShowRelativeError,
    # ShowCosmicFractions,
    # ShowChiSquare,
    BundlePredExecutor, 
    BundleTrueExecutor,
    ShowTrebleEll
    )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_sepprep1")
def prep_run(cfg):
    logger.info(f"Running prep_run in {__file__}")
    pipes = [
        MakeLutExecutor,
        FindDatasetStatsParallelExecutor,
        PreprocessMapsExecutor,
        PreprocessSpectraExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_sepprep2")
def train_runs(cfg):
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        TrainingExecutor,
        PredictExecutor,
        PowerSpectrumAnalysisExecutor,
        ShowPredictionsExecutor,
        ShowBWPredictionsExecutor,
        LossPlotExecutor,
        BundleTrueExecutor,
        BundlePredExecutor,
        ShowTrebleEll
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_sepprep1_planck")
def prep_run_planck(cfg):
    logger.debug(f"Running train in {__file__}")
    pipes = [
             PreprocessMapsExecutor,
             PreprocessSpectraExecutor,
             ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_sepprep2_planck")
def run_planck(cfg):
    logger.debug(f"Running train in {__file__}")
    pipes = [
             PredictExecutor,
             PowerSpectrumAnalysisExecutor,
             ShowPredictionsExecutor,
             ]
    run(cfg, pipes)


def run(cfg, pipes):
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
    prep_run()
    train_runs()
    prep_run_planck()
    run_planck()
