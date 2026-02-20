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
from cmbml.analysis import LossPlotExecutor
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
    # ShowRelativeError,
    # ShowCosmicFractions,
    # ShowChiSquare,
    BundlePredExecutor, 
    BundleTrueExecutor,
    ShowTrebleEll
    )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps")
def run_train(cfg):
    logger.debug(f"Running on simulations in {__file__}")
    pipes = [
            #  HydraConfigCheckerExecutor,
             MakeLutExecutor,
             FindDatasetStatsParallelExecutor,
             PreprocessMapsExecutor,
             PreprocessSpectraExecutor,
             CheckGPULoadExecutor,
             TrainingExecutor,
             PredictExecutor,
             PowerSpectrumAnalysisExecutor,
            #  PerSplitAnalysis,
            #  ShowRelativeError,
            #  ShowCosmicFractions,
            #  ShowChiSquare,
             ShowPredictionsExecutor,
             ShowBWPredictionsExecutor,
            #  ShowIndivPredictionsExecutor,
             LossPlotExecutor,
            #  ErrorAnalysisExecutor,
            #  ShowPredictionsWithChiSquareExecutor,
             BundleTrueExecutor,
             BundlePredExecutor,
             ShowTrebleEll
             ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_on_planck")
def run_planck(cfg):
    logger.debug(f"Running on Ref in {__file__}")
    pipes = [
            #  HydraConfigCheckerExecutor,
             PreprocessMapsExecutor,
             PreprocessSpectraExecutor,
             PredictExecutor,
             ShowPredictionsExecutor,
             PowerSpectrumAnalysisExecutor,
             BundleTrueExecutor,
             BundlePredExecutor
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
    run_train()
    run_planck()
