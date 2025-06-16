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
from cmbml.sims import MaskCreatorExecutor
from d2ps_nn import (
    MakeLutExecutor,
    FindDatasetStatsParallelExecutor,
    PreprocessMapsExecutor,
    PreprocessSpectraExecutor,
    TrainingExecutor,
    PredictExecutor,
    PSBundleExecutor,
    CheckGPULoadExecutor,
    ShowChosenEllsExecutor,
    PowerSpectrumAnalysisExecutor,
    PowerSpectrumAggExecutor,
    ShowPredictionsExecutor,
    LossPlotExecutor
    )
from cmbml.analysis import (
    MakeRealPowerSpectrumExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps")
def run(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    # pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # # Cleaning: Preprocessing
    pipeline_context.add_pipe(MakeLutExecutor)
    pipeline_context.add_pipe(FindDatasetStatsParallelExecutor)
    pipeline_context.add_pipe(PreprocessMapsExecutor)
    pipeline_context.add_pipe(PreprocessSpectraExecutor)

    pipeline_context.add_pipe(ShowChosenEllsExecutor)
    pipeline_context.add_pipe(CheckGPULoadExecutor)
    pipeline_context.add_pipe(ShowChosenEllsExecutor)
    pipeline_context.add_pipe(CheckGPULoadExecutor)

    # Generating realization power spectra (not needed currently; predicting just the Theory PS)
    # pipeline_context.add_pipe(MaskCreatorExecutor)
    # pipeline_context.add_pipe(MakeRealPowerSpectrumExecutor)

    # Cleaning: Training and Prediction
    pipeline_context.add_pipe(TrainingExecutor)
    pipeline_context.add_pipe(PredictExecutor)

    pipeline_context.add_pipe(PowerSpectrumAnalysisExecutor)
    pipeline_context.add_pipe(PowerSpectrumAggExecutor)

    # Analysis and Visualization
    pipeline_context.add_pipe(ShowPredictionsExecutor)
    pipeline_context.add_pipe(LossPlotExecutor)

    # Bulk processing
    # pipeline_context.add_pipe(PSBundleExecutor)

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
     run()
