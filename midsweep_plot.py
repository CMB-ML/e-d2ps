"""
This script runs a portion of the pipeline, to be used while a sweep is already running.
"""
import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from d2ps_nn.ex import (
    LossPlotExecutor,
    PredictExecutor,
    ShowPredictionsPeaksExecutor,
    BundlePredExecutor, 
    BundleTrueExecutor,
    ShowTrebleEll,
    NewMetricExecutor,
    ShowPredictionsCloudsExecutor
    )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="midsweep_plot")
def predict_and_plot(cfg):
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        PredictExecutor,
        ShowPredictionsPeaksExecutor,
        BundlePredExecutor,
        BundleTrueExecutor,
        ShowTrebleEll,
        NewMetricExecutor,
        ShowPredictionsCloudsExecutor
    ]
    if cfg.run_inference_on == "test":
        pipes = [LossPlotExecutor, *pipes]
    run(cfg, pipes)


def run(cfg, pipes):
    print(cfg.working_dir)

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
    predict_and_plot()
