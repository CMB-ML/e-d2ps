import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from d2ps_nn.ex import (
    # PrepGalNILCMaskExecutor,
    # PrepPSMaskExecutor,
    # PrepGalCutMaskExecutor,
    # ComposeMasksExecutor,
    # MakePosEncLutExecutor,
    MakeLutExecutor,
    FindDatasetStatsParallelExecutor,
    # FindPxMapStatsExecutor,
    # GetMonopoleExecutor,
    # ArrangeMasksExecutor
)

from d2ps_nn.ex import (
    # FindSpectraStatsExecutor,
    PreprocessMapsExecutor,
    # PreprocessMaskExecutor,
)

from d2ps_nn.ex import PreprocessSpectraExecutor

from d2ps_nn.ex import ModelInitExecutor

from d2ps_nn.ex import (
    TrainingExecutor,
    LossPlotExecutor,
    LossDeltaPlotExecutor,
)

from d2ps_nn.ex import PredictExecutor


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_1_prelim_px")
def run_1_prelim_px(cfg):
    logger.info(f"Running prelim_px in {__file__}")
    pipes = [
        ## For using masked regions in D2PS
        # PrepPSMaskExecutor,
        # PrepGalNILCMaskExecutor,
        # PrepGalCutMaskExecutor,
        # ComposeMasksExecutor,

        MakeLutExecutor,
        FindDatasetStatsParallelExecutor,

        ## For other specific settings
        # MakePosEncLutExecutor,
        # FindPxMapStatsExecutor,

        ## For using masked regions in D2PS
        # GetMonopoleExecutor,
        # ArrangeMasksExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_2_prelim_ps")
def run_2_prelim_ps(cfg):
    logger.info(f"Running prelim_ps in {__file__}")
    pipes = [
        # FindSpectraStatsExecutor,  # Only when standardizing spectra
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_3_prep_px")
def run_3_prep_px(cfg):
    logger.info(f"Running prep_px in {__file__}")
    pipes = [
        PreprocessMapsExecutor,
        # PreprocessMaskExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_4_prep_ps")
def run_4_prep_ps(cfg):
    logger.info(f"Running prep_ps in {__file__}")
    pipes = [
        PreprocessSpectraExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_d2ps_5_make_models")
def run_5_make_models(cfg):
    logger.info(f"Running prep_run in {__file__}")
    pipes = [
        ModelInitExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_6")
def run_6_train(cfg):
    if cfg.model.n_levels == 7 and cfg.nside == 512:
        raise ValueError("Seven encoder levels are not supported at Nside=512.")
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        TrainingExecutor,
        LossPlotExecutor,
        LossDeltaPlotExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="cfg_7")
def run_7_infer(cfg):
    logger.info(f"Running models_run in {__file__}")
    pipes = [
        PredictExecutor,
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
        log_maker.copy_hydra_run_to_dataset_log()
    logger.info("Pipeline completed.")


if __name__ == "__main__":
    run_1_prelim_px()
    # run_2_prelim_ps()
    run_3_prep_px()
    run_4_prep_ps()
    run_5_make_models()
    run_6_train()
    run_7_infer()
