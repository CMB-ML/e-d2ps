import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.analysis import (
    MakeRealPowerSpectrumExecutor
)
from d2ps_nn.ex import (
    BundleMapsExecutor,
    BundlePredExecutor,
    BundleRealExecutor,
    BundleTrueExecutor
)
from d2ps_fcn.ex import (
    CheapMakeRealPowerSpectrumExecutor,
    ModelInitExecutor,
    TrainFCNModelExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_32_A_Real")
def run_A_real(cfg):
    # print(OmegaConf.to_yaml(cfg))
    logger.info(f"Running run_A_real in {__file__}")

    pipes = [
        CheapMakeRealPowerSpectrumExecutor,
        BundleRealExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_32_B_Prep")
def run_B_prep(cfg):
    # print(OmegaConf.to_yaml(cfg))
    logger.info(f"Running run_B_prep in {__file__}")

    pipes = [
        BundleTrueExecutor,
        BundleMapsExecutor,
        ModelInitExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_32_C_Train")
def run_C_fcn(cfg):
    # print(OmegaConf.to_yaml(cfg))
    logger.info(f"Running run_C_fcn in {__file__}")

    pipes = [
        TrainFCNModelExecutor
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
    # run_A_real()
    # run_B_prep()
    run_C_fcn()
