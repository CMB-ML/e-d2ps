import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
# from cmbml.analysis import LossPlotExecutor
from cmbml.sims import (
    ChainsConfigExecutor,
    ParamConfigExecutor,
    TheoryPSExecutor
    )
from sph2ps_nn.ex import (
    CheapRealizationPSExecutor,
    GatherSpectraExecutor,
    GatherParamsExecutor,
    PrepSpectraExecutor,
    PrepParamsExecutor
) 


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_sph2ps_sim")
def run_sim(cfg):
    logger.debug(f"Running on simulations in {__file__}")
    if cfg.model.sim.cmb.use_chains:
        pipes = [ChainsConfigExecutor]
    else:
        pipes = [ParamConfigExecutor]
    pipes.extend([
        TheoryPSExecutor,
        CheapRealizationPSExecutor
        ])
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_sph2ps_prep")
def run_prep(cfg):
    logger.debug(f"Running on simulations in {__file__}")
    pipes = [
        GatherSpectraExecutor,
        GatherParamsExecutor,
        PrepSpectraExecutor,
        PrepParamsExecutor
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
    # run_sim()
    run_prep()
