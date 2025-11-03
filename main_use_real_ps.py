"""
This script runs a simulation pipeline for generating simulations based on a configuration.

The pipeline consists of several steps, including checking configurations, 
creating needed precursor assets (noise cache, configs, and theory power spectra), 
and creating the simulations.

The script uses the Hydra library for configuration management.

To run the simulation pipeline, execute the `run_simulations()` function.

Example usage:
    python main_sims.py
"""

import logging
import hydra
from omegaconf import OmegaConf
from cmbml.core import PipelineContext, LogMaker
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import (
    MaskCreatorExecutor,
)
from cmbml.analysis import (
    CommonRealPostExecutor,
    MakeRealPowerSpectrumExecutor
)
from d2ps_nn import (
    PowerSpectrumAnalysisExecutor,
    PerSplitAnalysis,
    ShowRelativeError,
    ShowCosmicFractions,
    ShowChiSquare,
    ShowPredictionsExecutor,
    ShowBWPredictionsExecutor,
    ShowIndivPredictionsExecutor,
    ErrorAnalysisExecutor,
    ShowPredictionsWithChiSquareExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_real_ps")
def run_simulations(cfg):
    """
    Runs the simulation pipeline.

    Args:
        cfg: The configuration object.

    Raises:
        Exception: If an exception occurs during the pipeline execution.
    """
    logger.debug(f"Running {__name__} in {__file__}")

    print(OmegaConf.to_yaml(cfg))

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)
    pipes = [
        PowerSpectrumAnalysisExecutor,
        MakeRealPowerSpectrumExecutor,
        PerSplitAnalysis,
        ShowRelativeError,
        ShowCosmicFractions,
        ShowChiSquare,
        ShowPredictionsExecutor,
        ShowBWPredictionsExecutor,
        ShowIndivPredictionsExecutor,
        ErrorAnalysisExecutor,
        ShowPredictionsWithChiSquareExecutor
    ]

    for pipe in pipes:
        pipeline_context.add_pipe(pipe)

    pipeline_context.prerun_pipeline()

    had_exception = False
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        had_exception = True
        logger.exception("An exception occurred during the pipeline.", exc_info=e)
        raise e
    finally:
        if had_exception:
            logger.error("Pipeline failed.")
        else:
            logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()

if __name__ == "__main__":
    run_simulations()
