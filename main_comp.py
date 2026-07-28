import logging
import hydra
from omegaconf import OmegaConf
from cmbml.core import PipelineContext, LogMaker
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims.ex import (
    MaskCreatorExecutor,
)
from cmbml.analysis.ex import (
    CommonRealPostExecutor,
    MakeRealPowerSpectrumExecutor
)
from d2ps_nn.ex import (
    ShowPredictionsExecutor,
    ShowPredictionsPeaksExecutor,
    ShowTrebleEll,
    BundlePredExecutor,
    BundleSmoothExecutor,
    BundleSmoothTrueExecutor,
    BundleTrueExecutor,
    NewMetricExecutor,
    NewMetricSmExecutor,
    # NewMetricSmTrueExecutor,
    ShowPredictionsCloudsExecutor
)


logger = logging.getLogger(__name__)


# @hydra.main(version_base=None, config_path="cfg", config_name="config_thry_sm_ps")
# def run_thry_sm_baseline(cfg):
#     # print(OmegaConf.to_yaml(cfg))

#     pipes = [
#         # MaskCreatorExecutor,
#         # BundlePredExecutor,
#         BundleSmoothExecutor,
#         # NewMetricExecutor,
#         NewMetricSmExecutor,
#         # MakeRealPowerSpectrumExecutor,
#         # PowerSpectrumAnalysisExecutor,
#         # PerSplitAnalysis,
#         # ShowChiSquare,
#         # ShowRelativeError,
#         # ShowCosmicFractions,
#         # ShowPredictionsExecutor,
#         # ShowBWPredictionsExecutor,
#         # ShowIndivPredictionsExecutor,
#         # ErrorAnalysisExecutor,
#         # ShowPredictionsWithChiSquareExecutor,
#     ]
#     run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_true_sm_ps")
def run_sm_theory(cfg):
    # print(OmegaConf.to_yaml(cfg))

    pipes = [
        BundleSmoothTrueExecutor,
        NewMetricSmExecutor,
        ShowPredictionsCloudsExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_real_ps")
def run_real(cfg):
    # print(OmegaConf.to_yaml(cfg))

    pipes = [
        MakeRealPowerSpectrumExecutor,
        MaskCreatorExecutor,
        BundlePredExecutor,
        BundleTrueExecutor,
        BundleSmoothExecutor,
        NewMetricSmExecutor,
        ShowPredictionsCloudsExecutor,
        ShowPredictionsPeaksExecutor,
        # ShowTrebleEll,
    ]
    run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_pred_ps_cmbnncs")
def run_cmbnncs(cfg):
    pipes = [
        BundlePredExecutor,
        BundleSmoothExecutor,
        BundlePredExecutor,
        BundleTrueExecutor,
        BundleSmoothExecutor,
        NewMetricSmExecutor,
        ShowPredictionsCloudsExecutor,
        ShowPredictionsPeaksExecutor,
        # ShowTrebleEll,
    ]
    run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_other_comp")
def run_d2ps_comp(cfg):
    pipes = [
        BundlePredExecutor,
        BundleSmoothExecutor,
        BundlePredExecutor,
        BundleTrueExecutor,
        BundleSmoothExecutor,
        NewMetricSmExecutor,
        ShowPredictionsCloudsExecutor,
        ShowPredictionsPeaksExecutor,
        # ShowTrebleEll,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_pred_ps_pyilc")
def run_pyilc(cfg):
    pipes = [
        BundlePredExecutor,
        BundleSmoothExecutor,
        NewMetricSmExecutor,
        ShowPredictionsCloudsExecutor,
        BundlePredExecutor,
        BundleTrueExecutor,
        BundleSmoothExecutor,
        NewMetricSmExecutor,
        ShowPredictionsCloudsExecutor,
        ShowPredictionsPeaksExecutor,
        # ShowTrebleEll,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_real_ps_planck")
def run_real_pl(cfg):
    pipes = [
        MakeRealPowerSpectrumExecutor,
        ShowPredictionsExecutor,
        BundlePredExecutor
        ]
    run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_pred_ps_cmbnncs_pl")
def run_cmbnncs_planck(cfg):
    pipes = [
        ShowPredictionsExecutor,
        BundlePredExecutor
        ]
    run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_pred_ps_pyilc_pl")
def run_pyilc_planck(cfg):
    pipes = [
        ShowPredictionsExecutor,
        BundlePredExecutor
        ]
    run(cfg, pipes)


def run(cfg, pipes):
    """
    Args:
        cfg: The configuration object.

    Raises:
        Exception: If an exception occurs during the pipeline execution.
    """
    logger.debug(f"Running {__name__} in {__file__}")

    # print(OmegaConf.to_yaml(cfg))

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

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
    # run_sm_theory()
    run_real()
    # run_real_pl()
    # run_d2ps_comp()
    # run_cmbnncs()
    # run_cmbnncs_planck()
    run_pyilc()
    # run_pyilc_planck()
