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
    # PowerSpectrumAnalysisExecutor,
    # PerSplitAnalysis,
    # ShowChiSquare,
    # ShowRelativeError,
    # ShowCosmicFractions,
    # ShowPredictionsExecutor,
    # ShowBWPredictionsExecutor,
    # ShowIndivPredictionsExecutor,
    # ErrorAnalysisExecutor,
    # ShowPredictionsWithChiSquareExecutor,
    BundlePredExecutor,
    GetPolyDegreeExecutor,
    BundleSmoothExecutor,
    BundleSmoothTrueExecutor,
    GetMetricsExecutor,
    BundleTrueExecutor,
    BaseFigsExecutor
    # NewMetricExecutor,
    # NewMetricSmExecutor,
    # NewMetricSmTrueExecutor,
    # ShowPredictionsCloudsExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_true_sm_ps")
def run_sm_theory(cfg):
    # print(OmegaConf.to_yaml(cfg))

    pipes = [
        BundleSmoothTrueExecutor,
        # NewMetricSmExecutor,
        # ShowPredictionsCloudsExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_real")
def run_bundle_real(cfg):
    # print(OmegaConf.to_yaml(cfg))

    pipes = [
        # MaskCreatorExecutor,
        # MakeRealPowerSpectrumExecutor,
        BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        GetMetricsExecutor,
        # NewMetricExecutor,
        # NewMetricSmExecutor,
        # ShowPredictionsCloudsExecutor,
        # MakeRealPowerSpectrumExecutor,
        # PowerSpectrumAnalysisExecutor,
        # PerSplitAnalysis,
        # ShowChiSquare,
        # ShowRelativeError,
        # ShowCosmicFractions,
        # ShowPredictionsExecutor,
        # ShowBWPredictionsExecutor,
        # ShowIndivPredictionsExecutor,
        # ErrorAnalysisExecutor,
        # ShowPredictionsWithChiSquareExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_pyilc")
def run_bundle_pyilc(cfg):
    pipes = [
        # BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        GetMetricsExecutor,
        # NewMetricSmExecutor,
        # ShowPredictionsCloudsExecutor
        # PowerSpectrumAnalysisExecutor,
        # PerSplitAnalysis,
        # ShowChiSquare,
        # ShowRelativeError,
        # ShowCosmicFractions,
        # ShowPredictionsExecutor,
        # ShowBWPredictionsExecutor,
        # ShowIndivPredictionsExecutor,
        # ErrorAnalysisExecutor,
        # ShowPredictionsWithChiSquareExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_cmbnncs")
def run_bundle_cmbnncs(cfg):
    pipes = [
        # BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        GetMetricsExecutor,
        # NewMetricExecutor,
        # NewMetricSmExecutor,
        # ShowPredictionsCloudsExecutor
        # PowerSpectrumAnalysisExecutor,
        # PerSplitAnalysis,
        # ShowChiSquare,
        # ShowRelativeError,
        # ShowCosmicFractions,
        # ShowPredictionsExecutor,
        # ShowBWPredictionsExecutor,
        # ShowIndivPredictionsExecutor,
        # ErrorAnalysisExecutor,
        # ShowPredictionsWithChiSquareExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_obs2ps")
def run_bundle_d2ps(cfg):
    pipes = [
        # BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        GetMetricsExecutor,
        # NewMetricExecutor,
        # NewMetricSmExecutor,
        # ShowPredictionsCloudsExecutor
        # PowerSpectrumAnalysisExecutor,
        # PerSplitAnalysis,
        # ShowChiSquare,
        # ShowRelativeError,
        # ShowCosmicFractions,
        # ShowPredictionsExecutor,
        # ShowBWPredictionsExecutor,
        # ShowIndivPredictionsExecutor,
        # ErrorAnalysisExecutor,
        # ShowPredictionsWithChiSquareExecutor,
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="produce_figs")
def run_all_model_analysis(cfg):
    pipes = [
        BaseFigsExecutor
    ]
    run(cfg, pipes)



@hydra.main(version_base=None, config_path="cfg", config_name="config_real_ps_planck")
def run_real_pl(cfg):
    pipes = [
        MakeRealPowerSpectrumExecutor,
        # ShowPredictionsExecutor,
        BundlePredExecutor
        ]
    run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_pred_ps_cmbnncs_pl")
def run_cmbnncs_planck(cfg):
    pipes = [
        # ShowPredictionsExecutor,
        BundlePredExecutor
        ]
    run(cfg, pipes)

@hydra.main(version_base=None, config_path="cfg", config_name="config_pred_ps_pyilc_pl")
def run_pyilc_planck(cfg):
    pipes = [
        # ShowPredictionsExecutor,
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
    # run_bundle_real()
    # run_bundle_pyilc()
    # run_bundle_cmbnncs()
    # run_bundle_d2ps()

    run_all_model_analysis()

    # run_real_pl()
    # run_cmbnncs_planck()
    # run_pyilc_planck()

    # run_sm_theory()
