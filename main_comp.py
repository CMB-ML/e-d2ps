import logging
import hydra
from cmbml.core import PipelineContext, LogMaker
from cmbml.sims.ex import (
    MaskCreatorExecutor,
)
from cmbml.analysis.ex import (
    MakeRealPowerSpectrumExecutor
)
from d2ps_nn.ex import (
    BundlePredExecutor,
    GetPolyDegreeExecutor,
    BundleSmoothExecutor,
    # BundleSmoothTrueExecutor,
    # GetPerEllMetricsExecutor,
    # GetPerSimMetricsExecutor,
    BundleTrueExecutor,
    # BundleMeanTheoryExecutor,
    # MakeMeanSpectrumExecutor,
    # GetScalarMetricsExecutor,
    # GetScalarUncertaintyExecutor,
    # ComparePerEllR2Executor,
    # ComparePerEllMSEExecutor,
    # ComparePerEllTensionExecutor,
    # ComparePerEllAbsRelErrPctExecutor,
    # ModelVsTheoryWithTensionExecutor,
    # ComparePerEllVarianceExecutor,
    # FixedSingleSimFigsExecutor,
    # RankedSingleSimFigsExecutor,
    # PredictionCloudExecutor,
    # R2DecompositionExecutor,
    # PairedRealizationPeaksExecutor,
    TestFPOverviewExecutor,
    TestVTMetricsExecutor,
    TestFPMetricsExecutor,
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_real")
def run_bundle_real(cfg):
    # print(OmegaConf.to_yaml(cfg))

    pipes = [
        MaskCreatorExecutor,
        MakeRealPowerSpectrumExecutor,
        BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        # GetPerEllMetricsExecutor,
        # GetScalarMetricsExecutor,
        # GetScalarUncertaintyExecutor,
        # GetPerSimMetricsExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_pyilc")
def run_bundle_pyilc(cfg):
    pipes = [
        BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        # GetPerEllMetricsExecutor,
        # GetScalarMetricsExecutor,
        # GetScalarUncertaintyExecutor,
        # GetPerSimMetricsExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_cmbnncs")
def run_bundle_cmbnncs(cfg):
    pipes = [
        BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        # GetPerEllMetricsExecutor,
        # GetScalarMetricsExecutor,
        # GetScalarUncertaintyExecutor,
        # GetPerSimMetricsExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_obs2ps")
def run_bundle_d2ps(cfg):
    pipes = [
        BundleTrueExecutor,
        BundlePredExecutor,
        GetPolyDegreeExecutor,
        BundleSmoothExecutor,
        # GetPerEllMetricsExecutor,
        # GetScalarMetricsExecutor,
        # GetScalarUncertaintyExecutor,
        # GetPerSimMetricsExecutor
    ]
    run(cfg, pipes)


# @hydra.main(version_base=None, config_path="cfg", config_name="pred_ps_mean_thry")
# def run_bundle_mean_thry(cfg):
#     pipes = [
#         MakeMeanSpectrumExecutor,
#         BundleMeanTheoryExecutor,
#         GetPolyDegreeExecutor,
#         BundleSmoothExecutor,
#         GetPerEllMetricsExecutor,
#         GetScalarMetricsExecutor,
#         GetScalarUncertaintyExecutor,
#         GetPerSimMetricsExecutor
#     ]
#     run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="produce_figs")
def run_all_model_analysis(cfg):
    pipes = [
        # ComparePerEllR2Executor,
        # ComparePerEllMSEExecutor,
        # ComparePerEllTensionExecutor,
        # ComparePerEllAbsRelErrPctExecutor,
        # ComparePerEllVarianceExecutor,
        # ModelVsTheoryWithTensionExecutor,
        # FixedSingleSimFigsExecutor,
        # RankedSingleSimFigsExecutor,
        # PredictionCloudExecutor,
        # R2DecompositionExecutor,
        # PairedRealizationPeaksExecutor,
        TestFPOverviewExecutor,
        # TestVTMetricsExecutor,
        # TestFPMetricsExecutor
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
    run_bundle_real()
    run_bundle_pyilc()
    run_bundle_cmbnncs()
    run_bundle_d2ps()
    # run_bundle_mean_thry()

    run_all_model_analysis()
