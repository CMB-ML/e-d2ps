from typing import Dict, List, NamedTuple, Callable
import logging
from pathlib import Path

from functools import partial
from multiprocessing import Pool, Manager
import numpy as np

from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset, AssetWithPathAlts, GenericHandler
from cmbml.analysis.statistics import get_func
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.core.asset_handlers.ps_handler import NumpyPowerSpectrum
from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.get_epochs import get_epochs_with_best


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    pred_asset: FrozenAsset
    base_asset: FrozenAsset
    baseline_label: str
    split_name: str
    sim_num: str
    epoch: int
    ells: np.ndarray
    use_alt_path: bool


class PowerSpectrumAnalysisExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_analysis")

        self.out_report: Asset = self.assets_out["report"]
        out_report_handler: PandasCsvHandler

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        # self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_ps_pred: Asset = self.assets_in["auto_pred"]
        in_ps_handler: NumpyPowerSpectrum

        self.stat_func_dict = self.cfg.model.analysis.metrics.ps_functions
        self.stat_funcs = self.get_stat_funcs()

        self.num_processes = cfg.model.analysis.metrics.ps_operations.num_processes
        self.ells = parse_ells(cfg.model.use_ells)
        self._model_epochs = cfg.use_epochs

    @property
    def model_epochs(self):
        """
        Returns the epochs to use for prediction.
        If the model asset has a 'best' epoch, it will be included in the list.
        """
        return self._model_epochs

    def execute(self) -> None:
        self._model_epochs = get_epochs_with_best(self._model_epochs, self.in_ps_pred, self.name_tracker)
        logger.debug(f"Running {self.__class__.__name__} execute().")
        
        # Create a method; 
        #   process_target needs a list of statistics functions, from the config file
        #   partial() makes a `process` that takes a single argument
        process = partial(process_target, stat_funcs=self.stat_funcs)
        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run a single task outside multiprocessing to catch issues quickly.
        self.try_a_task(process, tasks[0])

        self.run_all_tasks(process, tasks)

    def run_all_tasks(self, process, tasks):
        # Use multiprocessing to search through sims in parallel
        # A manager allows collection of information across separate threads
        with Manager() as manager:
            results = manager.list()
            # The Pool sets up the individual processes. 
            # Set processes according to the capacity of your computer
            with Pool(processes=self.num_processes) as pool:
                # Each result is the output of "process" running on each of the tasks
                for result in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
                    results.append(result)
            # Convert the results to a regular list after multiprocessing is complete
            #     and before the scope of the manager ends
            results_list = list(results)
        self.review_report(results_list)
        # Use the out_report asset to write all results to disk
        self.out_report.write(data=results_list)

    def review_report(self, report_list):
        found_error = False
        for res in report_list:
            if 'error' in res.keys():
                logger.error(f"Error in split {res['split']}, sim {res['sim']}, epoch {res['epoch']}: {res['error']}")
                found_error = True
        if found_error:
            raise OSError("Errors were found in the report. Please review the log for details.")

    def build_tasks(self):
        tasks = []
        for split in self.splits:
            for sim in split.iter_sims():
                for epoch in self.model_epochs:
                    context = dict(split=split.name, sim_num=sim, epoch=epoch)
                    with self.name_tracker.set_contexts(contexts_dict=context):
                        pred = self.in_ps_pred
                        pred = FrozenAsset(path=pred.path, handler=pred.handler)

                        # real = self.in_ps_real
                        # real = FrozenAsset(path=real.path, handler=real.handler)

                        thry = self.in_ps_theory
                        if split.ps_fidu_fixed:
                            thry_path = thry.path_alt
                        else:
                            thry_path = thry.path
                        thry = FrozenAsset(path=thry_path, handler=thry.handler)

                        tasks.append(TaskTarget(pred_asset=pred,
                                                base_asset=thry,
                                                baseline_label="thry",
                                                split_name=split.name, 
                                                sim_num=sim,
                                                epoch=epoch,
                                                ells=self.ells,
                                                use_alt_path=split.ps_fidu_fixed))

                        # tasks.append(TaskTarget(pred_asset=pred,
                        #                         base_asset=real,
                        #                         baseline_label="real",
                        #                         split_name=split.name, 
                        #                         sim_num=sim,
                        #                         epoch=epoch))

        return tasks

    def try_a_task(self, process, task: TaskTarget):
        """
        Get statistics for one sim (task) outside multiprocessing first, 
        to avoid painful debugging within multiprocessing.
        """
        res = process(task)
        if 'error' in res.keys():
            raise Exception(res['error'])

    def get_stat_funcs(self):
        stat_funcs = {}
        for name, details in self.stat_func_dict.items():
            func = get_func(details["func"])
            if 'kwargs' in details:
                func = partial(func, **details['kwargs'])
            stat_funcs[name] = func
        return stat_funcs


def process_target(task_target: TaskTarget, stat_funcs):
    """
    Each stat_func should accept true, pred, and **kwargs to catch other things
    """
    # if task_target.epoch is a string
    if isinstance(task_target.epoch, str) and task_target.epoch.startswith("best_"):
        # Convert "best_5" to 5
        epoch = int(task_target.epoch.split('_')[1])
    else:
        epoch = task_target.epoch
    res = dict(
        split=task_target.split_name,
        sim=task_target.sim_num,
        epoch=epoch,
        baseline=task_target.baseline_label
    )
    pred = task_target.pred_asset
    base = task_target.base_asset
    try:
        true_data = base.handler.read(base.path)
    except OSError as e:
        return {'error': f"Could not read true data from {base.path}. Error: {str(e)}", **res}
    try:
        pred_data = pred.handler.read(pred.path)
    except OSError as e:
        return {'error': f"Could not read pred data from {pred.path}. Error: {str(e)}", **res}

    true_data = true_data[task_target.ells]

    # Ensure that the shapes match
    if pred_data.shape[0] != true_data.shape[0]:
        # If so, use just the portion of true data needed.
        raise ValueError(
            f"Shape mismatch: true_data has shape {true_data.shape}, "
            f"pred_data has shape {pred_data.shape}. "
            "Ensure that the true data and predicted data have the same number of ell values."
        )
        # true_data = true_data[:pred_data.shape[0]]

    try:
        for stat_name, func in stat_funcs.items():
            res[stat_name] = func(true_data, pred_data)
    except Exception as e:
        res['error'] = f"Running '{stat_name}' caused '{str(e)}'. This stat function is defined in stat_funcs.yaml."

    return res
