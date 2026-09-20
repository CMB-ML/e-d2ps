from typing import NamedTuple
import logging
from pathlib import Path

from functools import partial
from multiprocessing import Pool

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import (
    BaseStageExecutor,
    Asset,
    GenericHandler
    )
from cmbml.core.asset_handlers import Config # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils.map_fields_helper import map_field_str2int
from cmbml.utils.fits_inspection import get_header_cards
from d2ps_nn.utils.masks.mask_composition import gal_mask_label, provenance_cards
from d2ps_nn.utils.masks.masked_monopole import masked_mean, summarize


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class MonopoleTask(NamedTuple):
    obs_asset: FrozenAsset
    split_name: str
    sim_num: str


# Masks are loaded once per worker process rather than pickled with every task
_WORKER_MASKS = None

def _init_worker(masks):
    global _WORKER_MASKS
    _WORKER_MASKS = masks


class GetMonopoleExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str="get_monopole") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = list(self.instrument.dets.keys())
        self.gal_source = cfg.masks.gal_source
        self.gal_cut_cfg = cfg.masks.gal_cut

        self.in_obs_maps: Asset = self.assets_in["obs_maps"]
        self.in_mask: Asset = self.assets_in["mask"]
        self.out_monopole: Asset = self.assets_out["monopole"]
        in_obs_map_handler: HealpyMap
        in_mask_handler: HealpyMap
        out_monopole_handler: Config

        self.field_idx = map_field_str2int("I")
        self.n_workers = self.get_stage_element("n_workers")
        self.masks = {}

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.load_masks()        
        tasks = self.build_tasks()
        scan = partial(find_masked_means, freqs=self.channels, field_idx=self.field_idx)

        # One task in-process first, to catch problems outside multiprocessing
        _init_worker(self.masks)
        scan(tasks[0])

        with Pool(processes=self.n_workers,
                  initializer=_init_worker,
                  initargs=(self.masks,)) as pool:
            results = list(tqdm(pool.imap_unordered(scan, tasks), total=len(tasks)))

        summary = sift_monopoles(results, self.channels)
        for ch in self.channels:
            summary[ch]["mask"] = self.mask_info[ch]
            logger.info(f"{ch} GHz monopole: {summary[ch]['mean']:.4g} ± {summary[ch]['std']:.2g}")
        out = dict(
            channels=summary,
            splits=[split.name for split in self.splits],
        )
        with self.name_tracker.set_context("gal_mask_source", self.gal_source):
            self.out_monopole.write(data=out)

    def load_masks(self) -> None:
        self.mask_info = {}
        for ch in self.channels:
            gal_mask = gal_mask_label(ch, self.gal_source, self.gal_cut_cfg).lower()
            with self.name_tracker.set_contexts(dict(freq=ch, gal_mask=gal_mask)):
                mask = self.in_mask.read()[0].astype(bool)
                cards = get_header_cards(self.in_mask.path)
                path = str(self.in_mask.path)
            self.masks[ch] = mask
            self.mask_info[ch] = dict(
                path=path,
                f_sky=float(mask.mean()),
                header={k: v for k, v, _ in provenance_cards(cards)},
            )

    def build_tasks(self):
        tasks = []
        for split in self.splits:
            for sim in split.iter_sims():
                with self.name_tracker.set_contexts(dict(split=split.name, sim_num=sim)):
                    with self.name_tracker.set_context("freq", "{freq}"):
                        obs = FrozenAsset(path=self.in_obs_maps.path,
                                          handler=self.in_obs_maps.handler)
                tasks.append(MonopoleTask(obs_asset=obs, split_name=split.name, sim_num=sim))
        return tasks


def find_masked_means(task: MonopoleTask, freqs, field_idx):
    """Masked mean of one sim's obs map for each frequency."""
    obs = task.obs_asset
    res = {}
    for freq in freqs:
        m = obs.handler.read(str(obs.path).format(freq=freq))[field_idx]
        res[freq] = masked_mean(m, _WORKER_MASKS[freq])
    return res


def sift_monopoles(results_list, freqs):
    return {freq: summarize([r[freq] for r in results_list]) for freq in freqs}
