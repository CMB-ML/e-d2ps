import logging

# from tqdm import tqdm
import time

import healpy as hp
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_ps import (
    SpectrumAsset, 
    SpectrumAssetNames,
    norm_ps_type
)

from d2ps_fcn.fixed_dataset import FixedDataset
from d2ps_fcn.models.fcn_from_string import FCNFromString
from d2ps_fcn.scaler_hydra_helper import make_scaler_from_prep_spec
from d2ps_nn.utils.px_scale import make_px_processor, MapProcessor
from d2ps_fcn.dyn_prod_dataset import TrainingMapSampler, make_map_sampler
from d2ps_fcn.realization_spectra import make_real_spectra_maker


logger = logging.getLogger(__name__)


class TrainDynProdExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train_dyn_prod") -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.ells = parse_ells(cfg.model.use_ells)
        self.lmax = self.ells[-1]
        self.spectrum_type = norm_ps_type(cfg.spectrum_type)  # "theory" | "realization"

        # Assets
        self.in_train_cmb_ps:  Asset = self.assets_in["train_cmb_ps"]
        self.in_valid_map:     Asset = self.assets_in["valid_map"]
        self.in_cmb_map_stats: Asset = self.assets_in["cmb_stats"]
        self.in_rps_stats: Asset = self.assets_in["rps_stats"]
        self.in_initial_model: Asset = self.assets_in["initial_model"]
        self.in_resume_model:  Asset = self.assets_in["resume_model"]

        self.out_model:        Asset = self.assets_out["model"]
        self.out_loss_record:  Asset = self.assets_out["loss_record"]

        valid_trgt_names = SpectrumAssetNames(
            theory="valid_thry_ps",
            realization="valid_real_ps",
        )
        self.in_valid_trgt_asset = SpectrumAsset.select(
            self.assets_in,
            self.spectrum_type,
            name_map=valid_trgt_names,
        )

        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.valid_batch_size = cfg.model.train.valid_batch_size
        self.n_batches        = cfg.model.train.n_batches
        self.n_epochs         = cfg.model.train.n_epochs
        self.learning_rate    = cfg.model.train.learning_rate
        self.dtype            = self.dtype_mapping[cfg.model.dtype]
        self.earliest_best    = cfg.model.train.earliest_best
        self.checkpoint       = cfg.model.train.checkpoint_every
        self.extra_check      = cfg.model.train.extra_check
        self.model_init       = cfg.model.model_init
        self.out_power        = cfg.out_power
        self.activation       = cfg.model.activation
        self.fcn_string       = cfg.fcn_string

        self.sht_engine       = str(cfg.sim_engine).lower()
         
        logger.info(f"Producing {self.train_batch_size * self.n_batches} maps per epoch.")
        logger.info(f"Running for {self.n_epochs} epochs.")

        self.label_prep       = cfg.model.prep_label
        self.feature_prep     = cfg.model.prep_feature

        # Processor constructed in __init__ since it only wires asset references
        self.map_processor = make_px_processor(mode=self.feature_prep, cfg=cfg, stage_str='train_dyn_prod')

        # Scalers constructed here; fitted in execute()
        self.label_scaler   = make_scaler_from_prep_spec(self.label_prep)
        logger.info(f"map_processor type: {type(self.map_processor).__name__}")
        logger.info(f"label_scaler: {self.label_scaler}")

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.choose_device()

        # Load bank and fit scalers
        logger.debug(f"Loading bank of Dls")
        dl_bank = self.in_train_cmb_ps.read()           # (n_spectra, some_lmax)
        if dl_bank.shape[1] < self.lmax+1:
            raise ValueError("Dl bank does not have enough ell. Please regenerate it or change lmax.")
        dl_bank = dl_bank[:, :self.lmax+1]
        logger.info(f"Done loading bank of Dls")

        if self.spectrum_type == "theory":
            dl_for_labels = dl_bank[:, 2:]            # ells 2..lmax
            self.label_scaler.fit(dl_for_labels)
            n_label_ells = dl_for_labels.shape[1]
        else:  # "realization"
            rps_stats = self.in_rps_stats.read()      # (4, lmax+1)
            rps_for_labels = rps_stats[:, 2:]         # (4, lmax-1), ells 2..lmax
            self.label_scaler.fit_from_stats(rps_for_labels)
            n_label_ells = rps_for_labels.shape[1]

        if n_label_ells != len(self.ells):
            raise ValueError(
                f"Mismatch: {self.spectrum_type} labels have {n_label_ells} ells "
                f"but model expects {len(self.ells)} output channels."
            )
    
        # Build generation pipeline
        valid_split = self.splits.get("valid")
        spectra_estimator = None
        if self.spectrum_type == "realization":
            spectra_estimator = make_real_spectra_maker(
                engine=self.sht_engine,
                nside=self.nside,
                lmax=self.lmax,
                return_cl=False,
                return_dl=True,
                compute_device=str(self.device),
                output_device=str(self.device),   # keep label on-GPU to match maps
                dtype=self.dtype,                  # float32 for training (not float64)
            )

        sampler = TrainingMapSampler(
            inner=make_map_sampler(
                sim_engine=self.sht_engine,
                dl_bank=dl_bank,
                nside=self.nside,
                batch_size=self.train_batch_size,
                n_batches=self.n_batches,
                lmax=self.lmax,
                device=self.device,
                dtype=self.dtype,
            ),
            map_processor=self.map_processor,
            label_scaler=self.label_scaler,
            spectrum_kind=self.spectrum_type,
            spectra_estimator=spectra_estimator,
        )

        # Validation dataset (disk-based, existing machinery)
        valid_dataset = self.set_up_valid_dataset(valid_split)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
        )
        n_infer_batches = valid_split.n_sims // self.train_batch_size

        # Model init
        model = self.init_model()
        context = dict(model_num=self.model_init)
        with self.name_tracker.set_contexts(context):
            self.in_initial_model.read(model=model, strict=True)
        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")
        model = model.to(self.device)

        self._crosscheck_model_dims(model)

        loss_function = torch.nn.MSELoss(reduction='mean').to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        loss_record_headers = [
            'Epoch', 'Batch Loss',
            'Validation Pre Loss', 'Training Pre Loss',
            'Validation Loss', 'Training Loss',
            'Learning Rate',
        ]
        self.out_loss_record.write(data=loss_record_headers)

        all_batch_loss = []
        all_train_pre_loss = []
        all_valid_pre_loss = []
        all_train_pst_loss = []
        all_valid_pst_loss = []

        n_epoch_digits = len(str(self.n_epochs))

        for epoch in range(self.n_epochs):
            time_start = time.time()
            # Training — on-the-fly generated maps
            model.train()
            batch_loss = 0.0
            for maps, scaled_dl in sampler:
                optimizer.zero_grad(set_to_none=True)
                output = model(maps)
                loss = loss_function(output, scaled_dl)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            batch_loss /= len(sampler)
            all_batch_loss.append(batch_loss)

            # Evaluation
            model.eval()
            train_pre_loss = train_pst_loss = 0.0
            valid_pre_loss = valid_pst_loss = 0.0

            with torch.no_grad():
                # Train-space estimate: sample n_infer_batches batches from sampler
                train_pre_loss = 0.0
                train_pst_loss = 0.0
                for _ in range(n_infer_batches):
                    maps, scaled_dl = sampler.sample()
                    output = model(maps)
                    train_pre_loss += loss_function(output, scaled_dl).item()
                    output_pst = self.label_scaler.inverse_transform_tensor(output)
                    label_pst = self.label_scaler.inverse_transform_tensor(scaled_dl)
                    train_pst_loss += loss_function(output_pst, label_pst).item()
                train_pre_loss /= n_infer_batches
                train_pst_loss /= n_infer_batches

                all_train_pre_loss.append(train_pre_loss)
                all_train_pst_loss.append(train_pst_loss)

                # Validation — disk-loaded
                for valid_features, valid_label in valid_dataloader:
                    output = model(valid_features)
                    valid_pre_loss += loss_function(output, valid_label).item()
                    output_pst = self.label_scaler.inverse_transform_tensor(output)
                    label_pst = self.label_scaler.inverse_transform_tensor(valid_label)
                    valid_pst_loss += loss_function(output_pst, label_pst).item()

                valid_pre_loss /= len(valid_dataloader)
                valid_pst_loss /= len(valid_dataloader)

                all_valid_pre_loss.append(valid_pre_loss)
                all_valid_pst_loss.append(valid_pst_loss)

            if (epoch + 1) % 1 == 0:
                note_min = " *" if valid_pst_loss == min(all_valid_pst_loss) else ""
                logger.info(
                    f"Epoch {epoch + 1:<{n_epoch_digits}} | "
                    f"time: {time.time() - time_start:.4f} | "
                    f"batch: {batch_loss:.02e} | "
                    f"train_pre: {train_pre_loss:.02e} | "
                    f"train_pst: {train_pst_loss:.02e} | "
                    f"valid_pre: {valid_pre_loss:.02e} | "
                    f"valid_pst: {valid_pst_loss:.02e}"
                    f"{note_min}"
                )

            self.out_loss_record.append([
                epoch + 1, batch_loss,
                train_pre_loss, valid_pre_loss,
                train_pst_loss, valid_pst_loss,
            ])

            # is_best = (
            #     valid_pst_loss == min(all_valid_pst_loss)
            #     and epoch >= self.earliest_best
            # )
            # if is_best:
            #     with self.name_tracker.set_context("epoch", "best"):
            #         self.out_model.write(model=model, epoch="best")

            # condition_a = (epoch + 1) in self.extra_check
            # condition_b = (epoch + 1) % self.checkpoint == 0
            # if condition_a or condition_b:
            #     with self.name_tracker.set_context("epoch", epoch + 1):
            #         self.out_model.write(
            #             model=model,
            #             optimizer=optimizer,
            #             best_loss=min(all_valid_pst_loss),
            #             epoch=epoch + 1,
            #         )

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

    def init_model(self):
        model_dict = dict(
            arch_str=self.fcn_string,
            n_in=hp.nside2npix(self.nside),
            n_out=len(self.ells))
        model = FCNFromString(**model_dict)
        logger.info(f"Model dict: {model_dict}")
        return model

    def set_up_valid_dataset(self, split: Split) -> FixedDataset:
        feat_path = self.make_fn_template(split, self.in_valid_map)
        trgt_path = self.make_fn_template(split, self.in_valid_trgt_asset)

        dataset = FixedDataset(
            n_sims=split.n_sims,
            label_path=trgt_path,
            label_handler=self.in_valid_trgt_asset.handler,
            feature_path=feat_path,
            feature_handler=self.in_valid_map.handler,
            lmax=self.lmax,
            feature_scaler=self.map_processor,
            label_scaler=self.label_scaler,
            dtype=self.dtype,
            fit_scalers=False,
            device=self.device,
        )
        return dataset


    def _crosscheck_model_dims(self, model):
            expected_npix = hp.nside2npix(self.nside)
            expected_out  = len(self.ells)

            linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
            if not linears:
                logger.warning("No Linear layers found; skipping model dim cross-check.")
                return

            first_in = linears[0].in_features
            # In pixel-pair mode the input is expanded to npix**2 before the first
            # Linear; otherwise it's npix directly.
            expected_in = expected_npix ** 2 if model.pixel_pair else expected_npix
            actual_out = linears[-1].out_features

            if first_in != expected_in:
                mode = "pixel-pair (npix**2)" if model.pixel_pair else "plain (npix)"
                raise ValueError(
                    f"Model input dim {first_in} != expected {expected_in} "
                    f"[{mode}, nside={self.nside}, npix={expected_npix}]."
                )
            if actual_out != expected_out:
                raise ValueError(
                    f"Model output dim {actual_out} != n_label_ells={expected_out}."
                )
            logger.info(
                f"Model dim cross-check OK: in={first_in}, out={actual_out} "
                f"(npix={expected_npix}, pixel_pair={model.pixel_pair})."
            )