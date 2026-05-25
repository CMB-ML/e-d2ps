import logging

# from tqdm import tqdm
import time

import healpy as hp
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
# from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
# from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
# from d2ps_nn.utils.select_features import FeatureAsset
# from d2ps_nn.utils.select_ps import SpectrumAsset
from d2ps_nn.utils.select_ps import (
    SpectrumAsset, 
    SpectrumAssetNames,
    norm_ps_type
)

from d2ps_fcn.fixed_dataset import FixedDataset
# from d2ps_fcn.models.fcn import FCNDouble, FCNSingle
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
        # self.in_valid_thry_ps: Asset = self.assets_in["valid_thry_ps"]
        # self.in_valid_real_ps: Asset = self.assets_in["valid_real_ps"]
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
        # self.hidden_features  = cfg.n_hidden_features
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
        # self.feature_scaler = make_scaler_from_prep_spec(self.feature_prep)
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

        # dl_for_labels = dl_bank[:, 2:]              # shape (n_spectra, lmax-1), ells 2..lmax
        # self.label_scaler.fit(dl_for_labels)

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

        # sampler = TrainingMapSampler(
        #     inner=make_map_sampler(
        #         sim_engine=self.sht_engine,
        #         dl_bank=dl_bank,
        #         nside=self.nside,
        #         batch_size=self.train_batch_size,
        #         n_batches=self.n_batches,
        #         lmax=self.lmax,
        #         device=self.device,
        #         dtype=self.dtype,
        #     ),
        #     map_processor=self.map_processor,
        #     label_scaler=self.label_scaler
        # )

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

        # # --- DIAGNOSTIC: realization label sanity (remove after verifying) ---
        # import numpy as np  # local import; remove with the block
        # maps_dbg, scaled_dbg = sampler.sample()
        # label_pst_dbg = self.label_scaler.inverse_transform_tensor(scaled_dbg)
        # label_pst_np = label_pst_dbg.detach().cpu().numpy()  # (B, n_ells)

        # # theory Dl the maps were drawn from, sliced to label ells, for comparison
        # # (inner sampler draws from dl_bank; we don't have the exact rows here,
        # #  so compare against the bank's mean theory Dl as a reference curve)
        # bank_mean_tdl = dl_bank[:, 2:].mean(axis=0)  # (n_ells,) mean theory Dl

        # ell_axis = np.array(self.ells)
        # print("=== realization label sanity ===")
        # print(f"spectrum_type      : {self.spectrum_type}")
        # print(f"self.label_prep    : {self.label_prep}")
        # print(f"label_pst shape    : {label_pst_np.shape}")
        # print(f"recovered Dl mean  : {label_pst_np.mean(axis=0)[:8]} ... (first 8 ells)")
        # print(f"bank theory Dl mean: {bank_mean_tdl[:8]} ... (first 8 ells)")
        # print(f"recovered Dl range : [{label_pst_np.min():.3e}, {label_pst_np.max():.3e}]")
        # print(f"theory Dl range    : [{bank_mean_tdl.min():.3e}, {bank_mean_tdl.max():.3e}]")
        # # peak location check: argmax of the mean recovered curve vs theory
        # print(f"recovered peak ell : {ell_axis[label_pst_np.mean(axis=0).argmax()]}")
        # print(f"theory peak ell    : {ell_axis[bank_mean_tdl.argmax()]}")
        # return
        # # --- END DIAGNOSTIC ---

        # # --- DIAGNOSTIC: train/valid scaled-label agreement at init (remove after) ---
        # model.eval()
        # with torch.no_grad():
        #     # one train batch
        #     maps_t, scaled_t = sampler.sample()
        #     out_t = model(maps_t)
        #     train_pre_dbg = loss_function(out_t, scaled_t).item()

        #     # one valid batch
        #     vf, vl = next(iter(valid_dataloader))
        #     out_v = model(vf)
        #     valid_pre_dbg = loss_function(out_v, vl).item()

        #     print("=== train/valid pre-loss at init ===")
        #     print(f"self.label_prep    : {self.label_prep}")
        #     print(f"train scaled-label mean/std: {scaled_t.mean().item():.4f} / {scaled_t.std().item():.4f}")
        #     print(f"valid scaled-label mean/std: {vl.mean().item():.4f} / {vl.std().item():.4f}")
        #     print(f"train_pre (untrained): {train_pre_dbg:.4e}")
        #     print(f"valid_pre (untrained): {valid_pre_dbg:.4e}")
        #     print(f"ratio v/t            : {valid_pre_dbg / train_pre_dbg:.3f}")
        # return
        # # --- END DIAGNOSTIC ---

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

    # def set_up_valid_dataset(self, split: Split) -> FixedDataset:
    #     feat_path = self.make_fn_template(split, self.in_valid_map)
    #     trgt_path = self.make_fn_template(split, self.in_valid_thry_ps)

    #     dataset = FixedDataset(
    #         n_sims=split.n_sims,
    #         label_path=trgt_path,
    #         label_handler=self.in_valid_thry_ps.handler,
    #         feature_path=feat_path,
    #         feature_handler=self.in_valid_map.handler,
    #         lmax=self.lmax,
    #         feature_scaler=self.map_processor,
    #         label_scaler=self.label_scaler,
    #         dtype=self.dtype,
    #         fit_scalers=False,
    #         device=self.device,
    #     )
    #     return dataset

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
