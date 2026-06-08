import logging

# from tqdm import tqdm
import time

# import healpy as hp
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
# from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

# from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_ps import (
    SpectrumAsset, 
    SpectrumAssetNames,
    # norm_ps_type
)

# from d2ps_fcn.models.fcn_from_string import FCNFromString
# from d2ps_fcn.scaler_hydra_helper import make_scaler_from_prep_spec
# from d2ps_nn.utils.px_scale import make_px_processor, MapProcessor
from d2ps_fcn.dataset_otf import TrainingMapSampler, make_map_sampler
from d2ps_fcn.dataset_fixed import FixedDataset
from d2ps_fcn.realization_spectra import make_real_spectra_maker
from d2ps_fcn.stage_executors._base_otf_executors import BaseOTFExecutor


logger = logging.getLogger(__name__)


def _fmt_diag(vals):
    """Format a 1D tensor/array of diagnostic values as compact 4-char fields.

    Produces strings like ' 1.00', ' 0.93', '-0.10', ' nan ' so columns line up.
    """
    out = []
    for v in vals:
        v = float(v)
        if not np.isfinite(v):
            out.append(" nan")
        elif abs(v) >= 10:
            out.append(f"{v:+.0f}")  # overflow guard
        else:
            out.append(f"{v:+.2f}")
    return " ".join(out)


class TrainFixedExecutor(BaseOTFExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train_otf") -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Assets
        self.in_map:           Asset = self.assets_in["valid_map"]  # Confusing name. Disregard.
        self.in_map_stats:     Asset = self.assets_in["cmb_stats"]
        # self.in_rps_stats:     Asset = self.assets_in["rps_stats"]
        self.in_initial_model: Asset = self.assets_in["initial_model"]
        self.in_resume_model:  Asset = self.assets_in["resume_model"]
        self.in_lut:           Asset = self.assets_in["lut"]

        self.out_model:        Asset = self.assets_out["model"]
        self.out_loss_record:  Asset = self.assets_out["loss_record"]

        trgt_names = SpectrumAssetNames(
            theory="valid_thry_ps",
            realization="valid_real_ps",
        )
        self.in_trgt_asset = SpectrumAsset.select(
            self.assets_in,
            self.spectrum_type,
            name_map=trgt_names,
        )

        # Config
        self.train_batch_size = cfg.model.train.train_batch_size
        self.valid_batch_size = cfg.model.train.valid_batch_size
        self.n_batches        = cfg.model.train.n_batches
        self.n_epochs         = cfg.model.train.n_epochs
        self.learning_rate    = cfg.model.train.learning_rate
        self.earliest_best    = cfg.model.train.earliest_best
        self.checkpoint       = cfg.model.train.checkpoint_every
        self.extra_check      = cfg.model.train.extra_check

        logger.info(f"map_processor type: {type(self.map_processor).__name__}")
        logger.info(f"label_scaler: {self.label_scaler}")

        self.diag_lmin        = cfg.diag_lmin
        self.diag_lmax_print  = cfg.diag_lmax_print
        self.diag_lmax_record = cfg.diag_lmax_record
        self._diag_ells_print,  self._diag_cols_print  = self._build_diag_indices(self.diag_lmax_print)
        self._diag_ells_record, self._diag_cols_record  = self._build_diag_indices(self.diag_lmax_record)

    def _build_diag_indices(self, ell_max):
        ells_arr = np.asarray(self.ells)
        ells, cols = [], []
        for ell in range(self.diag_lmin, ell_max + 1):
            matches = np.where(ells_arr == ell)[0]
            if len(matches):
                ells.append(ell)
                cols.append(int(matches[0]))
        return ells, cols

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.choose_device()

        logger.info(f"Running for {self.n_epochs} epochs.")

        if self._diag_cols_record:
            logger.info(
                f"Diagnostics: recording R(ell), rho(ell) for ell="
                f"{self._diag_ells_record} at output cols {self._diag_cols_record}"
            )
            logger.info(
                f"Diagnostics: printing ell={self._diag_ells_print}"
            )
        else:
            logger.warning("Diagnostic ell range not present in model outputs; skipping R/rho prints.")

        # Load bank and fit scalers
        logger.debug(f"Loading bank of Dls")
        dl_bank = self.load_dl_bank()
        self.fit_label_scaler(dl_bank)
        logger.info(f"Done loading bank of Dls")
    
        # Build generation pipeline
        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

        # Training dataset (disk-based, existing machinery)
        train_dataset = self.set_up_fixed_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        train_infer_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
        )

        # Validation dataset (disk-based, existing machinery)
        valid_dataset = self.set_up_fixed_dataset(valid_split)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
        )

        logger.info(f"Training on {len(train_dataset)} and valid on {len(valid_dataset)}")

        # Model init
        model = self.init_model()
        context = dict(model_num=self.model_num)
        with self.name_tracker.set_contexts(context):
            self.in_initial_model.read(model=model, strict=False)
        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")
        model = model.to(self.device)

        loss_function = torch.nn.MSELoss(reduction='mean').to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        loss_record_headers = [
            'Epoch', 'Batch Loss',
            'Training Pre Loss',
            'Validation Pre Loss', 
            'Training Loss',
            'Validation Loss', 
            # 'Learning Rate',
        ]
        for ell in self._diag_ells_record:
            loss_record_headers.append(f'R_ell{ell}')
        for ell in self._diag_ells_record:
            loss_record_headers.append(f'rho_ell{ell}')
        for ell in self._diag_ells_record:
            loss_record_headers.append(f'FVE_ell{ell}')
        if self._diag_ells_record:
            loss_record_headers.append('locked')
            loss_record_headers.append('score')
        self.out_loss_record.write(data=loss_record_headers)

        all_batch_loss = []
        all_train_pre_loss = []
        all_valid_pre_loss = []
        all_train_pst_loss = []
        all_valid_pst_loss = []

        n_epoch_digits = len(str(self.n_epochs))
        best_save_loss = np.inf       # loss at the last "best" write
        last_best_epoch = -1          # epoch index of the last "best" write

        for epoch in range(self.n_epochs):
            time_start = time.time()
            # Training — on-the-fly generated maps
            model.train()
            batch_loss = 0.0
            for maps, scaled_dl in train_dataloader:
                optimizer.zero_grad(set_to_none=True)
                output = model(maps)
                loss = loss_function(output, scaled_dl)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            batch_loss /= len(train_dataloader)
            all_batch_loss.append(batch_loss)

            # Evaluation
            model.eval()
            train_pre_loss = train_pst_loss = 0.0
            valid_pre_loss = valid_pst_loss = 0.0

            # Accumulate validation predictions/labels (post-scaler) for R, rho diagnostics
            diag_pred_chunks = []
            diag_true_chunks = []

            with torch.no_grad():
                # Train-space estimate: sample n_infer_batches batches from sampler
                train_pre_loss = 0.0
                train_pst_loss = 0.0
                # for _ in range(n_infer_batches):
                for maps, scaled_dl in train_infer_dataloader:
                    output = model(maps)
                    train_pre_loss += loss_function(output, scaled_dl).item()
                    output_pst = self.label_scaler.inverse_transform_tensor(output)
                    label_pst = self.label_scaler.inverse_transform_tensor(scaled_dl)
                    train_pst_loss += loss_function(output_pst, label_pst).item()
                
                train_pre_loss /= len(train_infer_dataloader)
                train_pst_loss /= len(train_infer_dataloader)

                all_train_pre_loss.append(train_pre_loss)
                all_train_pst_loss.append(train_pst_loss)

                # Validation — disk-loaded
                for valid_features, valid_label in valid_dataloader:
                    output = model(valid_features)
                    valid_pre_loss += loss_function(output, valid_label).item()
                    output_pst = self.label_scaler.inverse_transform_tensor(output)
                    label_pst = self.label_scaler.inverse_transform_tensor(valid_label)
                    valid_pst_loss += loss_function(output_pst, label_pst).item()

                    if self._diag_cols_record:
                        diag_pred_chunks.append(output_pst.detach())
                        diag_true_chunks.append(label_pst.detach())

                valid_pre_loss /= len(valid_dataloader)
                valid_pst_loss /= len(valid_dataloader)

                all_valid_pre_loss.append(valid_pre_loss)
                all_valid_pst_loss.append(valid_pst_loss)

                # Compute R(ell), rho(ell), FVE(ell) over the full validation set
                diag_R = diag_rho = diag_fve = None
                diag_locked = diag_score = None
                if self._diag_cols_record and diag_pred_chunks:
                    pred_all = torch.cat(diag_pred_chunks, dim=0)  # (N_valid, n_ells)
                    true_all = torch.cat(diag_true_chunks, dim=0)
                    cols = torch.tensor(self._diag_cols_record, device=pred_all.device, dtype=torch.long)
                    p = pred_all.index_select(1, cols).float()
                    t = true_all.index_select(1, cols).float()

                    # Sample std along realizations (unbiased, matches numpy default)
                    p_std = p.std(dim=0, unbiased=True)
                    t_std = t.std(dim=0, unbiased=True)
                    eps = 1e-12
                    diag_R = (p_std / (t_std + eps)).cpu().numpy()

                    p_c = p - p.mean(dim=0, keepdim=True)
                    t_c = t - t.mean(dim=0, keepdim=True)
                    num = (p_c * t_c).sum(dim=0)
                    den = torch.sqrt((p_c.pow(2).sum(dim=0) * t_c.pow(2).sum(dim=0)).clamp_min(eps))
                    diag_rho = (num / den).cpu().numpy()

                    # FVE(ell) = 1 - MSE(ell) / Var_true(ell)
                    #   per-ell MSE across realizations, divided by per-ell true variance.
                    # Equals 1 for perfect prediction, 0 for constant-mean predictor,
                    # negative if worse than the mean. Computed from MSE directly
                    # (not via 2 R rho - R^2) so it stays exact when pred/true means differ.
                    mse_ell = (p - t).pow(2).mean(dim=0)
                    true_var = t_std.pow(2)
                    diag_fve = (1.0 - mse_ell / (true_var + eps)).cpu().numpy()

                    # Scalar summaries
                    diag_locked = int((diag_fve > 0.5).sum())          # count of "locked-in" ells
                    diag_score = float(np.clip(diag_fve, 0.0, None).sum())  # continuous progress

            row = [
                epoch + 1, 
                batch_loss,
                train_pre_loss, 
                valid_pre_loss,
                train_pst_loss, 
                valid_pst_loss,
            ]
            if self._diag_ells_record:
                if diag_R is not None:
                    row.extend(float(v) for v in diag_R)
                    row.extend(float(v) for v in diag_rho)
                    row.extend(float(v) for v in diag_fve)
                    row.append(int(diag_locked))
                    row.append(float(diag_score))
                else:
                    # Pad with NaN so column count stays consistent.
                    # 3 per-ell columns (R, rho, FVE) + 2 scalars (locked, score).
                    row.extend([float('nan')] * (3 * len(self._diag_ells_record)))
                    row.extend([float('nan'), float('nan')])
            self.out_loss_record.append(row)

            note_save_interval = ""
            note_save_best = ""

            in_final_window = (self.n_epochs - epoch) <= self.checkpoint

            is_new_best = (
                valid_pst_loss == min(all_valid_pst_loss)
                and epoch >= self.earliest_best
            )
            if is_new_best:
                halved = valid_pst_loss <= 0.5 * best_save_loss
                stale = (epoch - last_best_epoch) >= self.checkpoint
                if halved or stale or in_final_window:
                    with self.name_tracker.set_context("epoch", "best"):
                        self.out_model.write(model=model, epoch="best")
                    best_save_loss = valid_pst_loss
                    last_best_epoch = epoch
                    note_save_best = " B"

            condition_a = (epoch + 1) in self.extra_check
            condition_b = (epoch + 1) % self.checkpoint == 0
            if condition_a or condition_b:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(
                        model=model,
                        optimizer=optimizer,
                        best_loss=min(all_valid_pst_loss),
                        epoch=epoch + 1,
                    )
                note_save_interval = " I"

            if (epoch + 1) % 1 == 0:
                note_min = " *" if valid_pst_loss == min(all_valid_pst_loss) else ""
                diag_str = ""
                if diag_R is not None and self._diag_cols_print:
                    # indices into the record arrays that correspond to print ells
                    print_idx = [self._diag_ells_record.index(e) for e in self._diag_ells_print]
                    ell_hdr = " ".join(f"  l{e}" for e in self._diag_ells_print)
                    diag_str = (
                        f" | ell:{ell_hdr}"
                        f" | R: {_fmt_diag(diag_R[print_idx])}"
                        f" | p: {_fmt_diag(diag_rho[print_idx])}"
                        f" | lk={diag_locked} sc={diag_score:.2f}"
                    )
                logger.info(
                    f"Epoch {epoch + 1:<{n_epoch_digits}} | "
                    f"time: {time.time() - time_start:.2f} | "
                    f"batch: {batch_loss:.02e} | "
                    f"train_pre: {train_pre_loss:.02e} | "
                    f"train_pst: {train_pst_loss:.02e} | "
                    f"valid_pre: {valid_pre_loss:.02e} | "
                    f"valid_pst: {valid_pst_loss:.02e}"
                    f"{diag_str}"
                    f"{note_min}"
                    f"{note_save_best}"
                    f"{note_save_interval}"
                )

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

    def set_up_fixed_dataset(self, split: Split) -> FixedDataset:
        feat_path = self.make_fn_template(split, self.in_map)
        trgt_path = self.make_fn_template(split, self.in_trgt_asset)

        dataset = FixedDataset(
            n_sims=split.n_sims,
            label_path=trgt_path,
            label_handler=self.in_trgt_asset.handler,
            feature_path=feat_path,
            feature_handler=self.in_map.handler,
            lmax=self.lmax,
            feature_scaler=self.map_processor,
            map_shaper=self.map_shaper,
            label_scaler=self.label_scaler,
            dtype=self.dtype,
            fit_scalers=False,
            device=self.device,
        )
        return dataset
