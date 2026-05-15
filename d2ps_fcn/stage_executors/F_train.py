import logging

from tqdm import tqdm
import time

import healpy as hp
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import NumpyMap, AppendingCsvHandler
from cmbml.torch.pytorch_model_handler import PyTorchModel
from cmbml.torch.pytorch_executor_base import BasePyTorchModelExecutor

from d2ps_nn.utils.parse_ell_list import parse_ells
from d2ps_nn.utils.select_features import FeatureAsset
from d2ps_nn.utils.select_ps import SpectrumAsset

from d2ps_fcn.dataset import TrainDataset
from d2ps_fcn.models.fcn import FCNDouble, FCNSingle
from d2ps_fcn.scaler_hydra_helper import make_scaler_from_prep_spec


logger = logging.getLogger(__name__)


class TrainFCNModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str="train") -> None:
        super().__init__(cfg, stage_str=stage_str)

        # Output shape
        self.ells = parse_ells(cfg.model.use_ells)

        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        # self.in_trgt_asset = self.assets_in["cmb_ps"]
        self.in_trgt_asset = SpectrumAsset.select(self.assets_in,
                                                  cfg.spectrum_type)
        # TODO: Fix this for cross-compatibility
        self.in_feat_asset = FeatureAsset.select(self.assets_in, 
                                                 cfg.features_type)
        # self.in_feat_asset = self.assets_in["cmb_alms"]
        self.in_resume_model: Asset = self.assets_in["resume_model"]
        self.in_initial_model: Asset = self.assets_in["initial_model"]
        in_trgt_handler: NumpyMap
        in_feat_handler: NumpyMap
        in_pos_enc_lut_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.input_kind = cfg.input_kind
        self.lmax = self.ells[-1]

        self.n_sims_train_infer = cfg.get("n_sims_train_infer", None)
        self.out_power          = cfg.out_power
        self.activation         = cfg.model.activation
        self.hidden_features    = cfg.n_hidden_features
        self.batch_size         = cfg.model.train.batch_size
        self.learning_rate      = cfg.model.train.learning_rate
        self.n_epochs           = cfg.model.train.n_epochs
        self.dtype              = self.dtype_mapping[cfg.model.dtype]
        self.earliest_best      = cfg.model.train.earliest_best
        self.checkpoint         = cfg.model.train.checkpoint_every
        self.extra_check        = cfg.model.train.extra_check
        self.model_init         = cfg.model.model_init
        self.restart_epoch      = None
        self.random_init        = False

        self.label_prep         = cfg.model.prep_label
        self.feature_prep       = cfg.model.prep_feature
        self.label_scaler       = None
        self.feature_scaler     = None
        self.device             = None

    def execute(self):
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.choose_device()

        self.label_scaler = make_scaler_from_prep_spec(self.label_prep)
        self.feature_scaler = make_scaler_from_prep_spec(self.feature_prep)

        train_split = self.splits.get("train")
        valid_split = self.splits.get("valid")

        train_dataset = self.set_up_dataset(
            train_split,
            fit_scalers=True,
        )
        train_dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=True
            )
        logger.info(f"Train dataloader: {train_dataloader_params}")
        train_dataloader = DataLoader(train_dataset, **train_dataloader_params)

        train_infer_dataset = self.set_up_dataset(
            train_split,
            early_stop=self.n_sims_train_infer,
            fit_scalers=False,
        )
        train_infer_dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=False,
        )
        logger.info(f"Train dataloader: {train_infer_dataloader_params}")
        train_infer_dataloader = DataLoader(train_infer_dataset, **train_infer_dataloader_params)

        valid_dataset = self.set_up_dataset(
            valid_split,
            fit_scalers=False,
        )
        valid_dataset = self.set_up_dataset(valid_split)
        valid_dataloader_params = dict(
            batch_size=self.batch_size, 
            shuffle=False,
        )
        logger.info(f"Valid dataloader: {valid_dataloader_params}")        
        valid_dataloader = DataLoader(valid_dataset, **valid_dataloader_params)

        model = self.init_model()
        model = model.to(self.device)

        loss_function = torch.nn.MSELoss(reduction='mean')
        loss_function = loss_function.to(self.device)
        logger.info(f"Loss: torch.nn.MSELoss")

        optim_dict = dict(lr=self.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), **optim_dict)
        logger.info(f"Optimizer: {optimizer.__class__.__name__}, {optim_dict}")

        do_loss_record = "new"
        start_epoch = 0
        if True:
            logger.info(f"Loading transfer-initialization model.")
            context = dict(model_num=self.model_init)
            with self.name_tracker.set_contexts(context):
                self.in_initial_model.read(model=model, strict=True)
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
        model = model.to(self.device)

        with torch.no_grad():
            if self.input_kind in ["alms", "alm"]:
                n_input = hp.Alm.getsize(self.lmax)
            elif self.input_kind in ["maps", "map"]:
                n_input = hp.nside2npix(self.nside)
            else:
                raise NotImplementedError("Must use input_kind of either 'alms' or 'maps'")

            torch.manual_seed(0)
            random_tensor = torch.rand((1, n_input)).to(self.device)
            logger.info(f"Consistency check, 'map': {random_tensor[0, :5]}")
            logger.info(f"Consistency check, 'pred': {model(random_tensor)[0, :5]}")

        all_batch_loss = []
        all_train_pre_loss = []
        all_valid_pre_loss = []
        all_train_pst_loss = []
        all_valid_pst_loss = []
        start_epoch = 0

        if do_loss_record == "continue":
            self.out_loss_record.start()  # New lines will be added, naively
                                          #  Naively = appended; if resuming, manual cleaning may be needed.
        else:
            loss_record_headers = ['Epoch', 
                                   'Batch Loss', 
                                   'Validation Pre Loss', 
                                   'Training Pre Loss', 
                                   'Validation Loss', 
                                   'Training Loss', 
                                   'Learning Rate',
                                   'Mean Grad Norm',
                                   'Max Grad Norm']
            self.out_loss_record.write(data=loss_record_headers)

        n_epoch_digits = len(str(self.n_epochs))

        # print(model)

        # print("Configured activation:", self.cfg.model.activation)
        # print("Actual activation modules:")
        # for name, module in model.named_modules():
        #     if isinstance(module, (torch.nn.ReLU, torch.nn.GELU)):
        #         print(name, module)

        for epoch in range(start_epoch, self.n_epochs):
            # Training
            epoch_grad_norm = []
            # with tqdm(train_dataloader, desc=f"Ep {epoch + 1:<{n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
            model.train()
            batch_loss = 0
            for train_features, train_label in train_dataloader:
                # train_features = train_features.to(device=self.device, dtype=self.dtype)
                # train_label = train_label.to(device=self.device, dtype=self.dtype)
                optimizer.zero_grad()
                output = model(train_features)
                batch_train_loss = loss_function(output, train_label)
                batch_train_loss.backward()
                optimizer.step()
                # pbar.set_postfix({'Loss': batch_train_loss.item()})
                batch_loss += batch_train_loss.item()
            batch_loss /= len(train_dataloader)
            all_batch_loss.append(batch_loss)

            # Validation
            model.eval()
            train_pre_loss = 0
            valid_pre_loss = 0
            train_pst_loss = 0
            valid_pst_loss = 0

            with torch.no_grad():
                for train_features, train_label in train_infer_dataloader:
                    # train_features = train_features.to(device=self.device, dtype=self.dtype)
                    # train_label = train_label.to(device=self.device, dtype=self.dtype)

                    output = model(train_features)

                    train_pre_loss += loss_function(output, train_label).item()

                    output_pst = self.label_scaler.inverse_transform_tensor(output)
                    label_pst = self.label_scaler.inverse_transform_tensor(train_label)
                    train_pst_loss += loss_function(output_pst, label_pst).item()

                train_pre_loss /= len(train_infer_dataloader)
                train_pst_loss /= len(train_infer_dataloader)

                all_train_pre_loss.append(train_pre_loss)
                all_train_pst_loss.append(train_pst_loss)

                for valid_features, valid_label in valid_dataloader:
                    # valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                    # valid_label = valid_label.to(device=self.device, dtype=self.dtype)

                    output = model(valid_features)

                    valid_pre_loss += loss_function(output, valid_label).item()

                    output_pst = self.label_scaler.inverse_transform_tensor(output)
                    label_pst = self.label_scaler.inverse_transform_tensor(valid_label)
                    valid_pst_loss += loss_function(output_pst, label_pst).item()

                valid_pre_loss /= len(valid_dataloader)
                valid_pst_loss /= len(valid_dataloader)

                all_valid_pre_loss.append(valid_pre_loss)
                all_valid_pst_loss.append(valid_pst_loss)

                note_min_loss = " *" if valid_pst_loss == min(all_valid_pst_loss) else ""

                if (epoch+1) % 20 == 0:
                    logger.info(
                        f"Epoch {epoch + 1:<{n_epoch_digits}} | "
                        f"train_pre: {train_pre_loss:.02e} | "
                        f"train_pst: {train_pst_loss:.02e} | "
                        f"valid_pre: {valid_pre_loss:.02e} | "
                        f"valid_pst: {valid_pst_loss:.02e}"
                        f"{note_min_loss}"
                    )

            record_line = [
                epoch + 1,
                batch_loss,
                train_pre_loss,
                valid_pre_loss,
                train_pst_loss,
                valid_pst_loss,
            ]
            self.out_loss_record.append(record_line)

            is_best_condition_a = valid_pst_loss == min(all_valid_pst_loss)
            is_best_condition_b = epoch >= self.earliest_best
            if is_best_condition_a and is_best_condition_b:
                with self.name_tracker.set_context("epoch", "best"):
                    self.out_model.write(model=model,
                                         epoch="best"
                                         )

            # Checkpoint every so many epochs
            condition_a = (epoch + 1) in self.extra_check
            condition_b = (epoch + 1) % self.checkpoint == 0
            if (condition_a or condition_b):
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         best_loss=min(all_valid_pst_loss),
                                         epoch=epoch + 1)
            
        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

    def init_model(self):
        if self.hidden_features == 0:
            model_dict = dict(
                out_channels=len(self.ells),
                out_power = self.out_power
            )
            model = FCNSingle(**model_dict)
            model_dict["model"] = "FCNSingle"
        else:
            model_dict = dict(
                mid_channels=self.hidden_features,
                out_channels=len(self.ells),
                out_power = self.out_power,
                activation = self.activation
            )
            model = FCNDouble(**model_dict)
            model_dict["model"] = "FCNDouble"
        logger.info(f"Model dict: {model_dict}")
        
        return model

    def set_up_dataset(
        self,
        split: Split,
        early_stop: int = None,
        fit_scalers: bool = False,
    ) -> TrainDataset:
        
        trgt_path = self.make_fn_template(split, self.in_trgt_asset)
        feat_path = self.make_fn_template(split, self.in_feat_asset)

        dataset_params = dict(
            n_sims=split.n_sims if early_stop is None else early_stop,
            label_path=trgt_path,
            label_handler=self.in_trgt_asset.handler,
            feature_path=feat_path,
            feature_handler=self.in_feat_asset.handler,
            feature_scaler=self.feature_scaler,
            label_scaler=self.label_scaler,
            dtype=self.dtype,
            fit_scalers=fit_scalers,
            device=self.device
        )

        logger.info(f"Dataset: {dataset_params}")

        dataset = TrainDataset(**dataset_params)
        return dataset
