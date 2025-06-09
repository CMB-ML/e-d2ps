from typing import Dict
from abc import ABC, abstractmethod
from cmbml.core import Asset
from cmbml.utils.planck_instrument import Instrument


class FeatureAsset(Asset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def select(assets_in: Dict[str, Asset], 
               choice: str, 
               instrument: Instrument = None) -> 'FeatureAsset':
        if choice == "cmb":
            asset = assets_in["cmb_map"]
            res = CMBFeatureAsset.__new__(CMBFeatureAsset)
            res.__dict__.update(asset.__dict__)
        elif choice == "obs":
            asset = assets_in["obs_maps"]
            res = ObsFeatureAsset.__new__(ObsFeatureAsset)
            res.__dict__.update(asset.__dict__)
            res.instrument = instrument
        else:
            raise ValueError(f"Unsupported choice: {choice}. Supported choices are 'cmb' and 'obs'.")
        return res

    @classmethod
    def create_from(cls, asset):
        new = cls.__new__(cls)
        new.__dict__.update(asset.__dict__)
        return new

    @property
    @abstractmethod
    def n_features(self):
        raise NotImplementedError("Subclasses must implement n_features property.")

    @abstractmethod
    def feature_iter(self):
        raise NotImplementedError("Subclasses must implement feature_iter method.")


class CMBFeatureAsset(FeatureAsset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def n_features(self):
        return 1

    def feature_iter(self):
        """
        CMB features are single maps, so we return an iterator with a single element.
        """
        yield "cmb", self

class ObsFeatureAsset(FeatureAsset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = None  # To be set on creation in .select()

    @property
    def n_features(self):
        return len(self.instrument.dets) if hasattr(self, 'instrument') else 0

    def feature_iter(self):
        """
        Obs features are maps for each detector, this sets the needed context.
        """
        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                yield freq, self

# class Features(ABC):
#     def __init__(self, cfg, assets_in, namer, instrument=None):
#         """
#         Initialize the Features class.

#         :param cfg: Configuration object containing feature settings.
#         :param assets_in: Input assets required for feature extraction.
#         :param instrument: Optional instrument object for observation features.
#         """
#         self.cfg = cfg
#         self.assets_in = assets_in
#         self.namer = namer
#         self.instrument = instrument

#         # self.feature to be set in subclasses BEFORE calling super().__init__()
#         self.handler = self.feature.handler

#     @classmethod
#     def create(self, cfg, assets_in, instrument=None):
#         if cfg.features_type is None:
#             raise ValueError("features_type must be specified in the configuration.")
#         elif cfg.features_type in ["cmb"]:
#             return CMBFeatures(cfg, assets_in)
#         elif cfg.features_type in ["obs"]:
#             if instrument is None:
#                 raise ValueError("Instrument must be provided for obs features.")
#             return ObsFeatures(cfg, instrument, assets_in)
#         else:
#             raise ValueError(f"Unsupported features_type: {cfg.features_type}. "
#                              "Supported types are 'cmb' and 'obs'.")

#     @abstractmethod
#     def get_features(self, sim_idx):
#         """
#         Abstract method to get features for a given simulation index.
#         Must be implemented by subclasses.
#         """
#         pass


# class CMBFeatures(Features):
#     def __init__(self, cfg, assets_in):
#         self.feature = assets_in["cmb_map"]
#         super().__init__(cfg, assets_in)

#     @property
#     def n_features(self):
#         return 1


# class ObsFeatures(Features):
#     def __init__(self, cfg, assets_in, namer, instrument: Instrument):
#         self.feature = assets_in["obs_maps"]
#         super().__init__(cfg, assets_in, namer, instrument)
    
#     @property
#     def n_features(self):
#         return len(self.instrument.dets)
