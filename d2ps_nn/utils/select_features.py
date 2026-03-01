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
