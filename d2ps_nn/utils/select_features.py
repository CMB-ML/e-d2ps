from typing import Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from cmbml.core import Asset
from cmbml.utils.planck_instrument import Instrument


@dataclass(frozen=True)
class FeatureAssetNames:
    cmb: str = "cmb_map"
    obs: str = "obs_maps"

    def get(self, choice: str) -> str:
        if choice == "cmb":
            return self.cmb
        elif choice == "obs":
            return self.obs
        raise ValueError(
            f"Unsupported choice: {choice}. Supported choices are 'cmb' and 'obs'."
        )


DEFAULT_NAMES = FeatureAssetNames()


class FeatureAsset(Asset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def select(
        assets: Dict[str, Asset],
        choice: str,
        instrument: Instrument = None,
        name_map: FeatureAssetNames = DEFAULT_NAMES,
    ) -> "FeatureAsset":
        asset = assets[name_map.get(choice)]

        if choice == "cmb":
            return CMBFeatureAsset.create_from(asset)

        elif choice == "obs":
            res = ObsFeatureAsset.create_from(asset)
            res.instrument = instrument
            return res

        raise ValueError(
            f"Unsupported choice: {choice}. Supported choices are 'cmb' and 'obs'."
        )

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
        CMB features are single maps, so return an iterator with a single element.
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
