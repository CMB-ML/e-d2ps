from typing import Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from cmbml.core import Asset


@dataclass(frozen=True)
class SpectrumAssetNames:
    theory: str = "cmb_ps_thry"
    realization: str = "cmb_ps_real"

    def get(self, choice: str) -> str:
        if choice in {"theory", "thry"}:
            return self.theory
        elif choice in {"realization", "real", "realized"}:
            return self.realization
        raise ValueError(
            f"Unsupported choice: {choice}. "
            "Supported choices are 'theory'/'thry' and 'realization'/'real'."
        )


DEFAULT_SPECTRUM_NAMES = SpectrumAssetNames()


class SpectrumAsset(Asset, ABC):
    @staticmethod
    def select(
        assets: Dict[str, Asset],
        choice: str,
        name_map: SpectrumAssetNames = DEFAULT_SPECTRUM_NAMES,
    ) -> "SpectrumAsset":
        asset = assets[name_map.get(choice)]

        if choice in {"theory", "thry"}:
            return TheorySpectrumAsset.create_from(asset)

        elif choice in {"realization", "real", "realized"}:
            return RealizationSpectrumAsset.create_from(asset)

        raise ValueError(
            f"Unsupported choice: {choice}. "
            "Supported choices are 'theory'/'thry' and 'realization'/'real'."
        )

    @classmethod
    def create_from(cls, asset):
        new = cls.__new__(cls)
        new.__dict__.update(asset.__dict__)
        return new

    @property
    @abstractmethod
    def spectrum_kind(self) -> str:
        raise NotImplementedError


class TheorySpectrumAsset(SpectrumAsset):
    @property
    def spectrum_kind(self) -> str:
        return "theory"


class RealizationSpectrumAsset(SpectrumAsset):
    @property
    def spectrum_kind(self) -> str:
        return "realization"