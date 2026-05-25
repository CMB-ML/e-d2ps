from typing import Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from cmbml.core import Asset


# Canonical kinds and their accepted aliases — defined once, used everywhere.
THEORY = "theory"
REALIZATION = "realization"

_THEORY_ALIASES = {"theory", "thry"}
_REALIZATION_ALIASES = {"realization", "real", "realized"}


def norm_ps_type(choice: str) -> str:
    """Map any accepted alias to a canonical kind: 'theory' | 'realization'."""
    if choice in _THEORY_ALIASES:
        return THEORY
    if choice in _REALIZATION_ALIASES:
        return REALIZATION
    raise ValueError(
        f"Unsupported choice: {choice!r}. "
        f"Supported choices are {sorted(_THEORY_ALIASES | _REALIZATION_ALIASES)}."
    )


@dataclass(frozen=True)
class SpectrumAssetNames:
    theory: str = "cmb_ps_thry"
    realization: str = "cmb_ps_real"

    def get(self, choice: str) -> str:
        kind = norm_ps_type(choice)
        if kind == THEORY:
            return self.theory
        return self.realization  # kind == REALIZATION (normalize guarantees one of the two)


DEFAULT_SPECTRUM_NAMES = SpectrumAssetNames()


class SpectrumAsset(Asset, ABC):
    @staticmethod
    def select(
        assets: Dict[str, Asset],
        choice: str,
        name_map: SpectrumAssetNames = DEFAULT_SPECTRUM_NAMES,
    ) -> "SpectrumAsset":
        kind = norm_ps_type(choice)
        asset = assets[name_map.get(choice)]

        if kind == THEORY:
            return TheorySpectrumAsset.create_from(asset)
        return RealizationSpectrumAsset.create_from(asset)  # kind == REALIZATION

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
        return THEORY


class RealizationSpectrumAsset(SpectrumAsset):
    @property
    def spectrum_kind(self) -> str:
        return REALIZATION
